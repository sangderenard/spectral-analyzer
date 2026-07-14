#version 430 core

/* Spectral VCM merge pass.  This implements the balance-heuristic form of
 * Georgiev et al.'s combined VC/VM estimator in the renderer's existing
 * deferred-normalization convention: spectral betas carry the unnormalised
 * measurement product and the complete compatible-technique density sum is
 * divided exactly once here. */
layout(local_size_x = 64) in;

#define LGV_STRIDE 56u
#define CGV_STRIDE 72u
#define LGV_BETA 16u
#define CGV_BETA 22u
#define LGV_DIR 48u
#define CGV_DIR 54u
#define LGV_EF 54u
#define LGV_EB 55u
#define CGV_EF 60u
#define CGV_EB 61u
#define MAX_BANDS 32
#define MAX_CHAIN 32
#define MAX_EDGES 31
#define MAX_NODES 33
#define PDF_DELTA_SPECULAR (1u << 2)
#define MAT_APERTURE_STOP 128u
#define VCM_PI 3.14159265358979323846

layout(std430, binding=0) readonly buffer LightVertices { float lv[]; };
layout(std430, binding=1) readonly buffer CameraVertices { float cv[]; };
layout(std430, binding=2) coherent buffer PixelAccum { uint pixels[]; };
layout(std430, binding=3) readonly buffer Params {
    float min_geom; float sensor_half_w; float sensor_half_h; uint cam_offset;
    uint n_light_verts; uint n_cam_verts; int sensor_res; uint light_batch_size;
    uint light_offset; int n_bands; int tile_x0; int tile_y0; int tile_w; int tile_h;
    uint light_sample_stride; float light_sample_weight;
};
layout(std430, binding=4) readonly buffer SpectralWeights { float rgb_w[]; };
layout(std430, binding=10) readonly buffer GridHeads { uint heads[]; };
layout(std430, binding=11) readonly buffer GridNext { uint next_link[]; };
layout(std430, binding=12) coherent buffer VcmDebug { uint dbg[]; };

uniform uint hash_mask;
uniform float merge_radius;
uniform int vcm_n_mats;
uniform int dispatch_base;

uint hash_cell(ivec3 c) {
    uint h = uint(c.x) * 0x8da6b343u;
    h ^= uint(c.y) * 0xd8163841u;
    h ^= uint(c.z) * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return h & hash_mask;
}

bool finite_pos(float x) { return x > 0.0f && !isnan(x) && !isinf(x); }

void atomic_add_float(uint idx, float val) {
    if (!finite_pos(val)) return;
    uint expected = pixels[idx];
    for (;;) {
        uint desired = floatBitsToUint(uintBitsToFloat(expected) + val);
        uint actual = atomicCompSwap(pixels[idx], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

void splat(float sy, float sz, vec3 value) {
    int tw = tile_w > 0 ? tile_w : sensor_res;
    int th = tile_h > 0 ? tile_h : sensor_res;
    int tx = tile_w > 0 ? tile_x0 : 0;
    int ty = tile_h > 0 ? tile_y0 : 0;
    float fy = (sy + sensor_half_w) * float(sensor_res)/(2.0f*sensor_half_w) - 0.5f;
    float fz = (sz + sensor_half_h) * float(sensor_res)/(2.0f*sensor_half_h) - 0.5f;
    int iy0 = int(floor(fy)), iz0 = int(floor(fz));
    float sumw = 0.0f;
    for (int dy=0; dy<2; ++dy) for (int dz=0; dz<2; ++dz) {
        int iy=iy0+dy, iz=iz0+dz;
        if (iy>=ty && iy<ty+th && iz>=tx && iz<tx+tw)
            sumw += max(0.0f,1.0f-abs(float(iy)-fy))*max(0.0f,1.0f-abs(float(iz)-fz));
    }
    if (sumw <= 0.0f) return;
    uint plane = uint(tw*th);
    for (int dy=0; dy<2; ++dy) for (int dz=0; dz<2; ++dz) {
        int iy=iy0+dy, iz=iz0+dz;
        if (iy<ty || iy>=ty+th || iz<tx || iz>=tx+tw) continue;
        float w=max(0.0f,1.0f-abs(float(iy)-fy))*max(0.0f,1.0f-abs(float(iz)-fz))/sumw;
        uint p=uint((iy-ty)*tw + (iz-tx));
        atomic_add_float(p, value.r*w); atomic_add_float(p+plane, value.g*w);
        atomic_add_float(p+2u*plane, value.b*w);
    }
}

bool chain_vertex(uint q, uint ci, uint cb, uint li, uint lb,
                  out uint vinfo, out uint flags, out uint pdf_flags, out uint blocked) {
    if (q <= ci) {
        uint b=(cb+q)*CGV_STRIDE;
        vinfo=floatBitsToUint(cv[b+9u]); flags=floatBitsToUint(cv[b+7u]);
        pdf_flags=floatBitsToUint(cv[b+17u]); blocked=floatBitsToUint(cv[b+18u]);
    } else {
        uint lq=li-(q-ci);
        uint b=(lb+lq)*LGV_STRIDE;
        vinfo=floatBitsToUint(lv[b+9u]); flags=floatBitsToUint(lv[b+7u]);
        pdf_flags=floatBitsToUint(lv[b+13u]); blocked=floatBitsToUint(lv[b+14u]);
    }
    return (vinfo>>31)!=0u && blocked==0u && (flags&MAT_APERTURE_STOP)==0u;
}

/* Density of every VC cut plus every VM weld for this extended path.  The
 * selected VM density is p_s(y)p_t(z) pi r^2.  Internal delta edges retain
 * their discrete probability masses; only a cut/weld located at a delta
 * endpoint is unavailable. */
bool vcm_density(uint ci, uint cb, uint li, uint lb, out float selected, out float denom) {
    uint N=ci+li+1u;
    if (N<3u || N>uint(MAX_CHAIN)) return false;
    float ef[MAX_EDGES], eb[MAX_EDGES];
    for (uint e=0u; e<N-1u; ++e) {
        if (e<ci) {
            uint b=(cb+e)*CGV_STRIDE; ef[e]=cv[b+CGV_EF]; eb[e]=cv[b+CGV_EB];
        } else {
            uint lparent=li-(e-ci)-1u;
            uint b=(lb+lparent)*LGV_STRIDE;
            ef[e]=lv[b+LGV_EB]; eb[e]=lv[b+LGV_EF];
        }
    }
    float pre[MAX_NODES], suf[MAX_NODES];
    pre[0]=1.0f;
    for (uint i=1u;i<N;++i) pre[i]=(pre[i-1u]>0.0f&&ef[i-1u]>0.0f)?max(pre[i-1u]*ef[i-1u],1e-30f):0.0f;
    suf[N-1u]=1.0f;
    for (int j=int(N)-2;j>=0;--j) suf[uint(j)]=(suf[uint(j)+1u]>0.0f&&eb[uint(j)]>0.0f)?max(suf[uint(j)+1u]*eb[uint(j)],1e-30f):0.0f;
    float area=VCM_PI*merge_radius*merge_radius;
    selected=pre[ci]*suf[ci]*area;
    if (!finite_pos(selected)) return false;
    denom=0.0f;
    /* Cut (vertex-connection) technique enumeration MUST match
     * candidate_strategy_density() in t5_full_connect.comp.glsl: cuts adjacent
     * to a delta vertex are retained with their discrete probability-mass
     * products (the delta-cut audit's documented policy).  A mismatched cut
     * set between the VC and VM denominators biases the combined estimator. */
    for (uint cut=1u;cut<N;++cut) {
        uint vi0,fl0,pf0,ob0,vi1,fl1,pf1,ob1;
        bool a=chain_vertex(cut-1u,ci,cb,li,lb,vi0,fl0,pf0,ob0);
        bool b=chain_vertex(cut,ci,cb,li,lb,vi1,fl1,pf1,ob1);
        if (a&&b) {
            float p=pre[cut-1u]*suf[cut]; if (finite_pos(p)) denom+=p;
        }
    }
    for (uint m=1u;m+1u<N;++m) {
        uint vi,fl,pf,ob;
        if (chain_vertex(m,ci,cb,li,lb,vi,fl,pf,ob) && (pf&PDF_DELTA_SPECULAR)==0u) {
            float p=pre[m]*suf[m]*area; if (finite_pos(p)) denom+=p;
        }
    }
    denom=max(denom,selected);
    return finite_pos(denom);
}

void main() {
    uint cidx=uint(dispatch_base)+gl_GlobalInvocationID.x;
    if (cidx>=n_cam_verts) return;
    uint cbv=cidx*CGV_STRIDE;
    uint cvinfo=floatBitsToUint(cv[cbv+9u]);
    uint ci=cvinfo&0xffffu;
    uint cpf=floatBitsToUint(cv[cbv+17u]);
    uint cflags=floatBitsToUint(cv[cbv+7u]);
    uint cblock=floatBitsToUint(cv[cbv+18u]);
    int cmat=floatBitsToInt(cv[cbv+21u]);
    if (ci==0u || (cvinfo>>31)==0u || cblock!=0u || cmat<0 || cmat>=vcm_n_mats ||
        (cflags&MAT_APERTURE_STOP)!=0u || (cpf&PDF_DELTA_SPECULAR)!=0u) return;
    atomicAdd(dbg[1],1u);
    vec3 cp=vec3(cv[cbv],cv[cbv+1u],cv[cbv+2u]);
    vec3 cn=normalize(vec3(cv[cbv+3u],cv[cbv+4u],cv[cbv+5u]));
    vec3 cdir=normalize(vec3(cv[cbv+CGV_DIR],cv[cbv+CGV_DIR+1u],cv[cbv+CGV_DIR+2u]));
    if (dot(cn,-cdir)<0.0f) cn=-cn;
    ivec3 center=ivec3(floor(cp/merge_radius));
    vec3 total=vec3(0.0f);
    uint cam_base=cidx-ci;
    for(int x=-1;x<=1;++x) for(int y=-1;y<=1;++y) for(int z=-1;z<=1;++z) {
        uint photon=heads[hash_cell(center+ivec3(x,y,z))];
        while(photon!=0xffffffffu) {
            atomicAdd(dbg[2],1u);
            uint lbv=photon*LGV_STRIDE;
            vec3 lp=vec3(lv[lbv],lv[lbv+1u],lv[lbv+2u]);
            ivec3 actual=ivec3(floor(lp/merge_radius));
            if (all(equal(actual,center+ivec3(x,y,z)))) {
                vec3 d=lp-cp;
                float plane=abs(dot(d,cn));
                vec3 tangent=d-cn*dot(d,cn);
                vec3 ln=normalize(vec3(lv[lbv+3u],lv[lbv+4u],lv[lbv+5u]));
                int lmat=floatBitsToInt(lv[lbv+10u]);
                if (lmat==cmat && plane<=0.125f*merge_radius &&
                    dot(tangent,tangent)<=merge_radius*merge_radius && abs(dot(cn,ln))>=0.8f) {
                    atomicAdd(dbg[3],1u);
                    uint lvinfo=floatBitsToUint(lv[lbv+9u]);
                    uint li=lvinfo&0xffffu;
                    if (photon>=li) {
                        float selected,denom;
                        if (vcm_density(ci,cam_base,li,photon-li,selected,denom)) {
                            vec3 ldir=normalize(vec3(lv[lbv+LGV_DIR],lv[lbv+LGV_DIR+1u],lv[lbv+LGV_DIR+2u]));
                            vec3 value=vec3(0.0f);
                            int nb=clamp(n_bands,1,MAX_BANDS);
                            for(int band=0;band<nb;++band) {
                                float a=cv[cbv+CGV_BETA+uint(band)];
                                float b=lv[lbv+LGV_BETA+uint(band)];
                                if (a<=0.0f||b<=0.0f) continue;
                                float f=meval_endpoint_response(cmat,band,cpf,cn,cdir,-ldir);
                                float v=a*b*f/denom;
                                if (!finite_pos(v)) continue;
                                value += v*vec3(rgb_w[band*3],rgb_w[band*3+1],rgb_w[band*3+2]);
                            }
                            if (value.r+value.g+value.b>0.0f) { total+=value; atomicAdd(dbg[4],1u); }
                        }
                    }
                }
            }
            photon=next_link[photon];
        }
    }
    if (total.r+total.g+total.b>0.0f) {
        splat(cv[cbv+13u],cv[cbv+14u],total);
        atomicAdd(dbg[5],1u);
    }
}
