"""Bounded axial divider kernel, matching the CPU derivative-bound reference.

Returns -1 for unresolved work/precision instead of silently reporting a miss.
The caller must terminate/diagnose that sample, never skip that interface.
"""
DIVIDER_GLSL = r'''
// dims=(eye_length, eye_curve, twist, phase), local radius-unit frame.
vec4 studio_divider_eval(vec3 p, float angle, vec4 dims) {
    float t=p.y/dims.x, x=p.x-dims.y*(1.0-t*t);
    float a=dims.w+dims.z*t+angle,c=cos(a),s=sin(a);
    float da=dims.z/dims.x, dxdy=2.0*dims.y*p.y/(dims.x*dims.x);
    return vec4(-s,-p.z*s*da-dxdy*s-x*c*da,c,p.z*c-x*s);
}
int studio_divider_root(vec3 ro,vec3 rd,float angle,vec4 dims,
                        float tmin,float tmax,float tolerance,out float root,out vec3 normal) {
    root=1e30;normal=vec3(0);
    if(tmax<=tmin)return 0;
    vec2 stack[32];int sp=1;stack[0]=vec2(tmin,tmax);
    float da=dims.z*rd.y/dims.x;
    float xsecond=2.0*dims.y*rd.y*rd.y/(dims.x*dims.x);
    for(int work=0;work<1024 && sp>0;++work) {
        vec2 ab=stack[--sp];float lo=ab.x,hi=ab.y;
        if(lo>=root)continue;
        hi=min(hi,root);float mid=(lo+hi)*.5,h=(hi-lo)*.5;
        vec3 p=ro+rd*mid;float y=p.y/dims.x;
        float x=p.x-dims.y*(1.0-y*y);
        float xp=rd.x+2.0*dims.y*p.y*rd.y/(dims.x*dims.x);
        float xmax=abs(x)+abs(xp)*h+abs(xsecond)*h*h*.5;
        float zmax=abs(p.z)+abs(rd.z)*h;
        float xpmax=abs(xp)+abs(xsecond)*h;
        float second=2.0*abs(rd.z*da)+zmax*da*da+abs(xsecond)+2.0*xpmax*abs(da)+xmax*da*da;
        vec4 ev=studio_divider_eval(p,angle,dims);float df=dot(ev.xyz,rd);
        // Cushion the interval inequalities against float32 rounding. This
        // still needs GPU cancellation/grazing validation before activation.
        float cushion=1e-6*(abs(ev.w)+abs(df)*h+second*h*h)+1e-12;
        if(abs(ev.w)>(abs(df)+second*h)*h+cushion)continue;
        float fl=studio_divider_eval(ro+rd*lo,angle,dims).w;
        float fh=studio_divider_eval(ro+rd*hi,angle,dims).w;
        if(abs(df)>second*h+cushion) {
            if(fl==0.0)root=min(root,lo);
            else if(fh==0.0)root=min(root,hi);
            else if((fl<0.0)!=(fh<0.0)) {
                for(int step=0;step<48 && hi-lo>tolerance;++step) {
                    float m=(lo+hi)*.5;
                    if(m==lo || m==hi)break;
                    float fm=studio_divider_eval(ro+rd*m,angle,dims).w;
                    if((fm<0.0)==(fl<0.0)){lo=m;fl=fm;}else hi=m;
                }
                root=min(root,(lo+hi)*.5);
            }
            continue;
        }
        if(hi-lo<=tolerance || mid==lo || mid==hi || sp>29)return -1;
        stack[sp++]=vec2(mid,hi);stack[sp++]=vec2(lo,mid);
    }
    if(sp>0)return -1;
    if(root>=1e29)return 0;
    normal=normalize(studio_divider_eval(ro+rd*root,angle,dims).xyz);
    return 1;
}
'''

def pack_bundle_records(records, bundle, *, scale_m, center_m, host_triangle=0):
    """Extend binding15 with four explicit affine/recipe rows, no extra SSBO."""
    import numpy as np
    if bundle is None:return records
    r=np.asarray(records,np.float32).copy();ids=bundle['regions']
    if not 2<=len(ids)<=32 or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<i<len(r) or r[i,0]!=1 for i in ids):
        raise ValueError('bundle must name distinct glass rod regions')
    frame=bundle['frame'];axes=np.asarray(frame['axes'],np.float64)
    origin=np.asarray(center_m,np.float64)+float(scale_m)*np.asarray(frame['origin'],np.float64)
    scale=float(scale_m)*float(frame['scale'])
    if axes.shape!=(3,3) or not np.isfinite(axes).all() or scale<=0 or not np.isfinite(scale):raise ValueError('invalid bundle affine frame')
    if not np.allclose(axes@axes.T,np.eye(3),atol=1e-6,rtol=0) or np.linalg.det(axes)<0:raise ValueError('bundle axes must be proper orthonormal basis')
    inverse=axes/scale
    extra=np.zeros((4,4),np.float32);extra[:3,:3]=inverse;extra[:3,3]=-inverse@origin
    extra[3]=[bundle[k] for k in ('eye_length','eye_curve','twist','phase')]
    if not np.isfinite(extra).all() or extra[3,0]<=0:raise ValueError('invalid bundle parameters')
    r[0,2:]=[1,len(r)]
    for index,ri in enumerate(ids):r[ri,2]=index+1
    if type(host_triangle) is not int or host_triangle<0:raise ValueError('invalid host representative')
    diagnostic=np.zeros((1,4),np.float32);diagnostic[0,1]=host_triangle
    return np.concatenate((r,extra,diagnostic))


def region_record_count(records):
    if records is None:return 0
    return int(records[0,3]) if len(records) and records[0,2]==1 else len(records)


BUNDLE_GLSL = r'''
bool studio_has_bundle(){return uStudioRegionCount>0 && studio_regions[0].z==1.0;}
vec3 studio_bundle_local(vec3 p){
    return vec3(dot(studio_regions[uStudioRegionCount],vec4(p,1)),
                dot(studio_regions[uStudioRegionCount+1],vec4(p,1)),
                dot(studio_regions[uStudioRegionCount+2],vec4(p,1)));
}
vec3 studio_bundle_direction(vec3 d){
    return vec3(dot(studio_regions[uStudioRegionCount].xyz,d),
                dot(studio_regions[uStudioRegionCount+1].xyz,d),
                dot(studio_regions[uStudioRegionCount+2].xyz,d));
}
int studio_bundle_owner(vec3 world){
    vec3 p=studio_bundle_local(world);vec4 dims=studio_regions[uStudioRegionCount+3];
    float t=p.y/dims.x,x=p.x-dims.y*(1.0-t*t);int count=0;
    for(int i=1;i<uStudioRegionCount;++i)if(studio_regions[i].z>0.0)++count;
    float best=-1.;int owner=-1;
    for(int i=1;i<uStudioRegionCount;++i)if(studio_is_active(i) && studio_regions[i].z>0.0){
        float a=dims.w+dims.z*t+3.141592653589793*(studio_regions[i].z-1.)/float(count);
        float score=abs(x*cos(a)+p.z*sin(a));
        if(score>best){best=score;owner=i;}
    }
    return owner;
}
bool studio_partition_hit=false;
vec3 studio_partition_normal=vec3(0);
int studio_partition_owner_triangle=-1;
int studio_partition_nearest(vec3 ro,vec3 rd,inout float best){
    if(!studio_has_bundle() || !studio_is_active(0))return -1;
    int count=0,occupied=0;int rod_index[32];
    for(int i=1;i<uStudioRegionCount;++i){
        if(studio_regions[i].z>0.0){
            int index=int(studio_regions[i].z)-1;if(index<0 || index>=32)return -2;
            rod_index[index]=i;++count;if(studio_is_active(i))++occupied;
        }
        if(studio_is_active(i) && int(studio_regions[i].x)==3)return -1;
    }
    if(occupied<2)return -1;
    vec3 local=studio_bundle_local(ro),dir=studio_bundle_direction(rd);
    vec4 dims=studio_regions[uStudioRegionCount+3];
    float epsilon=studio_region_epsilon(ro)/max(length(rd),1e-30);
    float lower=epsilon,upper=best;
    if(abs(dir.y)>1e-20){vec2 span=vec2(-dims.x-local.y,dims.x-local.y)/dir.y;
        lower=max(lower,min(span.x,span.y));upper=min(upper,max(span.x,span.y));}
    else if(abs(local.y)>dims.x)return -1;
    if(upper<=lower || upper>=1e29)return -1;
    bool found=false;vec3 bestnormal=vec3(0);
    for(int i=0;i<count;++i)for(int j=0;j<i;++j)for(int turn=0;turn<2;++turn){
        // Only simultaneously occupied, distinct materials can form an
        // optical cut before the nearest old-volume boundary.
        int ri=rod_index[i],rj=rod_index[j];
        if(!studio_is_active(ri) || !studio_is_active(rj) ||
           studio_regions[ri].y==studio_regions[rj].y)continue;
        float angle=3.141592653589793*float(i+j)/(2.*float(count))+float(turn)*1.570796326794897;
        float start=lower;
        bool exhausted=true;
        for(int retry=0;retry<32 && start<min(upper,best);++retry){
            float hit;vec3 normal;
            int status=studio_divider_root(local,dir,angle,dims,start,min(upper,best),epsilon,hit,normal);
            if(status<0)return -2;
            if(status==0){exhausted=false;break;}
            vec3 p=ro+rd*hit;float e=studio_region_epsilon(p)/max(length(rd),1e-30);
            int a=studio_bundle_owner(ro+rd*(hit-e)),b=studio_bundle_owner(ro+rd*(hit+e));
            if(a>=0 && b>=0 && studio_regions[a].y!=studio_regions[b].y){
                best=hit;bestnormal=normal;found=true;exhausted=false;break;
            }
            start=hit+2.*e;
        }
        if(exhausted && start<min(upper,best))return -2;
    }
    if(!found)return -1;
    vec3 n=studio_regions[uStudioRegionCount].xyz*bestnormal.x+
           studio_regions[uStudioRegionCount+1].xyz*bestnormal.y+
           studio_regions[uStudioRegionCount+2].xyz*bestnormal.z;
    studio_partition_normal=normalize(n);
    int ti=int(studio_regions[uStudioRegionCount+4].y);
    if(ti<0 || ti>=geom_tris.length() || studio_region_id(geom_tris[ti].v0.w)!=0)return -2;
    studio_partition_owner_triangle=ti;return ti;
}
'''


def read_bundle_failures(ssbo, count):
    """Explicit unresolved-root diagnostic, shared by forward/camera dispatches."""
    import numpy as np
    from OpenGL import GL
    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,int(ssbo))
    raw=GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,(int(count)+4)*16,16)
    return int(np.frombuffer(raw,np.uint32)[0])
