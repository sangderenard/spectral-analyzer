"""Opt-in observation of unchanged canonical sensor continuation, never radiance.

Rows retain the existing shared band-0 path limitation. They are not a field
estimate, a completeness certificate, or an image. No GL imports occur here.
"""
VERSION = 'native-camera-intervals-v1'
WORDS = 128
BINDING = 22


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('canonical camera observer seam missing or ambiguous: ' + old[:80])
    return source.replace(old, new, 1)


def observe_camera_continuation(source, *, enabled=False):
    """Specialize only explicitly opted-in, fully assembled native sensor source.

    Binding22: uint header[2] (attempted rows, overflow), then 128 uint words/row.
    Capacity is explicitly bound by owner; zero capacity writes no records.
    Header must be reset before dispatch. Owner retains exact source, scene,
    camera, sample seed and field identities alongside returned bytes.
    """
    if not enabled:
        return source
    if 'StudioCameraIntervals' in source:
        raise ValueError('camera observer already installed')
    if 'binding=22)' in source or 'binding = 22)' in source:
        raise ValueError('camera observer binding22 is occupied')
    regions = 'uint studio_active[32]' in source
    spheres = 'bool studio_sphere_hit(' in source
    partitions = 'bool studio_partition_hit=false;' in source
    if spheres:
        source = _once(source, 'int nearest_filtered(', 'int sci_selected_sphere=-1;\nint nearest_filtered(')
        source = _once(source, 'best = 1e30; int best_tri = -1;', 'sci_selected_sphere=-1;best = 1e30; int best_tri = -1;')
        source = _once(source, '{best=ht;best_tri=ti;}', '{best=ht;best_tri=ti;sci_selected_sphere=si;}')
        source = _once(source, 'best = ht; best_tri = ti;', 'best = ht; best_tri = ti;sci_selected_sphere=-1;')

    pre = 'for(int j=0;j<32;++j)sci_words[o+48u+uint(j)]=studio_active[j];' if regions else ''
    post = 'for(int j=0;j<32;++j)sci_words[o+80u+uint(j)]=studio_active[j];' if regions else ''
    helpers = r'''
layout(std430,binding=22) buffer StudioCameraIntervals { uint sci_words[]; };
uniform uint uCameraIntervalCapacity;
uint sci_begin(ivec2 pixel,uint invocation,int sample_id,int bounce,vec3 ro,vec3 rd,float distance,
               int hit,int medium,int bands,int active_band,int first_hit,float tp[8]) {
    uint row=atomicAdd(sci_words[0],1u);
    uint cap=min(uCameraIntervalCapacity,uint(max(sci_words.length()-2,0))/128u);
    if(row>=cap){atomicOr(sci_words[1],1u);return 0xffffffffu;}
    uint o=2u+row*128u;
    for(uint j=0u;j<128u;++j)sci_words[o+j]=0u;
    sci_words[o]=uint(pixel.x);sci_words[o+1u]=uint(pixel.y);
    sci_words[o+15u]=invocation;
    sci_words[o+2u]=uint(sample_id);sci_words[o+3u]=uint(bounce);
    sci_words[o+4u]=uint(hit);sci_words[o+5u]=uint(medium);
    sci_words[o+115u]=uint(first_hit);
    sci_words[o+6u]=uint(bands);sci_words[o+7u]=uint(uFixed3Transport);
    for(int j=0;j<3;++j){sci_words[o+8u+uint(j)]=floatBitsToUint(ro[j]);sci_words[o+12u+uint(j)]=floatBitsToUint(rd[j]);}
    // A miss has no finite geometry interval; distance is explicitly unknown.
    sci_words[o+11u]=hit>=0?floatBitsToUint(distance):0x7fc00000u;
    for(int j=0;j<bands;++j){
        sci_words[o+16u+uint(j)]=floatBitsToUint(tp[j]);
        sci_words[o+24u+uint(j)]=floatBitsToUint(uFixed3Transport!=0?studio_band_frequency(j):uLayerCentersHz[j]);
    }
    sci_words[o+32u]=hit< -1?6u:(medium==-2?5u:(hit==-1?3u:0u));
    sci_words[o+33u]=uint(medium);
    // Boundary direction is selected with band0, not a per-band trace.
    sci_words[o+38u]=floatBitsToUint(uFixed3Transport!=0?studio_band_frequency(max(active_band,0)):uLayerCentersHz[max(active_band,0)]);
    sci_words[o+40u]=active_band<0?((1u<<uint(bands))-1u):(1u<<uint(active_band));
    __PRE__
    return o;
}
void sci_boundary(uint o,bool ok,vec3 rd,int medium,float factor) {
    if(o==0xffffffffu)return;
    sci_words[o+32u]=ok?1u:2u;sci_words[o+33u]=uint(medium);
    for(int j=0;j<3;++j)sci_words[o+34u+uint(j)]=floatBitsToUint(rd[j]);
    sci_words[o+37u]=floatBitsToUint(factor);
    __POST__
}
'''.replace('__PRE__', pre + ('sci_words[o+39u]=uint(uStudioRegionCount);' if regions else '')).replace('__POST__', post)
    source = _once(source, 'void main() {', helpers + '\nvoid main() {')
    source = _once(source, 'int currentMedium=studio_origin_medium(ro,rd);', 'int sci_first_hit=-1;vec3 sci_first_position=vec3(0.0);uvec4 sci_first_metadata=uvec4(0u);int sci_first_kind=-1,sci_first_owner=-1,sci_first_stage=-1;\n        int currentMedium=studio_origin_medium(ro,rd);')
    seam = 'int   hit_id = nearest_hit(ro, rd, surf_t);'
    band = 'studio_field_band' if 'int studio_field_band=' in source else '-1'
    count = '(uFixed3Transport!=0?3:nl)' if band != '-1' else 'nl'
    primitive = ('(studio_partition_hit?2:' if partitions else '') + ('(sci_selected_sphere>=0?1:0)' if spheres else '0') + (')' if partitions else '')
    owner = ('(studio_partition_hit?studio_partition_owner_triangle:' if partitions else '') + ('sci_selected_sphere' if spheres else '-1') + (')' if partitions else '')
    source = _once(source, seam, seam + f'\n            int sci_kind={primitive},sci_owner={owner};\n            bool sci_first_interval=sci_first_hit<0 && hit_id>=0 && sci_kind!=2;\n            if(sci_first_interval){{sci_first_hit=hit_id;sci_first_kind=sci_kind;sci_first_owner=sci_owner;}}\n            uint sci_row=sci_begin(sensor_px,linear_id,s,bounce,ro,rd,hit_id>=0?surf_t:0.0,hit_id,currentMedium,{count},{band},sci_first_hit,throughput);\n            if(sci_row!=0xffffffffu){{sci_words[sci_row+41u]=uint(sci_first_stage);sci_words[sci_row+124u]=uint(sci_first_kind);sci_words[sci_row+125u]=uint(sci_first_owner);sci_words[sci_row+126u]=uint(sci_kind);sci_words[sci_row+127u]=uint(sci_owner);}}')
    source = _once(source, 'vec3  hp       = ro + rd * surf_t;', 'vec3  hp       = ro + rd * surf_t;\n            if(sci_first_interval)sci_first_position=hp;\n            if(sci_row!=0xffffffffu && sci_first_hit>=0){for(int j=0;j<3;++j)sci_words[sci_row+116u+uint(j)]=floatBitsToUint(sci_first_position[j]);sci_words[sci_row+119u]=1u;}')
    tri_seam = 'Tri   tri      = studio_hit_tri(hit_id,hp);' if 'Tri   tri      = studio_hit_tri(hit_id,hp);' in source else 'Tri   tri      = load_tri(hit_id);'
    source = _once(source, tri_seam, tri_seam + '\n            if(sci_first_interval)sci_first_metadata=uvec4(floatBitsToUint(tri.emissive.z),floatBitsToUint(tri.normal.w),floatBitsToUint(tri.v0.w),floatBitsToUint(tri.e1.w));\n            if(sci_row!=0xffffffffu)for(int j=0;j<4;++j)sci_words[sci_row+120u+uint(j)]=sci_first_metadata[j];')
    def stage_marker(stage):
        return f'if(sci_first_interval)sci_first_stage={stage};if(sci_row!=0xffffffffu)sci_words[sci_row+41u]=uint(sci_first_stage);'
    source = _once(source, 'if(studio_boundary(tri)) {', 'if(studio_boundary(tri)) {' + stage_marker(2))
    source = _once(source, 'if (opacity < 0.999) {', 'if (opacity < 0.999) {' + stage_marker(2))
    source = _once(source, 'float survival = spectral_reflector ?', stage_marker(1) + 'float survival = spectral_reflector ?')
    source = _once(source, 'if(tri.mat_out.w>=0.999)break;', 'if(tri.mat_out.w>=0.999){'+stage_marker(1)+'break;}')
    source = _once(source, 'if (opacity >= 0.999) break;', 'if (opacity >= 0.999) {'+stage_marker(1)+'break;}')
    if 'if(studio_pbr_opaque(tri)) {' in source:
        source = _once(source, 'if(studio_pbr_opaque(tri)) {', 'if(studio_pbr_opaque(tri)) {'+stage_marker(1))
    old = 'if(!studio_boundary_step(tri,rd,frequency,rand01(rng),currentMedium,factor)) break;'
    source = _once(source, old, 'bool sci_ok=studio_boundary_step(tri,rd,frequency,rand01(rng),currentMedium,factor);\n                sci_boundary(sci_row,sci_ok,rd,currentMedium,factor);\n                if(!sci_ok) break;')
    source = _once(source, 'if (hit_id < 0 || tp_max < 0.003) break;', 'if (hit_id < 0 || tp_max < 0.003) {\n                if(sci_row!=0xffffffffu){sci_words[sci_row+42u]=floatBitsToUint(tp_max);sci_words[sci_row+43u]=1u;if(hit_id>=0)sci_words[sci_row+32u]=4u;}\n                break;\n            }')
    source = _once(source, 'if (tp_after < 0.005) break;', 'if (tp_after < 0.005) {if(sci_row!=0xffffffffu){sci_words[sci_row+32u]=7u;sci_words[sci_row+42u]=floatBitsToUint(tp_after);sci_words[sci_row+43u]=2u;}break;}')
    return source

class CameraIntervalObserver:
    """Borrow caller-owned storage; never allocate, dispatch, or render.

    Caller resets/readbacks the two-word header and rows, and retains identity.
    Counters include overflow so truncation cannot masquerade as complete data.
    """
    def __init__(self, buffer, capacity, *, scene_revision, camera_revision):
        if type(buffer) is not int or buffer <= 0 or type(capacity) is not int or not 0 < capacity < 2**24:
            raise ValueError('observer needs a positive buffer and bounded row capacity')
        if not isinstance(scene_revision, str) or not scene_revision or not isinstance(camera_revision, str) or not camera_revision:
            raise ValueError('observer needs exact scene and camera revision identities')
        self.buffer, self.capacity = buffer, capacity
        self.identity = dict(version=VERSION, scene_revision=scene_revision,
                             camera_revision=camera_revision, shared_band0_path=True,
                             row_words=WORDS, required_bytes=8 + capacity * WORDS * 4)

    def bind(self, program):
        from OpenGL import GL
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer)
        size = int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_BUFFER_SIZE))
        if size < self.identity['required_bytes']:
            raise ValueError('camera interval storage is shorter than declared capacity')
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, BINDING, self.buffer)
        GL.glUniform1ui(GL.glGetUniformLocation(program, 'uCameraIntervalCapacity'), self.capacity)


def plan_interval_dispatch(*, width, rows, samples, path_bands, max_bounces,
                           memory_budget_bytes, reserved_bytes=0, row_granularity=1, response_copies=False):
    """Bound GPU storage plus one CPU readback; no empirical evidence cap."""
    values = (width, rows, samples, path_bands, max_bounces, memory_budget_bytes, row_granularity)
    if any(type(v) is not int or v <= 0 for v in values) or type(reserved_bytes) is not int or reserved_bytes < 0:
        raise ValueError('invalid observer dispatch budget')
    per_row = width * samples * path_bands * (max_bounces + 1)
    available = memory_budget_bytes - reserved_bytes
    buffers = 4 if response_copies else 2
    fit = max(0, (available // buffers - 8) // (WORDS * 4 * per_row))
    selected = rows
    if rows % row_granularity or fit < rows:
        raise MemoryError('original dispatch strip cannot fit camera interval budget without changing its rays')
    capacity = selected * per_row
    if capacity >= 2**24:
        raise MemoryError('camera interval dispatch exceeds bounded counter capacity')
    size = 8 + capacity * WORDS * 4
    return dict(rows=selected, capacity=capacity, storage_bytes=size,
                working_bytes=buffers*size, reserved_bytes=reserved_bytes,
                total_peak_bytes=reserved_bytes+buffers*size)
