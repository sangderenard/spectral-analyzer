"""Bounded, process-local observation of the actual Studio GLSL tracers.

This is a diagnostic composer, not a transport implementation. It changes no
ray, RNG draw, material rule, cutoff, scalar splat or directional deposit.
Forward ROI records have a reserved pool, so unrelated paths cannot evict them.
Camera records describe adjoint throughput, never emitted energy. Misses have
no finite endpoint and are never fabricated into transport segments.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

BINDING = 21
COUNTER_BUFFER_BINDING = 4
PENDING_COUNT_WORD = 4
PENDING_RAY_BUFFER_BINDING = 6
HEADER_WORDS = 256
EVENT_WORDS = 64
REASONS = {
    0: "continued", 1: "no_geometry_hit", 2: "boundary_step_rejected",
    3: "packet_below_0.002", 4: "absorber_surface", 5: "conductor_weight_nonpositive",
    6: "bounce_budget_exhausted", 7: "camera_throughput_below_0.003",
    8: "opaque_emitter", 9: "russian_roulette", 10: "camera_throughput_below_0.005",
}
COUNTERS = {
    0: "actual_hit_segments", 1: "roi_intersections", 2: "non_roi_events",
    3: "roi_records_dropped", 4: "non_roi_records_dropped", 5: "misses",
    6: "roi_valid_medium", 7: "roi_positive_entry_throughput",
    8: "deposit_attempts", 9: "positive_quantized_deposits",
    10: "roi_invalid_medium", 11: "roi_fixed3_disabled", 12: "roi_zero_root_weight",
    13: "roi_nonpositive_entry_throughput", 14: "recorder_calls",
    15: "terminated_paths", 16: "started_paths", 18: "counter_overflow",
    19: "quantization_overflow",
}


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("ray ledger seam missing or ambiguous: " + old[:90])
    return source.replace(old, new, 1)


def _block(source, signature):
    """Return a braced source region without counting braces in comments."""
    start = source.index(signature)
    opening = source.index("{", start)
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: " " * len(m[0]), source, flags=re.S)
    depth = 1
    end = opening + 1
    while depth and end < len(source):
        depth += (stripped[end] == "{") - (stripped[end] == "}")
        end += 1
    if depth:
        raise ValueError("unterminated tracer region")
    return start, end


_GLSL = r'''
// Diagnostic storage is disjoint from forward energy and sensor image storage.
layout(std430,binding=21) buffer StudioArrivalLedger { uint sal_words[]; };
uniform uint uArrivalPopulation;
uniform vec3 uArrivalMin;
uniform vec3 uArrivalMax;
int sal_slot=-1;
uint sal_ray=0u, sal_root=0u, sal_sample=0u;
int sal_bounce=0;
bool sal_roi=false;
void sal_count(int i) {
    if(atomicAdd(sal_words[i],1u)==0xffffffffu) atomicOr(sal_words[18],1u);
}
void sal_u(int i,uint v) { if(sal_slot>=0)sal_words[sal_slot+i]=v; }
void sal_f(int i,float v) { sal_u(i,floatBitsToUint(v)); }
void sal_v(int i,vec3 v) { sal_f(i,v.x);sal_f(i+1,v.y);sal_f(i+2,v.z); }
bool sal_clip(vec3 a,vec3 d,float length_m,out float first,out float last) {
    first=0.0;last=length_m;
    for(int k=0;k<3;++k) {
        if(d[k]==0.0) { if(a[k]<uArrivalMin[k] || a[k]>=uArrivalMax[k])return false; }
        else { float x=(uArrivalMin[k]-a[k])/d[k],y=(uArrivalMax[k]-a[k])/d[k];
            first=max(first,min(x,y));last=min(last,max(x,y)); }
    }
    return last>first;
}
void sal_begin(vec3 a,vec3 d,float distance,int hit,int medium,vec3 energy,
               float frequency,float root_weight,uint channel) {
    sal_slot=-1;sal_roi=false;
    float first=0.0,last=0.0;
    if(hit>=0) {
        sal_count(0);sal_count(64+min(sal_bounce,63));
        sal_roi=sal_clip(a,d,distance,first,last);
    } else sal_count(5);
    uint index;
    if(sal_roi) {
        index=atomicAdd(sal_words[1],1u);
        sal_count(128+min(sal_bounce,63));
        if(index<__ROI_CAP__u)sal_slot=256+int(index)*64;
        else sal_count(3);
        if(medium>=-1)sal_count(6);else sal_count(10);
        if(max(energy.x,max(energy.y,energy.z))>0.0)sal_count(7);else sal_count(13);
        if(uFixed3Transport==0)sal_count(11);
        if(channel==0u && root_weight<=0.0)sal_count(12);
    } else {
        index=atomicAdd(sal_words[2],1u);
        if(index<__OTHER_CAP__u)sal_slot=256+(__ROI_CAP__+int(index))*64;
        else sal_count(4);
    }
    if(sal_slot<0)return;
    for(int i=0;i<64;++i)sal_words[sal_slot+i]=0u;
    sal_u(0,channel);sal_u(1,sal_ray);sal_u(2,sal_root);sal_u(3,uint(uSeed));
    sal_u(4,uArrivalPopulation);sal_u(5,uint(sal_bounce));sal_u(6,uint(medium));
    sal_u(7,uint(hit));sal_u(11,sal_roi?1u:0u);
    if(hit>=0) {
        Tri item=load_tri(hit);sal_u(8,floatBitsToUint(item.emissive.z));
        sal_u(9,floatBitsToUint(item.normal.w));
    }
    sal_v(12,a);sal_f(15,hit>=0?distance:0.0);
    sal_v(16,hit>=0?a+d*distance:a);sal_f(19,frequency);sal_v(20,d);
    sal_f(23,first);sal_f(24,last);sal_f(25,energy.x);sal_f(29,root_weight);
    sal_u(30,(medium< -1?1u:0u)|(uFixed3Transport==0?2u:0u)|
        (channel==0u && root_weight<=0.0?4u:0u)|(energy.x<=0.0?8u:0u));
    sal_v(44,energy);sal_u(59,sal_sample);
    // Canonical helper selected by the actual material engine, no new Beer law.
    vec3 end_energy=energy,entry_energy=energy,exit_energy=energy;
    for(int i=0;i<3;++i) {
        float f=channel==0u?frequency:studio_band_frequency(i);
        if(hit>=0)end_energy[i]*=studio_medium_transmittance(medium,f,distance);
        if(sal_roi) {
            entry_energy[i]*=studio_medium_transmittance(medium,f,first);
            exit_energy[i]*=studio_medium_transmittance(medium,f,last);
        }
    }
    sal_f(26,end_energy.x);sal_f(27,entry_energy.x);sal_f(28,exit_energy.x);
    sal_v(47,end_energy);sal_v(50,entry_energy);sal_v(53,exit_energy);
    sal_u(60,(sal_roi && first==0.0?1u:0u)|(sal_roi && last==distance?2u:0u));
    sal_u(63,0x53414c32u);
}
void sal_finish(int reason,vec3 energy,vec3 outgoing,int medium) {
    sal_u(10,uint(reason));sal_f(31,energy.x);sal_v(32,outgoing);
    sal_u(35,uint(medium));sal_v(56,energy);
    if(reason!=0) { sal_count(15);sal_count(32+reason); }
}
void sal_deposit(float scaled,int quantized) {
    sal_count(8);if(quantized>0)sal_count(9);
    if(sal_slot<0)return;
    sal_words[sal_slot+36]+=1u;
    if(quantized>0)sal_words[sal_slot+37]+=1u;
    uint before=sal_words[sal_slot+38];
    uint after=before+uint(max(quantized,0));
    sal_words[sal_slot+38]=after;
    if(after<before) { sal_words[sal_slot+39]=1u;sal_count(19); }
    sal_f(40,uintBitsToFloat(sal_words[sal_slot+40])+max(scaled,0.0));
}
'''


def _header(roi_capacity, other_capacity):
    for value in (roi_capacity, other_capacity):
        if type(value) is not int or not 0 <= value <= 1048576:
            raise ValueError("ledger pool capacity must be an integer in0..1048576")
    if not roi_capacity:
        raise ValueError("ROI capacity must be positive")
    return _GLSL.replace("__ROI_CAP__", str(roi_capacity)).replace("__OTHER_CAP__", str(other_capacity))


def forward_shader(source, *, roi_capacity=131072, other_capacity=131072):
    """Compose after directional + internal-medium observer; only PASS_FORWARD.

    Pending secondaries are not dispatched by _gpu_pump_prebuilt. This composer
    rejects any other pass instead of assigning invented primary lineage.
    """
    if "studio_volume_path(" in source:
        raise ValueError("joint ExtinctionGrid attenuation ledger is not implemented")
    if "sal_begin(" in source:
        raise ValueError("ledger already installed")
    if "studio_record_segment(ro,hp,packet,currentMedium);" not in source:
        raise ValueError("ledger requires entry-packet internal directional observer")
    source = _once(source, "void studio_record_track(ivec3 q,vec3 d,vec4 packet,float meters) {",
                   _header(roi_capacity, other_capacity) + "\nvoid studio_record_track(ivec3 q,vec3 d,vec4 packet,float meters) {")
    source = _once(source, "int w=int(round(max(scaled,0.0)));",
                   "int w=int(round(max(scaled,0.0))); sal_deposit(scaled,w);")
    source = _once(source, "if(scaled>=2147483000.0){",
                   "if(scaled>=2147483000.0){sal_count(19);sal_deposit(scaled,0);")
    start, end = _block(source, "void main() {")
    body = source[start:end]
    body = _once(body, "int currentMedium=studio_origin_medium(ro,rd);",
                 "int currentMedium=studio_origin_medium(ro,rd);\n"
                 "    sal_ray=ray_id;sal_root=uint(_ray_idx);sal_count(16);")
    body = _once(body, "int hit = nearest_hit(ro, rd, best);",
                 "int hit = nearest_hit(ro, rd, best);\n"
                 "        sal_bounce=bounce;sal_begin(ro,rd,best,hit,currentMedium,vec3(packet.z),packet.x,studio_record_root_weight,0u);")
    body = _once(body, "studio_record_segment(ro,hp,packet,currentMedium);",
                 "sal_count(14);studio_record_segment(ro,hp,packet,currentMedium);")
    body = _once(body, "if(spectral_reflector)reflectivity=studio_reflector_rho(tri,packet.x,cos_i);",
                 "if(spectral_reflector)reflectivity=studio_reflector_rho(tri,packet.x,cos_i);\n"
                 "        sal_f(41,packet.z);sal_f(42,reflectivity);sal_f(43,absorption);sal_f(62,opacity);\n"
                 "        sal_u(61,(diffuse_surface?1u:0u)|(spectral_reflector?2u:0u)|(studio_rough_conductor(tri)?4u:0u));")
    finish = lambda reason: f"sal_finish({reason},vec3(packet.z),rd,currentMedium);"
    substitutions = [
        ("if (hit < 0) break;", "if (hit < 0) {" + finish(1) + "break;}"),
        ("if(!studio_boundary_step(tri,rd,packet.x,rand01(rng),currentMedium,factor)) break;",
         "if(!studio_boundary_step(tri,rd,packet.x,rand01(rng),currentMedium,factor)) {" + finish(2) + "break;}"),
        ("if ((mat_flags_bits & MAT_FLAG_ABSORBER) != 0u) break;",
         "if ((mat_flags_bits & MAT_FLAG_ABSORBER) != 0u) {" + finish(4) + "break;}"),
        ("if(weight<=0.0)break;", "if(weight<=0.0){" + finish(5) + "break;}"),
    ]
    for old, new in substitutions:
        body = _once(body, old, new)
    if body.count("if (packet.z < 0.002) break;") != 3:
        raise ValueError("forward throughput cutoffs changed")
    body = body.replace("if (packet.z < 0.002) break;",
                        "if (packet.z < 0.002) {" + finish(3) + "break;}")
    # Only the two bounce-level continues, not nested LUT-neighbour continues.
    body = body.replace("            continue;", "            " + finish("bounce+1>=uMaxBounces?6:0") + "continue;", 1)
    body = _once(body, "                        continue;", "                        " + finish("bounce+1>=uMaxBounces?6:0") + "continue;")
    tail = "\n    }\n}"
    if not body.endswith(tail):
        raise ValueError("forward loop closing seam changed")
    body = body[:-len(tail)] + "\n        " + finish("bounce+1>=uMaxBounces?6:0") + tail
    source = source[:start] + body + source[end:]
    return source


def camera_shader(source, *, roi_capacity=131072, other_capacity=131072):
    """Observe actual camera main-loop paths, separately labelled adjoint."""
    if "sal_begin(" in source:
        raise ValueError("ledger already installed")
    start, end = _block(source, "void main() {")
    body = source[start:end]
    body = _once(body, "int currentMedium=studio_origin_medium(ro,rd);",
                 "int currentMedium=studio_origin_medium(ro,rd);\n"
                 "        sal_ray=uint(sensor_px.x+sensor_px.y*full_size.x);sal_root=0xffffffffu;sal_sample=uint(s);sal_count(16);")
    body = _once(body, "int   hit_id = nearest_hit(ro, rd, surf_t);",
                 "int   hit_id = nearest_hit(ro, rd, surf_t);\n"
                 "            sal_bounce=bounce;sal_begin(ro,rd,hit_id>=0?surf_t:0.0,hit_id,currentMedium,vec3(throughput[0],throughput[1],throughput[2]),0.0,0.0,1u);")
    finish = lambda reason: f"sal_finish({reason},vec3(throughput[0],throughput[1],throughput[2]),rd,currentMedium);"
    substitutions = [
        ("if (hit_id < 0 || tp_max < 0.003) break;", "if (hit_id < 0 || tp_max < 0.003) {" + finish("hit_id<0?1:7") + "break;}"),
        ("if(tri.mat_out.w>=0.999)break;", "if(tri.mat_out.w>=0.999){" + finish(8) + "break;}"),
        ("if(!studio_boundary_step(tri,rd,frequency,rand01(rng),currentMedium,factor)) break;", "if(!studio_boundary_step(tri,rd,frequency,rand01(rng),currentMedium,factor)) {" + finish(2) + "break;}"),
        ("if (opacity >= 0.999) break;", "if (opacity >= 0.999) {" + finish(8) + "break;}"),
        ("if (rand01(rng) > survival) break;", "if (rand01(rng) > survival) {" + finish(9) + "break;}"),
        ("if(conductor_weight<=0.0)break;", "if(conductor_weight<=0.0){" + finish(5) + "break;}"),
        ("if (tp_after < 0.005) break;", "if (tp_after < 0.005) {" + finish(10) + "break;}"),
        ("                continue;", "                " + finish("bounce>=uMaxBounces?6:0") + "continue;"),
    ]
    for old, new in substitutions:
        body = _once(body, old, new)
    body = _once(body, "        }\n        for (int i = 0; i < nl; i++) {\n            float measured",
                 "            " + finish("bounce>=uMaxBounces?6:0") + "\n        }\n        for (int i = 0; i < nl; i++) {\n            float measured")
    return source[:start] + _header(roi_capacity, other_capacity) + body + source[end:]


class ArrivalLedger:
    """Explicit lifetime and bounded readback for one labelled diagnostic run."""
    def __init__(self, bounds, *, roi_capacity=131072, other_capacity=131072):
        from OpenGL import GL
        _header(roi_capacity, other_capacity)
        if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)) <= BINDING:
            raise RuntimeError("ray ledger requires SSBO binding21")
        self.roi_capacity, self.other_capacity = roi_capacity, other_capacity
        self.bounds = np.asarray(bounds, np.float32)
        if self.bounds.shape != (2, 3) or not np.isfinite(self.bounds).all() or not np.all(self.bounds[1] > self.bounds[0]):
            raise ValueError("invalid ray ledger bounds")
        self.nbytes = (HEADER_WORDS + (roi_capacity + other_capacity) * EVENT_WORDS) * 4
        # The bounded maximum here is below2GiB; the GLint query is sufficient
        # and avoids PyOpenGL's platform-dependent int64 scalar conversion.
        if self.nbytes > int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE)):
            raise RuntimeError("ray ledger exceeds GPU storage block size")
        self.buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, self.nbytes, None, GL.GL_DYNAMIC_READ)
        self.clear()

    def clear(self):
        from OpenGL import GL
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer)
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, HEADER_WORDS * 4, np.zeros(HEADER_WORDS, '<u4'))

    def bind(self, program, population):
        from OpenGL import GL
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, BINDING, self.buffer)
        old = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        GL.glUseProgram(program)
        GL.glUniform1ui(GL.glGetUniformLocation(program, 'uArrivalPopulation'), int(population))
        GL.glUniform3fv(GL.glGetUniformLocation(program, 'uArrivalMin'), 1, self.bounds[0])
        GL.glUniform3fv(GL.glGetUniformLocation(program, 'uArrivalMax'), 1, self.bounds[1])
        GL.glUseProgram(old)

    def read(self):
        from OpenGL import GL
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glFinish()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer)
        head = np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, HEADER_WORDS * 4), '<u4').copy()
        def pool(offset, count):
            if not count:
                return np.zeros((0, EVENT_WORDS), '<u4')
            return np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, (HEADER_WORDS + offset * EVENT_WORDS) * 4, count * EVENT_WORDS * 4), '<u4').reshape(-1, EVENT_WORDS).copy()
        roi = pool(0, min(int(head[1]), self.roi_capacity))
        other = pool(self.roi_capacity, min(int(head[2]), self.other_capacity))
        return head, roi, other

    def close(self):
        from OpenGL import GL
        if self.buffer:
            GL.glDeleteBuffers(1, [self.buffer]);self.buffer = 0


def event_dict(row):
    u = np.asarray(row, '<u4');f = u.view('<f4');i = u.view('<i4')
    camera = int(u[0]) == 1
    hit = int(i[7])
    return dict(kind='camera-adjoint' if camera else 'forward-energy', ray_id=int(u[1]),
        root_id=None if camera else int(u[2]), rng_seed=int(u[3]), source_population=int(u[4]),
        sample=int(u[59]), bounce=int(u[5]), medium=int(i[6]), triangle=hit,
        material=int(u[8]) if hit >= 0 else None, flags=int(u[9]),
        reason=REASONS.get(int(u[10]), 'unknown'), roi_intersects=bool(u[11]),
        a_m=f[12:15].tolist(), b_m=f[16:19].tolist() if hit >= 0 else None,
        direction=f[20:23].tolist(), segment_length_m=float(f[15]) if hit >= 0 else None,
        frequency_hz=None if camera else float(f[19]), roi_interval_m=f[23:25].tolist() if u[11] else None,
        segment_entry_throughput=f[44:47].tolist() if camera else float(f[25]),
        segment_post_beer_throughput=f[47:50].tolist() if camera else float(f[26]),
        roi_entry_throughput=(f[50:53].tolist() if camera else float(f[27])) if u[11] else None,
        roi_exit_throughput=(f[53:56].tolist() if camera else float(f[28])) if u[11] else None,
        post_interaction_throughput=f[56:59].tolist() if camera else float(f[31]),
        outgoing_direction=f[32:35].tolist(), next_medium=int(i[35]), root_replay_weight=float(f[29]),
        exclusion_mask=int(u[30]), deposit_attempts=int(u[36]), nonzero_quantized_deposits=int(u[37]),
        quantized_sum=int(u[38]), quantized_sum_overflow=bool(u[39]), attempted_scaled_sum=float(f[40]),
        pre_surface_throughput=None if camera or u[63]!=0x53414c32 else float(f[41]),
        surface_reflectivity=None if camera or u[63]!=0x53414c32 else float(f[42]),surface_absorption=None if camera or u[63]!=0x53414c32 else float(f[43]),
        surface_opacity=None if camera or u[63]!=0x53414c32 else float(f[62]),
        surface_branch=None if camera or u[63]!=0x53414c32 else dict(diffuse=bool(u[61]&1),spectral_reflector=bool(u[61]&2),rough_conductor=bool(u[61]&4)),
        starts_inside_roi=bool(u[60] & 1), ends_inside_roi=bool(u[60] & 2))


def save_capture(destination, capture, *, metadata):
    path = Path(destination);path.mkdir(parents=True, exist_ok=True)
    head, roi, other = capture
    np.savez_compressed(path/'events.npz', header=head, roi=roi, non_roi=other)
    with (path/'roi-arrivals.jsonl').open('w', encoding='utf-8') as stream:
        for row in roi:
            stream.write(json.dumps(event_dict(row), allow_nan=False) + '\n')
    receipt = dict(schema='studio.ray-arrival-ledger.v2', **metadata,
        counts={name:int(head[index]) for index,name in COUNTERS.items()},
        termination_counts={name:int(head[32+index]) for index,name in REASONS.items() if index},
        bounce_segments=head[64:128].tolist(), bounce_roi_intersections=head[128:192].tolist(),
        roi_records=len(roi), non_roi_records=len(other),
        roi_records_complete=not bool(head[3] or head[18]),
        all_event_records_complete=not bool(head[3] or head[4] or head[18]),
        events_sha256=hashlib.sha256((path/'events.npz').read_bytes()).hexdigest(),
        exclusions={1:'invalid medium',2:'fixed3 disabled',4:'no complete root-sweep weight',8:'nonpositive throughput'},
        limits='Actual hit segments only. Misses have no endpoint. Camera adjoint throughput is not forward energy. Primary forward pass only; secondary pending queue is not dispatched by this pump. No NEE shadow connection ledger. Record-pool truncation and counter overflow are explicit.')
    (path/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (path/'README.txt').write_text(FORMAT_README,encoding='utf-8')
    return receipt


FORMAT_README = '''Studio actual-ray arrival ledger

events.npz is a NumPy compressed archive, not a rendered image. header is256
little-endian uint32 words. roi and non_roi are Nx64 little-endian uint32 rows.
Floating fields below are exact IEEE754 float32 bit patterns: view their words
as <f4, do not numerically cast the uint32 values. Signed IDs use a <i4 view.
The schema and provenance are in receipt.json; roi-arrivals.jsonl decodes all
retained ROI events. It is intentionally empty when there are no ROI crossings.
Non-ROI records retain real full-scene paths and why they ended. No endpoint is
claimed for a miss. The first pools are GPU scheduling order, not path order;
sort by population, ray_id, sample, bounce to follow individual trajectories.

Event words (u=uint32,i=int32,f=float32):
0 u channel:0 forward-energy,1 camera-adjoint
1 u ray_id (forward dispatch ID; camera full-frame pixel ID)
2 u root_id (forward source row);0xffffffff for camera
3 u bounce RNG seed;4 u source-population seed;5 u bounce (zero-based)
6 i medium on segment;7 i hit triangle;8 u material;9 u material flags
10 u end reason;11 u ROI intersection flag
12:15 f actual origin;15 f actual hit distance;16:19 f actual endpoint
19 f forward frequency;20:23 f actual normalized direction
23 f ROI first distance;24 f ROI last distance
25 f entry throughput;26 f post-canonical-Beer throughput
27 f ROI entry throughput;28 f ROI exit throughput
29 f forward inverse replay multiplicity;30 u recording exclusion mask
31 f post-interaction throughput;32:35 f outgoing direction
35 i medium after interaction
36 u exact directional DDA deposit attempts;37 u positive quantized deposits
38 u sum of positive quantized weights;39 u sum overflow flag
40 f sum of attempted pre-quantized scaled weights
41 f pre-surface throughput;42 f surface reflectivity;43 f surface absorption
44:47 f segment entry throughput vector;47:50 f post-Beer vector
50:53 f ROI entry vector;53:56 f ROI exit vector;56:59 f post-interaction vector
59 u camera within-dispatch sample ID
60 u clipped interval flags:bit0 starts inside ROI;bit1 ends inside ROI
61 u selected material branches:bit0 fully diffuse,bit1 spectral,bit2 rough conductor
62 f opacity
63 u layout marker:0x53414c31=v1,0x53414c32=v2.
Words41:44,61:63 are unavailable (zero padding) in v1, NOT observed zero values.
Camera vectors are the existing650/550/450nm adjoint throughput. Camera data are
not forward energy and do not update the forward atlas. Forward vector columns
repeat that path's single spectral packet; use its frequency, not an RGB label.
ROI throughput/interval is meaningful only when11!=0. Hit endpoint/distance,
material and flags are meaningful only when signed hit7>=0. A miss stores origin
in endpoint storage as padding, never a physical segment to an invented endpoint.

Header counters are decoded in receipt.json. Raw indices:
0 actual hit segments,1 ROI intersections,2 non-ROI events,3 ROI rows dropped,
4 non-ROI rows dropped,5 misses,6 ROI valid medium,7 ROI positive entry,
8 deposit attempts,9 positive quantized deposits,10 ROI invalid medium,
11 ROI fixed3 disabled,12 ROI zero root weight,13 ROI nonpositive entry,
14 recorder calls,15 terminated paths,16 started paths,18 counter overflow,
19 quantization/sum overflow.32+reason are termination counts.
64:128 hit-segment bounce histogram;128:192 ROI bounce histogram. Last bin
collects bounce63 and higher. Other header words are reserved.

End reasons:0 continued,1 no hit,2 boundary rejection,3 packet below0.002,
4 absorber,5 conductor rejected,6 bounce budget,7 camera below0.003,
8 opaque emitter,9 roulette,10 camera below0.005.
Exclusion bits:1 invalid medium,2 fixed3 disabled,4 zero root weight,
8 nonpositive throughput. ROI intersections are counted BEFORE these checks.

The primary prebuilt pump does not dispatch queued reactive/edge secondaries.
No pending-secondary parent root IDs are invented. No direct NEE shadow
connections are in this ledger. Camera paths are independently labelled.
Pool capacities and every truncation/overflow are explicit in the receipt.
'''
