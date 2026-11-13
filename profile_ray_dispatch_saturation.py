"""Bounded original Spectral T1→T2→T3 CPU/GPU ray-queue throughput probe.

Each submission is a ray job against the same native triangle/material scene.
This measures the existing Spectral pipeline, not Nodus's separate market or
complete BDPT exposure. A finite in-flight window prevents unbounded queues.
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _spectral_kernels as kernels
from ray_tracer_bridge import per_tri_spectral_to_mat_buf

parser = argparse.ArgumentParser()
parser.add_argument('--rays', type=int, default=10000)
parser.add_argument('--batches', type=int, default=4)
parser.add_argument('--mode', choices=('cpu', 'gpu', 'mixed'), default='mixed')
parser.add_argument('--window', type=int, default=30000)
parser.add_argument('--record-budget', type=int, default=100000000)
parser.add_argument('--job', default='one')
parser.add_argument('--grid', type=int, default=1)
parser.add_argument('--bands', type=int, default=4)
args = parser.parse_args()
if not (1 <= args.rays <= 100000 and 1 <= args.batches <= 10000 and
        1 <= args.grid <= 128 and 1 <= args.bands <= 32 and
        args.window >= args.rays and args.record_budget >= 1000000):
    parser.error('rays/batches/grid/bands/window/record budget exceed bounded probe limits')

freq = 299792458.0 / np.linspace(420e-9,700e-9,args.bands)
reflectance = np.zeros((1, args.bands), np.float64)
mat_idx, mat_buf, mat_count = per_tri_spectral_to_mat_buf(
    reflectance, reflectance, reflectance, freq)
if args.grid == 1:
    triangle = np.array([[0,0,0, 1,0,0, 1,1,0]], np.float64)
else:
    faces=[]
    for iy in range(args.grid):
        for ix in range(args.grid):
            x0,x1=ix/args.grid,(ix+1)/args.grid
            y0,y1=iy/args.grid,(iy+1)/args.grid
            faces.extend(((x0,y0,0,x1,y0,0,x1,y1,0),
                          (x0,y0,0,x1,y1,0,x0,y1,0)))
    triangle=np.asarray(faces,np.float64)
    mat_idx=np.zeros(len(triangle),np.int32)
normal = np.repeat(np.array([[0,0,1]], np.float64),len(triangle),axis=0)
tracer = kernels.RayTracer(len(triangle), triangle, normal, mat_idx, mat_buf,
                           int(mat_count), freq, 299792458.0,
                           np.zeros(args.bands, np.float64))
tracer.set_bdpt_record_float_budget(args.record_budget)
tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12,
    use_gpu_compute=args.mode!='cpu', gpu_all_stages=args.mode=='gpu',
    shader_dir='csrc/shaders')
xy = np.linspace(0.10, 0.60, args.rays, dtype=np.float64)
origins = np.stack((xy, 0.05+0.4*xy, np.ones(args.rays)), axis=1)
directions = np.zeros((args.rays,3), np.float64)
directions[:,2] = -1
amplitudes = np.ones((args.rays,args.bands), np.complex128)
started=time.perf_counter()
cpu_started=time.process_time()
drained=0
record_kinds=[0,0,0,0]
strike_hits=0
def drain():
    global drained, strike_hits
    while tracer.pipeline_stats()['output_queue_depth']:
        records=tracer.drain_records(100000)
        count=len(records['kind'])
        if not count:break
        drained+=count
        kinds=np.bincount(records['kind'], minlength=4)
        for i in range(4): record_kinds[i]+=int(kinds[i])
        strike_hits+=int(np.count_nonzero(records['hit_tri'][records['kind']==0]>=0))
for batch in range(args.batches):
    if args.window:
        while tracer.in_flight_count()>max(0,args.window-args.rays):
            drain()
            time.sleep(.001)
    tracer.submit_rays(origins, directions, amplitudes,
        max_bounces=1, min_amplitude=1e-12)
deadline=time.perf_counter()+90
while tracer.in_flight_count() and time.perf_counter()<deadline:
    drain()
    time.sleep(.005)
drain()
elapsed=time.perf_counter()-started
stats=dict(tracer.pipeline_stats())
print(json.dumps({'job':args.job,'mode':args.mode,'grid':args.grid,'bands':args.bands,
    'rays':args.rays,'batches':args.batches,
    'elapsed_s':elapsed,'process_cpu_s':time.process_time()-cpu_started,
    'drained':drained,'record_kinds':dict(zip(('strike','terminal','miss','field'),record_kinds)),
    'strike_hits':strike_hits,'in_flight':tracer.in_flight_count(),
    'stats':{k:dict(v) if k.startswith('t') and len(k)==2 else v
             for k,v in stats.items() if k in ('t1','t2','t3','t4','t5','in_flight')}},
    default=float), flush=True)
if tracer.in_flight_count():raise SystemExit(2)
