"""Replay a verified canonical source history, with exact per-population seeds."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.job/'config.json').read_text())
    expected=json.loads((args.job.parent.parent.parent/'live-receipt.json').read_text())['field']['directional']['source_populations']
    from camera_designer.studio_native_emitters import native_emissive_sources
    scene=np.load(args.cache/'glass_rod.npz')
    source_dir=args.output/'sources';source_dir.mkdir(exist_ok=True)
    h='0'*64;populations=[]
    for index in range(expected['first_index'],expected['last_index']+1):
        seed=config['emitter_sampling']['seed']+index
        roots=native_emissive_sources((scene['geom'],scene['shade'],scene['mat'],scene['triangles']),[.45,.55,.65],
            ray_budget=config['emitter_sampling']['ray_budget'],seed=seed)
        digest=hashlib.sha256(roots.tobytes()).hexdigest()
        h=hashlib.sha256(bytes.fromhex(h)+index.to_bytes(8,'little')+bytes.fromhex(digest)).hexdigest()
        np.save(source_dir/f'sources-{seed}.npy',roots)
        populations.append(dict(index=index,seed=seed,bounce_rng_seed=index*7,source_sha256=digest,root_count=len(roots)))
    if h!=expected['rolling_sha256']:raise ValueError('regenerated source history does not match actual published rolling hash')
    source_proof=dict(source_history_matches_published=True,rolling_sha256=h,expected=expected,populations=populations)
    (args.output/'source-history.json').write_text(json.dumps(source_proof,indent=2))
    print(json.dumps(dict(source_history_matches_published=True,populations=len(populations),rolling_sha256=h)),flush=True)
    import studio_ray_arrival_ledger_gpu_check as proof
    original_argv=sys.argv
    receipts=[];started=time.monotonic()
    try:
        for pop in populations:
            folder=args.output/f"population-{pop['seed']}"
            sys.argv=['arrival-ledger',str(args.job),str(args.cache),str(folder),'--population',str(pop['seed']),
                      '--rng-seed',str(pop['bounce_rng_seed']),'--source-records',str(source_dir/f"sources-{pop['seed']}.npy")]
            proof.main()
            receipts.append(json.loads((folder/'receipt.json').read_text()))
    finally:sys.argv=original_argv
    summary=dict(schema='studio.ray-arrival-history.v1',source_history_matches_published=True,rolling_sha256=h,
        population_count=len(receipts),source_seeds=[p['seed'] for p in populations],bounce_rng_seeds=[p['bounce_rng_seed'] for p in populations],
        counts={k:sum(r['counts'][k] for r in receipts) for k in receipts[0]['counts']},
        termination_counts={k:sum(r['termination_counts'][k] for r in receipts) for k in receipts[0]['termination_counts']},
        bounce_segments=np.sum([r['bounce_segments'] for r in receipts],axis=0).tolist(),
        bounce_roi_intersections=np.sum([r['bounce_roi_intersections'] for r in receipts],axis=0).tolist(),
        roi_records_complete=all(r['roi_records_complete'] for r in receipts),all_event_records_complete=all(r['all_event_records_complete'] for r in receipts),
        pending_queue_counts=[r['pending_queue']['newly_queued'] for r in receipts],atlas_nonzero_values=[r['atlas_nonzero_values'] for r in receipts],
        total_event_bytes=sum((args.output/f"population-{p['seed']}"/'events.npz').stat().st_size for p in populations),
        wall_seconds=time.monotonic()-started,
        limits='Re-execution of actual saved full geometry and exact regenerated native source bytes; source history matches published rolling digest. Deterministic bounce RNG sequence from original live pump. No material/transport changes. Every finite actual primary-forward segment is recorded; all queue, camera and NEE exclusions remain explicit. Not a field convergence claim.')
    (args.output/'receipt.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
