"""Positive ROI, Beer/Snell and scalar parity using the established slab test."""
from pathlib import Path
import json
import runpy
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import demo_pluck_gl as r
    from camera_designer.studio_ray_arrival_ledger import ArrivalLedger,forward_shader,save_capture
    r._ensure_worker_gl_context()
    ledger=ArrivalLedger(([-.008]*3,[.008]*3),roi_capacity=4096,other_capacity=128)
    original=r._prog
    def observed(*pairs):
        rewritten=[]
        for source,kind in pairs:
            if 'studio_record_segment(ro,hp,packet,currentMedium);' in source:
                source=forward_shader(source,roi_capacity=4096,other_capacity=128)
            rewritten.append((source,kind))
        program=original(*rewritten);ledger.bind(program,0);return program
    r._prog=observed
    try:
        slab=runpy.run_path(str(Path(__file__).with_name('studio_internal_field_integrated_gpu_check.py')))
        slab['main']()
        head,roi,other=ledger.read()
        assert head[1]>0 and head[8]>0 and head[9]>0
        assert not head[3] and not head[4] and not head[18]
        assert np.any(roi[:,6].view('<i4')>=0)
        # Existing slab script above checked exact scalar parity, Beer integral
        # and Snell against the same transport with all observers absent.
        result=save_capture(Path(sys.argv[1]),(head,roi,other),metadata=dict(kind='controlled-positive-validation',result='PASS',
            checks=['existing scalar bit-identical parity','positive ROI intersections','positive exact quantized deposits','internal glass segments','canonical Beer integral','canonical Snell direction'],
            warning='Controlled slab validation only; not the actual studied scene or source history.'))
        print(json.dumps(result))
    finally:r._prog=original;ledger.close()


if __name__=='__main__':main()
