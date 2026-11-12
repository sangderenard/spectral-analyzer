"""CPU budget oracles and source/recorder contracts; no GPU execution."""
import itertools
from pathlib import Path
import runpy
import numpy as np
from camera_designer.studio_forward_responses import copy_forward_responses

ROOT=Path(__file__).resolve().parents[1]

def test_same_parent_budget_is_conserved_and_permutation_invariant():
    rng=np.random.default_rng(20261112)
    for parent in (1e-5,.01,1.,100.):
        for fractions in [(0.,0.,0.),(.5,.5,.5),(.1,.2,.3),*rng.uniform(0,.5,(100,3))]:
            offers=np.array(fractions)*parent
            shares=offers*min(1.,parent/offers.sum()) if offers.sum()>0 else offers
            for order in itertools.permutations(range(3)):
                reordered=offers[list(order)]
                result=reordered*min(1.,parent/reordered.sum()) if reordered.sum()>0 else reordered
                assert np.allclose(result,shares[list(order)],rtol=1e-14,atol=1e-20)
            for admitted in itertools.product((0,1),repeat=3):
                sent=np.dot(shares,admitted)
                remaining=max(0.,parent-sent)
                assert np.isclose(remaining+sent,parent,rtol=1e-14,atol=1e-20)
                assert remaining>=0

def test_actual_composed_budget_and_explicit_debit_copy():
    source=runpy.run_path(str(ROOT/'tests/studio_forward_source_control.py'))['native_control_source'](ROOT)
    forward=source.split('void main() {',1)[1].split('void reactive_main(',1)[0]
    assert forward.index('vec3 offers=')<forward.index('if(studio_pbr_opaque(tri))')
    assert 'float scale=total>packet.z?packet.z/total:1.0;' in forward
    assert forward.count('if(uPendingCap>0)offers[')==3
    assert forward.count('if(studio_edge_enqueued)sent+=studio_edge_share;')==3
    assert 'packet.z*=studio_edge_remainder;' in forward
    # Disabled queue/no accepted children debit zero; exact original enqueue
    # overflow early returns cannot reach its successful-write marker.
    append=source.split('void append_pending_ray(vec3 origin',2)[2].split('void sensor_main',1)[0]
    assert append.index('if (idx >= uint(uPendingCap)) return;')<append.index('studio_edge_enqueued=true;')
    observed,seams=copy_forward_responses(source,terminal_only=True)
    assert any(row['expression']=='studio_edge_remainder' for row in seams)
    assert 'packet.z*=studio_copied_factor; studio_response_stage_presence|=2u;' in observed
    assert 'studio_response_copy(2u,1u,' in observed
    # Fluorescence is unchanged and remains separately additive.
    assert 'packet.z * clamp(reflectivity, 0.0, 1.0)' in source


def test_float32_budget_rounding_is_bounded_not_exact():
    rng=np.random.default_rng(83)
    for parent in np.float32([1e-5,.01,1.,100.,1e12]):
        offers=(rng.uniform(0,.5,(1000,3))*parent).astype(np.float32)
        total=(offers[:,0]+offers[:,1]+offers[:,2]).astype(np.float32)
        scale=np.minimum(np.float32(1),parent/total)
        shares=offers*scale[:,None]
        sent=(shares[:,0]+shares[:,1]+shares[:,2]).astype(np.float32)
        remainder=np.maximum(np.float32(0),np.float32(1)-sent/parent)
        final=parent*remainder+sent
        assert np.all(np.isfinite(final))
        assert np.all(np.abs(final-parent)<=4*np.finfo(np.float32).eps*parent)
