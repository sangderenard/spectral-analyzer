"""Control source generation only: never launches GL or a shader compiler."""
from pathlib import Path
import runpy

def test_control_uses_native_composers_without_evaluated_response_hooks():
    root=Path(__file__).resolve().parents[1]
    source=runpy.run_path(str(Path(__file__).with_name('studio_forward_source_control.py')))['native_control_source'](root)
    for absent in ('StudioScalarVisits','studio_scalar_visit(','studio_response_copy(','studio_terminal_latch(','uStudioResponseCapacity','studio_bounce_exit'):
        assert absent not in source
    for present in ('studio_record_segment(','studio_pbr_weight(','studio_medium_transmittance(','if (uMode == PASS_SENSOR)','if (uMode == PASS_REACTIVE)','studio_active[32]'):
        assert present in source
    assert 'if (uMode != PASS_FORWARD) return;' not in source
