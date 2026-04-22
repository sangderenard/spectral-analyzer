import json
import subprocess
import sys
from pathlib import Path


def test_extended_diagnostic_script_runs() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "diagnostic_analytic_driver_solver.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(proc.stdout)

    assert payload["solver_contract"]["scalar_node_assumption"] is False
    assert payload["solver_contract"]["broadcast_edge_weights"] is True
    assert payload["metadata_check"]["meta_edge_count"] == 1
    assert payload["metadata_check"]["edge_group_count"] == 3
    assert payload["metadata_check"]["edge_group_keys"] == ["voice_bus", "print_bus", "feedback_bus"]
    assert payload["metadata_check"]["layer_plan"]["signal_layers"] == ["voice", "master"]
    assert payload["metadata_check"]["layer_plan"]["parameter_layer"] == "param"
    assert payload["metadata_check"]["layer_plan"]["parameter_any_in"] is True
    assert payload["metadata_check"]["layer_plan"]["parameter_any_out"] is True
    assert payload["metadata_check"]["layer_plan"]["one_network_per_sample"] is True
    assert payload["metadata_check"]["transport_sides"]["network_link_count"] == (
        payload["metadata_check"]["transport_sides"]["grid_edge_count"]
        + payload["metadata_check"]["transport_sides"]["patch_link_count"]
    )
    assert payload["metadata_check"]["node_layer_presence"]["lin_a"]["inputs"] == ["voice"]
    assert payload["metadata_check"]["node_layer_presence"]["lin_c"]["outputs"] == ["master"]
    assert payload["metadata_check"]["node_layer_presence"]["lin_a"]["parameter_inputs"] is True
    assert payload["concurrent_known_answer_check"]["max_abs_error_a"] <= 1e-14
    assert payload["concurrent_known_answer_check"]["max_abs_error_b"] <= 1e-14
    assert payload["linear_exact_check"]["max_abs_error"] == 0.0
    assert payload["nonlinear_cycle_check"]["tier"] == 2
    assert payload["nonlinear_cycle_check"]["nonlinear_sccs"] >= 1
