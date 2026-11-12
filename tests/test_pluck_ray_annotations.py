"""Original Python signature contracts; never import the renderer or invoke GL."""
import ast
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "demo_pluck_gl.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def function(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def annotations(tree, name):
    args = function(tree, name).args
    return {arg.arg: ast.unparse(arg.annotation) for arg in
            [*args.posonlyargs, *args.args, *args.kwonlyargs]}


@pytest.mark.parametrize("name", ["_prog", "_gpu_ray_field", "_gpu_ray_field_prebuilt"])
def test_original_ray_entry_signatures_are_fully_annotated(tree, name):
    fn = function(tree, name)
    args = fn.args
    parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    parameters.extend(arg for arg in (args.vararg, args.kwarg) if arg is not None)
    assert all(arg.annotation is not None for arg in parameters)
    assert fn.returns is not None
    # These are the Python API's actual inputs, not replacement tensor inputs.
    assert not any(isinstance(node, ast.Name) and node.id in {"Any", "AbstractTensor"}
                   for annotation in [*(arg.annotation for arg in parameters), fn.returns]
                   for node in ast.walk(annotation))


def test_program_takes_variadic_source_kind_pairs(tree):
    fn = function(tree, "_prog")
    assert fn.args.vararg.arg == "shader_pairs"
    assert ast.unparse(fn.args.vararg.annotation) == "tuple[str, int]"
    assert ast.unparse(fn.returns) == "int"


def test_legacy_ray_source_and_return_contracts(tree):
    types = annotations(tree, "_gpu_ray_field")
    assert types["scene"] == "Optional[CavityScene]"
    assert types["outline"] == "np.ndarray"
    assert types["body_h"] == "float"
    assert types["sources"] == "Optional[Sequence[_RaySource]]"
    assert types["dims"] == "tuple[int, int, int]"
    assert {types[name] for name in ("max_bounces", "segment_cap", "dispatch_batch")} == {"int"}
    assert types["include_stage"] == "bool"
    result = function(tree, "_gpu_ray_field").returns
    assert isinstance(result, ast.Subscript) and isinstance(result.slice, ast.Tuple)
    assert len(result.slice.elts) == 8
    assert ast.unparse(result.slice.elts[-1]) == "SensorAccumulator | Literal[0]"
    # Existing unavailable/empty/link-failure paths return an integer sentinel,
    # not a SensorAccumulator and not None in the last tuple position.
    failure_returns = [node for node in ast.walk(function(tree, "_gpu_ray_field"))
                       if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
                       and isinstance(node.value.elts[0], ast.Constant)
                       and node.value.elts[0].value is None]
    assert len(failure_returns) == 3
    assert all(ast.literal_eval(node.value) == (None, None, None, None, 0, None, 0, 0)
               for node in failure_returns)


def test_legacy_sources_allow_optional_trailing_fields(tree):
    alias = next(node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "_RaySource"
                         for target in node.targets))
    tuples = [node for node in ast.walk(alias) if isinstance(node, ast.Subscript)
              and isinstance(node.value, ast.Name) and node.value.id == "tuple"]
    assert sorted(len(node.slice.elts) for node in tuples) == [4, 5, 6, 7, 8]
    assert all([ast.unparse(item) for item in node.slice.elts[:4]] ==
               ["np.ndarray", "np.ndarray", "int", "np.ndarray"] for node in tuples)
    assert any(isinstance(node, ast.Name) and node.id == "LightSpec" for node in ast.walk(alias))


def test_prebuilt_buffers_match_their_original_producers(tree):
    types = annotations(tree, "_gpu_ray_field_prebuilt")
    for name in ("packed_geom", "packed_shade", "mat_buf", "bvh_nodes", "context_buf", "source_buf"):
        assert types[name] == "NDArray[np.float32]"
    assert types["bvh_ids"] == "NDArray[np.int32]"
    producer_return = function(tree, "_build_gpu_bvh").body[-1]
    ids_call = producer_return.value.elts[1]
    assert next(ast.unparse(item.value) for item in ids_call.keywords if item.arg == "dtype") == "np.int32"
    for name in ("sphere_regions", "region_records", "native_pbr"):
        assert types[name] == "Optional[NDArray[np.float32]]"
    # Existing test and production callers supply both arrays and Python lists.
    assert types["bounds"] == "tuple[ArrayLike, ArrayLike]"


def test_prebuilt_object_and_optional_scalar_contracts(tree):
    types = annotations(tree, "_gpu_ray_field_prebuilt")
    assert types["film_stack"] == "Optional[FilmStack]"
    assert types["volume_extinction"] == "Optional[ExtinctionGrid]"
    assert types["forward_response_consumer"] == "Optional[SceneResponseConsumer]"
    for name in ("forward_response_identity", "light_field_region"):
        assert types[name] == "Optional[dict[str, object]]"
    for name in ("forward_response_reserved_bytes", "light_field_grid_size"):
        assert types[name] == "Optional[int]"
    for name in ("record_directional", "light_field_internal_media"):
        assert types[name] == "bool"
    assert types["speed_of_medium"] == "float"
    assert ast.unparse(function(tree, "_gpu_ray_field_prebuilt").returns) == "Optional[dict[str, object]]"


def test_annotation_dependencies_are_type_only(tree):
    assert any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
               and any(alias.name == "annotations" for alias in node.names) for node in tree.body)
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
    imports = {alias.name for node in guard.body if isinstance(node, ast.ImportFrom)
               for alias in node.names}
    assert {"ArrayLike", "NDArray", "CavityScene", "ExtinctionGrid", "SceneResponseConsumer"} <= imports
    assert not any(isinstance(node, ast.ImportFrom) and
                   any(alias.name in {"CavityScene", "ExtinctionGrid", "SceneResponseConsumer"}
                       for alias in node.names) for node in tree.body)
