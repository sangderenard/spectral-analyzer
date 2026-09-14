"""Compile Pluck's desktop base-material fragment shader to WebGL 2.

Raw GLSL is first ingested into Turing's project shader IR.  Backend selection
and WebGL storage specialization happen only after that project representation
exists; this script is the repository-local bridge, not a second translator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
TURING_ROOT = ROOT.parent / "turing"
FRAGMENT_SOURCE = ROOT / "csrc" / "shaders" / "base_material.frag.glsl"
VERTEX_SOURCE = ROOT / "csrc" / "shaders" / "base_material.vert.glsl"
DEFAULT_OUTPUT = (
    ROOT / "csrc" / "shaders" / "generated" / "base_material.webgl.frag.glsl"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(DEFAULT_OUTPUT.suffix + ".json")
DEFAULT_VERTEX_OUTPUT = (
    ROOT / "csrc" / "shaders" / "generated" / "base_material.webgl.vert.glsl"
)
DEFAULT_VERTEX_MANIFEST = DEFAULT_VERTEX_OUTPUT.with_suffix(
    DEFAULT_VERTEX_OUTPUT.suffix + ".json"
)


def _compiler():
    if not TURING_ROOT.is_dir():
        raise RuntimeError(
            f"the sibling Turing compiler repository is required at {TURING_ROOT}"
        )
    if str(TURING_ROOT) not in sys.path:
        sys.path.insert(0, str(TURING_ROOT))
    from src.compiler.glsl_project_ir import compile_glsl_project_to_webgl

    return compile_glsl_project_to_webgl


def compile_base_material_webgl():
    return _compiler()(
        FRAGMENT_SOURCE.read_text(encoding="utf-8"),
        source_name="base_material",
        stage_hint="fragment",
    )


def compile_base_material_webgl_chain():
    compile_project = _compiler()
    vertex = compile_project(
        VERTEX_SOURCE.read_text(encoding="utf-8"),
        source_name="base_material_vertex",
        stage_hint="vertex",
    )
    fragment = compile_project(
        FRAGMENT_SOURCE.read_text(encoding="utf-8"),
        source_name="base_material_fragment",
        stage_hint="fragment",
    )
    return vertex, fragment


def write_base_material_webgl(
    output: Path = DEFAULT_OUTPUT,
    manifest: Path = DEFAULT_MANIFEST,
):
    translated = compile_base_material_webgl()
    if not translated.complete:
        report = "\n".join(item.format() for item in translated.diagnostics)
        raise RuntimeError(f"base material WebGL translation is incomplete:\n{report}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(translated.source, encoding="utf-8", newline="\n")
    manifest.write_text(
        json.dumps(translated.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return translated, output, manifest


def write_base_material_webgl_chain(
    vertex_output: Path = DEFAULT_VERTEX_OUTPUT,
    fragment_output: Path = DEFAULT_OUTPUT,
):
    vertex, fragment = compile_base_material_webgl_chain()
    incomplete = [shader for shader in (vertex, fragment) if not shader.complete]
    if incomplete:
        report = "\n".join(
            item.format() for shader in incomplete for item in shader.diagnostics
        )
        raise RuntimeError(f"base material WebGL chain is incomplete:\n{report}")
    for shader, output in ((vertex, vertex_output), (fragment, fragment_output)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(shader.source, encoding="utf-8", newline="\n")
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(shader.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return vertex, fragment, vertex_output, fragment_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--vertex-output", type=Path, default=DEFAULT_VERTEX_OUTPUT)
    args = parser.parse_args(argv)
    vertex, translated, vertex_output, output = write_base_material_webgl_chain(
        args.vertex_output, args.output
    )
    # Preserve the explicit fragment manifest override offered by the original
    # single-stage command surface.
    generated_manifest = output.with_suffix(output.suffix + ".json")
    manifest = args.manifest or generated_manifest
    if manifest != generated_manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(translated.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(vertex_output)
    print(output)
    print(manifest)
    print(
        f"{len(vertex.project_ir.instructions) + len(translated.project_ir.instructions)} "
        "source lines -> "
        f"{len(vertex.execution_plan.sections) + len(translated.execution_plan.sections)} "
        "planned function sections"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
