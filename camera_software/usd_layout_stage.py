"""OpenUSD scene layers for layout design objects and composed interfaces."""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .layout_object_work import (
    LayoutDesignObjectWork,
    LayoutDesignSubtypeWork,
    LayoutObjectWorkManifest,
    LayoutWorkProgressCache,
)


def _identifier(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(value))
    result = "_".join(words) or "Object"
    if result[0].isdigit():
        result = "_" + result
    return result


def _relative_asset(source_file: str, target_file: str) -> str:
    return os.path.relpath(
        os.path.abspath(target_file), os.path.dirname(os.path.abspath(source_file))
    ).replace("\\", "/")


def _quoted(value: str) -> str:
    return json.dumps(str(value))


def _custom_attribute(name: str, value: Any) -> str:
    field = _identifier(name)
    if isinstance(value, bool):
        return f"        custom bool {field} = {str(value).lower()}"
    if isinstance(value, int):
        return f"        custom int {field} = {value}"
    if isinstance(value, float):
        return f"        custom double {field} = {value:.12g}"
    if (
        isinstance(value, (tuple, list))
        and len(value) in {2, 3, 4}
        and all(isinstance(item, (int, float)) for item in value)
    ):
        value_type = (
            f"int{len(value)}"
            if all(isinstance(item, int) for item in value)
            else f"float{len(value)}"
        )
        rendered = ", ".join(
            str(item) if isinstance(item, int) else f"{float(item):.9g}"
            for item in value
        )
        return (
            f"        custom {value_type} {field} = ({rendered})"
        )
    return f"        custom string {field} = {_quoted(value)}"


def _write(path: str, content: str) -> str:
    final_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    temporary = final_path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content.rstrip() + "\n")
    os.replace(temporary, final_path)
    return final_path


@dataclass(frozen=True)
class UsdLayoutPackage:
    package_root: str
    stage_path: str
    object_layers: Mapping[str, str]
    subtype_layers: Mapping[str, str]
    animation_layers: Mapping[str, str]
    text_image_layers: Mapping[str, str]
    assembly_manifest_path: str
    package_manifest_path: str
    window_element_layer_path: str = ""
    window_element_manifest_path: str = ""

    def mapping(self) -> dict[str, Any]:
        return asdict(self)


def _animation_layer(
    path: str,
    animation: Any,
    *,
    near_focus_cm: float,
    far_focus_cm: float,
) -> str:
    samples = []
    percent_samples = []
    phase_samples = []
    for frame in animation.frames:
        percent = float(frame.focus_location_percent)
        distance = near_focus_cm + (
            far_focus_cm - near_focus_cm
        ) * percent / 100.0
        samples.append(f"            {frame.frame_index}: {distance:.9g},")
        percent_samples.append(
            f"        {frame.frame_index}: {percent:.9g},"
        )
        phase_samples.append(
            f"        {frame.frame_index}: {_quoted(frame.phase)},"
        )
    end = len(animation.frames) - 1
    content = f"""#usda 1.0
(
    defaultPrim = "RackFocus"
    metersPerUnit = 0.01
    startTimeCode = 0
    endTimeCode = {end}
    timeCodesPerSecond = 24
    framesPerSecond = 24
)

def Xform "RackFocus" (
    kind = "subcomponent"
)
{{
    custom string animationKey = {_quoted(animation.animation_key)}
    custom string animationName = {_quoted(animation.name)}
    custom int rackFrames = {animation.rack_frames}
    custom int holdFrames = {animation.hold_frames}
    custom int focusLocationCount = {animation.focus_location_count}
    custom int replacementFrame = {animation.replacement_frame_index}
    custom float focusLocationPercent.timeSamples = {{
{chr(10).join(percent_samples)}
    }}
    custom string phase.timeSamples = {{
{chr(10).join(phase_samples)}
    }}

    def Camera "DemonstrationCamera"
    {{
        float focalLength = 50
        float fStop = 2.8
        float focusDistance.timeSamples = {{
{chr(10).join(samples)}
        }}
        float horizontalAperture = 36
        float verticalAperture = 24
    }}
}}
"""
    return _write(path, content)


def _subtype_layer(
    path: str,
    design_object: LayoutDesignObjectWork,
    subtype: LayoutDesignSubtypeWork,
    animation_paths: Iterable[str],
    replacement_path: str,
    source_scene_path: str = "",
) -> str:
    root_name = _identifier(
        f"{design_object.display_name}_{subtype.display_name}"
    )
    animation_refs = []
    for index, animation_path in enumerate(animation_paths):
        relative = _relative_asset(path, animation_path)
        animation_refs.append(
            f'''        def Xform "Animation_{index:02d}" (
            references = @{relative}@</RackFocus>
        )
        {{
        }}'''
        )
    texture_block = (
        f"            asset inputs:file = @{_relative_asset(path, replacement_path)}@"
        if replacement_path else
        '            asset inputs:file'
    )
    diffuse_input = (
        "                color3f inputs:diffuseColor.connect = "
        "<../RenderedTexture.outputs:rgb>"
        if replacement_path else
        "                color3f inputs:diffuseColor = "
        "(0.1098, 0.1216, 0.1490)"
    )
    render_source = "rendered" if replacement_path else "fallback"
    source_scene = (
        f"    custom asset sourceScene = "
        f"@{_relative_asset(path, source_scene_path)}@\n"
        if source_scene_path and os.path.isfile(source_scene_path) else ""
    )
    content = f"""#usda 1.0
(
    defaultPrim = "{root_name}"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "{root_name}" (
    kind = "subcomponent"
)
{{
    custom string objectKey = {_quoted(design_object.object_key)}
    custom string subtypeKey = {_quoted(subtype.subtype_key)}
    custom token sourceCapture = "{subtype.source_capture}"
    custom token renderSource = "{render_source}"
    custom string progressCacheKey = {_quoted(subtype.still_cache_key)}
{source_scene}

    def Mesh "RepresentativeSquare"
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = <../Materials/PanelMaterial>
    }}

    def Scope "Materials"
    {{
        def Material "PanelMaterial"
        {{
            token outputs:surface.connect = <PanelPreview.outputs:surface>
            def Shader "PanelPreview"
            {{
                uniform token info:id = "UsdPreviewSurface"
{diffuse_input}
                token outputs:surface
            }}
            def Shader "RenderedTexture"
            {{
                uniform token info:id = "UsdUVTexture"
{texture_block}
                float2 inputs:st.connect = <PrimvarReader.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "PrimvarReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}

    def Scope "PoseAnimations"
    {{
{chr(10).join(animation_refs)}
    }}
}}
"""
    return _write(path, content)


def _object_layer(
    path: str,
    design_object: LayoutDesignObjectWork,
    subtype_paths: Iterable[tuple[LayoutDesignSubtypeWork, str]],
) -> str:
    root_name = _identifier(design_object.display_name)
    references = []
    for subtype, subtype_path in subtype_paths:
        relative = _relative_asset(path, subtype_path)
        subtype_root = _identifier(
            f"{design_object.display_name}_{subtype.display_name}"
        )
        name = _identifier(subtype.subtype_key)
        references.append(
            f'''        def Xform "{name}" (
            references = @{relative}@</{subtype_root}>
        )
        {{
        }}'''
        )
    content = f"""#usda 1.0
(
    defaultPrim = "{root_name}"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "{root_name}" (
    kind = "component"
)
{{
    custom string objectKey = {_quoted(design_object.object_key)}
    custom token objectKind = "{design_object.object_kind}"
    def Scope "Subtypes"
    {{
{chr(10).join(references)}
    }}
}}
"""
    return _write(path, content)


def _text_image_layer(path: str, image: Any) -> str:
    root_name = _identifier(f"TextImage_{image.owner_id}")
    length = max(1, len(image.text))
    parts = []
    for index, part in enumerate(image.parts):
        if image.resolution == "exact_token":
            center_x, scale_x = 0.0, 1.0
        else:
            center_x = -0.5 + (part.text_offset + 0.5) / length
            scale_x = 1.0 / length
        artifact = str(part.artifact_path)
        texture_file = (
            artifact
            if artifact.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")
            ) and os.path.isfile(artifact)
            else ""
        )
        scene_asset = (
            f"        custom asset sourceScene = "
            f"@{_relative_asset(path, part.scene_path)}@\n"
            if part.scene_path and os.path.isfile(part.scene_path) else ""
        )
        texture_asset = (
            f"                asset inputs:file = "
            f"@{_relative_asset(path, texture_file)}@\n"
            if texture_file else ""
        )
        diffuse = (
            "                color3f inputs:diffuseColor.connect = "
            "<Texture.outputs:rgb>"
            if texture_file else
            "                color3f inputs:diffuseColor = (0.82, 0.76, 0.62)"
        )
        parts.append(f'''    def Xform "Part_{index:04d}"
    {{
        double3 xformOp:translate = ({center_x:.12g}, 0, 0)
        double3 xformOp:scale = ({scale_x:.12g}, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        custom string objectKey = {_quoted(part.object_key)}
        custom string subtypeKey = {_quoted(part.subtype_key)}
        custom token subtypeKind = "{part.subtype_kind}"
        custom int textOffset = {part.text_offset}
        custom float convergence = {part.convergence:.9g}
{scene_asset}        def Mesh "Card"
        {{
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
                interpolation = "vertex"
            )
            uniform token subdivisionScheme = "none"
            rel material:binding = <../Material>
        }}
        def Material "Material"
        {{
            token outputs:surface.connect = <Preview.outputs:surface>
            def Shader "Preview"
            {{
                uniform token info:id = "UsdPreviewSurface"
{diffuse}
                token outputs:surface
            }}
            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
{texture_asset}                float2 inputs:st.connect = <PrimvarReader.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "PrimvarReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}''')
    content = f"""#usda 1.0
(
    defaultPrim = "{root_name}"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "{root_name}" (
    kind = "component"
)
{{
    custom string layoutOwner = {_quoted(image.owner_id)}
    custom string authoredText = {_quoted(image.text)}
    custom token resolution = "{image.resolution}"
{chr(10).join(parts)}
}}
"""
    return _write(path, content)


def compose_usd_stage(
    stage_path: str,
    references: Iterable[Mapping[str, Any]],
    *,
    stage_name: str = "World",
    sublayers: Iterable[str] = (),
) -> str:
    """Compose retained scene/object layers into one additive OpenUSD stage."""

    sublayer_paths = [
        f"        @{_relative_asset(stage_path, layer)}@,"
        for layer in sublayers
    ]
    sublayer_block = (
        "    subLayers = [\n"
        + "\n".join(sublayer_paths)
        + "\n    ]\n"
        if sublayer_paths else ""
    )
    prims = []
    for item in references:
        source = str(item["asset_path"])
        prim_path = str(item["prim_path"])
        name = _identifier(str(item["name"]))
        metadata = dict(item.get("metadata", {}))
        translate = item.get("translate")
        scale = item.get("scale")
        custom = "\n".join(
            _custom_attribute(key, value)
            for key, value in sorted(metadata.items())
        )
        prims.append(
            f'''    def Xform "{name}" (
        instanceable = true
        references = @{_relative_asset(stage_path, source)}@<{prim_path}>
    )
    {{
{f'        double3 xformOp:translate = ({translate[0]:.12g}, {translate[1]:.12g}, {translate[2]:.12g})' if translate is not None else ''}
{f'        double3 xformOp:scale = ({scale[0]:.12g}, {scale[1]:.12g}, {scale[2]:.12g})' if scale is not None else ''}
{('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]' if translate is not None and scale is not None else '')}
{custom}
    }}'''
        )
    content = f"""#usda 1.0
(
    defaultPrim = "{_identifier(stage_name)}"
    metersPerUnit = 0.01
    upAxis = "Y"
{sublayer_block})

def Xform "{_identifier(stage_name)}" (
    kind = "assembly"
)
{{
{chr(10).join(prims)}
}}
"""
    return _write(stage_path, content)


def _window_element_layer(path: str, manifest: Any) -> str:
    children = []
    for index, element in enumerate(manifest.elements):
        rect = tuple(element.layout_rect_px)
        children.append(f'''    def Xform "Element_{index:04d}"
    {{
        custom string elementKey = {_quoted(element.element_key)}
        custom string parentKey = {_quoted(element.parent_key)}
        custom token elementKind = {_quoted(element.kind)}
        custom token stateSubtype = {_quoted(element.state_subtype)}
        custom int4 layoutRectPx = ({rect[0]}, {rect[1]}, {rect[2]}, {rect[3]})
        custom int siblingOrder = {element.sibling_order}
        custom int zIndex = {element.z_index}
        custom string contentSignature = {_quoted(element.content_signature)}
        custom string authoredText = {_quoted(element.authored_text)}
        custom string styleObjectKey = {_quoted(element.style_object_key)}
        custom string styleSubtypeKey = {_quoted(element.style_subtype_key)}
    }}''')
    return _write(path, f'''#usda 1.0
(
    defaultPrim = "WindowElements"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "WindowElements" (
    kind = "group"
)
{{
    custom string sourceLayout = {_quoted(manifest.source_layout)}
    custom string contentSignature = {_quoted(manifest.content_signature)}
{chr(10).join(children)}
}}
''')


def write_layout_usd_package(
    manifest: LayoutObjectWorkManifest,
    cache: LayoutWorkProgressCache,
    output_directory: str,
    *,
    near_focus_cm: float = 35.0,
    far_focus_cm: float = 200.0,
    resolved_text_images: Iterable[Any] = (),
    render_library: Any = None,
    assembly_readiness: Any = None,
    window_element_manifest: Any = None,
) -> UsdLayoutPackage:
    """Publish object, subtype, animation, and final composition layers."""

    root = os.path.abspath(output_directory)
    object_layers: dict[str, str] = {}
    subtype_layers: dict[str, str] = {}
    animation_layers: dict[str, str] = {}
    text_image_layers: dict[str, str] = {}
    stage_references = []
    for object_index, design_object in enumerate(manifest.objects):
        subtype_path_pairs = []
        object_consumers = []
        library_bundle = (
            render_library.find(design_object.object_key)
            if render_library is not None else None
        )
        for subtype_index, subtype in enumerate(design_object.subtypes):
            animation_paths = []
            for animation_index, animation in enumerate(subtype.pose_animations):
                animation_path = os.path.join(
                    root,
                    "animations",
                    f"{object_index:02d}_{subtype_index:02d}_{animation_index:02d}_rack_focus.usda",
                )
                _animation_layer(
                    animation_path,
                    animation,
                    near_focus_cm=near_focus_cm,
                    far_focus_cm=far_focus_cm,
                )
                animation_paths.append(animation_path)
                animation_layers[animation.animation_key] = animation_path
            library_subtype = next(
                (
                    item for item in (
                        () if library_bundle is None
                        else library_bundle.subtypes
                    )
                    if item.source_asset_key == subtype.subtype_key
                ),
                None,
            )
            source_scene_path = (
                "" if library_subtype is None or not library_subtype.scenes
                else library_subtype.scenes[0].scene_path
            )
            subtype_path = os.path.join(
                root,
                "subtypes",
                f"{object_index:02d}_{subtype_index:02d}_{_identifier(subtype.subtype_key)}.usda",
            )
            _subtype_layer(
                subtype_path,
                design_object,
                subtype,
                animation_paths,
                cache.replacement_path(subtype),
                source_scene_path,
            )
            subtype_layers[
                f"{design_object.object_key}/{subtype.subtype_key}"
            ] = subtype_path
            subtype_path_pairs.append((subtype, subtype_path))
            for consumer in subtype.consumers:
                object_consumers.append((subtype_index, subtype, consumer))
        object_path = os.path.join(
            root, "objects", f"{object_index:02d}_{_identifier(design_object.object_key)}.usda"
        )
        _object_layer(object_path, design_object, subtype_path_pairs)
        object_layers[design_object.object_key] = object_path
        object_root = _identifier(design_object.display_name)
        for subtype_index, subtype, consumer in object_consumers:
            target_x, target_y, target_w, target_h = consumer.target_rect_px
            layout_x, layout_y, layout_w, layout_h = consumer.layout_rect_px
            translate = (
                (target_x - layout_x + 0.5 * target_w) / layout_w - 0.5,
                0.5 - (target_y - layout_y + 0.5 * target_h) / layout_h,
                0.0,
            )
            scale = (
                target_w / layout_w,
                target_h / layout_h,
                1.0,
            )
            stage_references.append({
                "name": (
                    f"{consumer.owner_id}_{consumer.role}_"
                    f"{object_index}_{subtype_index}"
                ),
                "asset_path": object_path,
                "prim_path": f"/{object_root}",
                "translate": translate,
                "scale": scale,
                "metadata": {
                    "layoutOwner": consumer.owner_id,
                    "objectKey": design_object.object_key,
                    "subtypeKey": subtype.subtype_key,
                    "patchRole": consumer.role,
                    "fill": consumer.fill,
                    "cropMinimum": consumer.crop_vectors[0],
                    "cropMaximum": consumer.crop_vectors[1],
                    "targetRectPx": consumer.target_rect_px,
                    "samplingPolicy": consumer.sampling_spec.get(
                        "uv_transform", ""
                    ),
                    "terminalTilePolicy": consumer.sampling_spec.get(
                        "terminal_tile_policy", ""
                    ),
                },
            })
    for text_index, image in enumerate(resolved_text_images):
        text_path = os.path.join(
            root,
            "text_images",
            f"{text_index:02d}_{_identifier(image.owner_id)}.usda",
        )
        _text_image_layer(text_path, image)
        text_image_layers[image.owner_id] = text_path
        target_x, target_y, target_w, target_h = image.target_rect_px
        layout_x, layout_y, layout_w, layout_h = image.layout_rect_px
        stage_references.append({
            "name": f"text_image_{image.owner_id}",
            "asset_path": text_path,
            "prim_path": f"/{_identifier(f'TextImage_{image.owner_id}')}",
            "translate": (
                (target_x - layout_x + 0.5 * target_w) / layout_w - 0.5,
                0.5 - (target_y - layout_y + 0.5 * target_h) / layout_h,
                0.001,
            ),
            "scale": (
                target_w / layout_w,
                target_h / layout_h,
                1.0,
            ),
            "metadata": {
                "layoutOwner": image.owner_id,
                "resolution": image.resolution,
                "convergence": image.convergence,
            },
        })
    window_element_layer_path = ""
    window_element_manifest_path = ""
    if window_element_manifest is not None:
        window_element_manifest_path = window_element_manifest.save(
            os.path.join(root, "window_elements", "manifest.json")
        )
        window_element_layer_path = _window_element_layer(
            os.path.join(root, "window_elements", "elements.usda"),
            window_element_manifest,
        )
        stage_references.append({
            "name": "window_elements",
            "asset_path": window_element_layer_path,
            "prim_path": "/WindowElements",
            "metadata": {
                "contentSignature": window_element_manifest.content_signature,
                "semanticSource": "retained_2d_layout_before_rasterization",
            },
        })
    stage_path = compose_usd_stage(
        os.path.join(root, "stages", "program_backdrop.usda"),
        stage_references,
        stage_name=manifest.source_layout,
    )
    package_manifest_path = os.path.join(root, "package_manifest.json")
    assembly_manifest_path = os.path.join(root, "assembly_manifest.json")
    readiness_mapping = (
        assembly_readiness.mapping()
        if assembly_readiness is not None
        and callable(getattr(assembly_readiness, "mapping", None))
        else dict(assembly_readiness or {})
    )
    aspect_library = [
        {
            "object_key": design_object.object_key,
            "subtype_key": subtype.subtype_key,
            **asdict(variant),
        }
        for design_object in manifest.objects
        for subtype in design_object.subtypes
        for variant in subtype.aspect_variants
    ]
    _write(assembly_manifest_path, json.dumps({
        "schema_version": 1,
        "manifest_kind": "holistic_static_texture_assembly",
        "source_layout": manifest.source_layout,
        "stage_path": os.path.abspath(stage_path),
        "tier_order": [
            "alphabet", "token", "panel", "scene",
            "holistic_static_texture",
        ],
        "provider_signoffs": readiness_mapping,
        "aspect_library": aspect_library,
        "dynamic_fallback": {
            "active_while_constituents_change": True,
            "panel_source": "parametric_empty_panel",
            "text_source": "alphabet_then_token",
        },
        "holistic_exposure": {
            "start_policy": "content_static_and_render_work_idle",
            "output_policy": "one_configuration_whole_static_texture",
        },
        "recrop_reuse": {
            "allowed": True,
            "enabled_when": "window_element_manifest_complete",
            "window_element_manifest_path": window_element_manifest_path,
            "crop_stage": "after_holistic_exposure",
            "requires_same_camera_lighting_material_signature": True,
            "requires_same_reflection_boundary_signature": True,
            "reject_if_reflection_crosses_crop_boundary": True,
        },
    }, indent=2, sort_keys=True))
    package = UsdLayoutPackage(
        package_root=root,
        stage_path=stage_path,
        object_layers=object_layers,
        subtype_layers=subtype_layers,
        animation_layers=animation_layers,
        text_image_layers=text_image_layers,
        assembly_manifest_path=assembly_manifest_path,
        package_manifest_path=package_manifest_path,
        window_element_layer_path=window_element_layer_path,
        window_element_manifest_path=window_element_manifest_path,
    )
    _write(
        package_manifest_path,
        json.dumps(package.mapping(), indent=2, sort_keys=True),
    )
    return package


__all__ = [
    "UsdLayoutPackage",
    "compose_usd_stage",
    "write_layout_usd_package",
]
