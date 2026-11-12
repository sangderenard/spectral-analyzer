"""Original native film columns reduced by the existing sensor evidence kernel.

No response reconstruction, optical replay, spectral filter evaluation or pixel
image. Runtime invocation remains explicitly opted in; CPU tests inject a sum
oracle without constructing a GL context.
"""
import hashlib
import json
import numpy as np
from .studio_copied_components import native_response_sums
from .marble_color_profile import sensor_to_linear_srgb
from .studio_response_lineage import recipient_id

VERSION='native-evaluated-rgb-v1'

def reactive_queue_complete(queue):
    """Validate observed native work accounting, never infer from response RGB."""
    if not isinstance(queue,dict) or queue.get('observed') is not True or queue.get('complete') is not True:
        return False
    if queue.get('incomplete_observed') is True or queue.get('counter_regression') is True:return False
    names=('attempted','capacity','stored','dispatched','dropped_enqueue','pending_unprocessed')
    if any(type(queue.get(k)) is not int or queue[k]<0 for k in names):return False
    return (queue['stored']==min(queue['attempted'],queue['capacity']) and
            queue['attempted']==queue['stored']==queue['dispatched'] and
            queue['dropped_enqueue']==0 and queue['pending_unprocessed']==0)


def authored_volume_disabled(dispatch):
    """Only actual disabled native controls establish this independence."""
    controls=dispatch.get('volume_controls')
    if not isinstance(controls,dict):return False
    values=[controls.get(k) for k in ('master_alpha','diffuse_scatter','specular_scatter')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in values):return False
    return values[0]==0.0 or (values[1]==0.0 and values[2]==0.0)


def build_scene_application(store, *, film_response, compile_program, memory_budget_bytes,
                            accumulator=native_response_sums, current_metadata=None):
    slots=np.flatnonzero(store.used & (store.index[:,0]==4))
    # Conservative bounded staging+native accumulator+readback reservation,
    # checked before per-sample contexts/keys/arrays are allocated.
    if type(memory_budget_bytes) is not int or len(slots)*4096+65536>memory_budget_bytes:
        raise MemoryError('scene application staging exceeds explicitly reserved budget')
    queries=[];query_ids={};keys={};samples=[];suppressed=set();current_query_ids=set();queue_unavailable_samples=0
    for slot in slots:
        idx=store.index[slot];row=store.native[slot]
        # Native-only/virtual/unlocated rows remain raw evidence. They have
        # no raster recipient and must never become a zero-ID application.
        if idx[4]==0:continue
        if np.any(idx[16:20]==0xffffffff) or not row[11]:continue
        context=store._contexts[int(idx[2])][0]
        dispatch=context['dispatch'];camera=dispatch.get('camera_uniforms')
        if not isinstance(camera,dict) or not isinstance(camera.get('eye'),list) or len(camera['eye'])!=3:continue
        layer_specs=dispatch.get('layer_specs')
        if not isinstance(layer_specs,list) or len(layer_specs)!=3:continue
        if context.get('film_response')!=film_response:continue
        if current_metadata is None:continue
        if (current_metadata.get('film_response')!=film_response or
            current_metadata.get('dispatch',{}).get('layer_specs')!=layer_specs):continue
        # Lens and physical origin are compatibility, not pixel identity.
        # Orientation is represented by the existing declared angular bucket.
        lens={k:v for k,v in camera.items() if k!='eye'}
        query=dict(origin_m=camera['eye'],lens=lens,
            film=film_response,layer_specs=layer_specs,transport=dict(scene_revision=store.scene_revision,
            source_sha256=context['source_sha256'],source_kind=context.get('source_kind'),
            transport_program_sha256=context.get('transport_program_sha256'),transport_scene_sha256=context.get('transport_scene_sha256'),shared_band0_path=True,reactive_queue=context.get('reactive_queue')),
            volume_controls=dispatch.get('volume_controls'))
        encoded=json.dumps(query,sort_keys=True,separators=(',',':'),allow_nan=False)
        if encoded not in query_ids:
            query_ids[encoded]=len(queries);queries.append(dict(id=len(queries),**query))
        qid=query_ids[encoded]
        if current_metadata is not None:
            current_dispatch=current_metadata.get('dispatch',{})
            current_camera=current_dispatch.get('camera_uniforms')
            # Only producer-supplied exact current identity can select contexts.
            # Historical cache order/sequence is explicitly not a selector.
            if (current_metadata.get('camera_revision') and
                current_metadata.get('transport_scene_sha256') and
                current_metadata['transport_scene_sha256']==context.get('transport_scene_sha256') and
                current_metadata.get('source_sha256',current_dispatch.get('source_sha256'))==context['source_sha256'] and
                current_metadata.get('source_kind')==context.get('source_kind') and
                current_metadata.get('transport_program_sha256')==context.get('transport_program_sha256') and
                current_metadata.get('reactive_queue')==context.get('reactive_queue') and
                current_dispatch.get('volume_controls')==dispatch.get('volume_controls') and
                current_camera==camera and current_dispatch.get('layer_specs')==layer_specs):
                current_query_ids.add(qid)
        # Exact native first-recipient stage controls ownership. Unknown stage
        # is retained raw but cannot be relabeled as a renderable material.
        stage=int(row[5]);component=1 if stage==1 else 2 if stage==2 else None
        if component is None:continue
        queue_known=reactive_queue_complete(context.get('reactive_queue'))
        disabled_volume=authored_volume_disabled(dispatch)
        straight_prefix=bool(row[115] and np.array_equal(row[12:15],row[116:119]))
        base=(recipient_id(int(idx[4]),int(idx[5])),*map(int,idx[16:19]),int(idx[19]),qid)
        for comp,value_at,known_at in ((0,72,88),(component,80,96)):
            key=base+(comp,)
            if key not in keys:keys[key]=[len(keys),0,0]
            item=keys[key];item[1]+=1
            if store.support_has_eviction(slot):suppressed.add(item[0])
            value=row[value_at:value_at+3].view(np.float32)
            if np.any(value<0):raise ValueError('negative native film component is unsupported by existing evidence kernel')
            dependency_known=queue_known or (comp==0 and disabled_volume)
            if not dependency_known:queue_unavailable_samples+=1
            usable=bool(dependency_known and straight_prefix and item[0] not in suppressed and row[6]==1 and row[7]==3 and np.all(row[known_at:known_at+3]==3) and np.isfinite(value).all())
            if usable:item[2]+=1;samples.append((item[0],value.copy()))
    count=len(keys)
    rows=np.zeros((count,12),dtype='<u4');profile=None
    if samples:
        spectra=np.asarray([v for _,v in samples],np.float32)
        lineage=np.zeros((len(samples),8),np.uint32)
        lineage[:,0]=[node for node,_ in samples];lineage[:,6]=np.float32(1).view(np.uint32)
        sums=np.asarray(accumulator(count,lineage,spectra,compile_program),np.float32)
        if sums.shape!=(count,3) or not np.isfinite(sums).all():raise ValueError('invalid native scene sums')
    else:sums=np.zeros((count,3),np.float32)
    for key,(node,denominator,available) in keys.items():
        if node in suppressed:available=0
        conditional=sums[node]/available if available else np.zeros(3,np.float32)
        rgb,profile=sensor_to_linear_srgb(conditional[None,:],film_response)
        rows[node,:7]=key;rows[node,7:9]=[available,denominator]
        rows[node,9:12]=np.asarray(rgb[0],np.float32).view(np.uint32)
    metadata=dict(version=VERSION,ready=False,section='application_rows',query_contexts=queries,
        current_query_context_ids=sorted(current_query_ids),
        current_camera_revision=None if current_metadata is None else current_metadata.get('camera_revision'),
        current_live_revision=None if current_metadata is None else current_metadata.get('live_revision'),
        current_transport_scene_sha256=None if current_metadata is None else current_metadata.get('transport_scene_sha256'),
        eviction_suppressed_support_components=len(suppressed),
        eviction_policy='discarded denominator makes matching support unavailable; bounded conservative false positives allowed',
        denominator_scope='retained original samples; any possible support eviction forces availability zero',
        queue_unavailable_component_samples=queue_unavailable_samples,
        queue_support='observed complete native queue required; authored-disabled volume independent; missing/pending/dropped/sticky incomplete unknown',
        prefix_support='exact copied original-camera and first-recipient direction bits must match; otherwise unavailable',
        support='recipient-cell-stencil-v1',rgb='linear-srgb',alpha='available-original-camera-samples',
        component_stage='native-film-filtered-before-linear-profile',rgb_premultiplied=False,
        profile=profile,emission_ownership='included once in first-recipient downstream component',
        components={'0':'pre-first-recipient native volume','1':'first-opaque downstream native response',
                    '2':'first-glass complete downstream native path'},
        specular_override_supported=False,physical_completeness_claim=False)
    return metadata,rows
