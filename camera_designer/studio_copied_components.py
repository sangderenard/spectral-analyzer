"""Producer-only reduction of actual response copies through native evidence.

No optical response is evaluated here. The only numerical integrator is the
existing sensor_mip_accumulate kernel; CPU code builds identity/availability
columns and preserves original camera-sample denominators.
"""
from collections import defaultdict
import hashlib
import json
import numpy as np
from .studio_response_lineage import copied_recipient
from .spectral_sensor_basis import fixed3_response_matrix
from .marble_color_profile import sensor_to_linear_srgb

VERSION = 'studio-copied-components-v1'


def native_response_sums(node_count, lineage, spectra, compile_program):
    """Use the existing native evidence ABI, not a second estimator."""
    from .studio_sensor_evidence import StudioSensorEvidence
    evidence = StudioSensorEvidence(1,node_count,3,16,compile_program,
        response_node_count=node_count,sample_capacity=len(spectra))
    try:
        evidence.accumulate_response_rows(lineage,spectra)
        return evidence.read_moments()['sum']
    finally:
        evidence.destroy()


class CopiedComponentConsumer:
    """Synchronous bounded normal-product consumer; no raw-history spool.

    state_provider returns current scene_revision, camera_revision and copied
    recipient lineage. The caller updates these only from actual scene/camera
    changes. Publish receives a complete current component descriptor. Native
    sums are reseeded through the same accumulator between batches; event
    direct_weight/sum_sq are deliberately not camera weights or variance.
    """
    def __init__(self, *, state_provider, film_response, compile_program, publish,
                 memory_budget_bytes, reserved_bytes, accumulator=None):
        if not callable(state_provider) or not callable(publish):
            raise TypeError('explicit state and publication callbacks required')
        self.state_provider=state_provider
        self.film_response=film_response
        self.compile_program=compile_program
        self.publish=publish
        self.memory_budget_bytes=int(memory_budget_bytes)
        self.reserved_bytes=int(reserved_bytes)
        self.accumulator=accumulator or native_response_sums
        self.identity=None
        self.sums={}
        self.camera_weights=defaultdict(int)
        self.available_weights=defaultdict(int)
        self.last_sequence=-1
        self.latest=None

    def preflight_capacity(self, capacity, *, live_copy_bytes):
        # Conservative bound includes Python dict/list/tuple/NumPy staging, not
        # merely packed arrays. Fail BEFORE constructing those objects or GL
        # dispatch. Existing accumulated state is included every time.
        needed=(int(live_copy_bytes)+int(capacity)*8192+
                len(self.sums)*2048+len(self.camera_weights)*512)
        if self.reserved_bytes+needed>self.memory_budget_bytes:
            raise MemoryError('copied component staging cannot fit total working budget')
        return needed

    def __call__(self, metadata, interval_bytes, response_bytes):
        state=self.state_provider()
        for name in ('scene_revision','camera_revision'):
            if metadata.get(name)!=state.get(name):
                raise ValueError('stale copied component '+name)
        if metadata.get('response_weighting')!='original_path_throughput_not_availability_premultiplied':
            raise ValueError('unsupported response weighting')
        if metadata.get('response_schema')!='native-camera-response-copies-v1':
            raise ValueError('unsupported response copy schema')
        dispatch=metadata['dispatch']
        if int(dispatch.get('history_samples',0))!=0:
            raise ValueError('copied component accumulation does not yet support the selected rolling-history window')
        camera_json=json.dumps(dispatch.get('camera_uniforms',{}),sort_keys=True,separators=(',',':'))
        camera_hash=hashlib.sha256(camera_json.encode()).hexdigest()
        identity=(metadata['scene_revision'],metadata['camera_revision'],dispatch['source_sha256'],camera_hash,int(metadata.get('field_generation',0)))
        if self.identity!=identity:
            self.identity=identity;self.sums={};self.camera_weights.clear();self.available_weights.clear();self.last_sequence=-1
        sequence=int(metadata['sequence'])
        if sequence <= self.last_sequence:
            raise ValueError('duplicate response dispatch would replay contributions')
        self.preflight_capacity(int(metadata['row_count']),live_copy_bytes=len(interval_bytes)+len(response_bytes)+2*int(metadata.get('required_bytes',0)))
        rows=np.frombuffer(interval_bytes,dtype='<u4')
        responses=np.frombuffer(response_bytes,dtype='<u4')
        n=int(metadata['row_count'])
        if rows.size!=n*128 or responses.size!=n*128 or not metadata.get('complete_dispatch_observation'):
            raise ValueError('incomplete response-copy dispatch')
        rows=rows.reshape(n,128);responses=responses.reshape(n,128)
        if n and (np.any(rows[:,7]!=1) or np.any(rows[:,6]!=3)):
            raise ValueError('normal component product requires the actual fixed-three-line sensor')
        specs=dispatch.get('layer_specs')
        if not isinstance(specs,list) or len(specs)!=3:
            raise ValueError('original three sensor filter specifications required')
        projection=fixed3_response_matrix(specs)
        grouped=defaultdict(list)
        pixel_samples=defaultdict(set)
        for i,row in enumerate(rows):
            sample=(int(row[15]),int(row[2]))
            pixel=(int(row[0]),int(row[1]))
            grouped[(pixel,sample)].append(i)
            pixel_samples[pixel].add(sample)
        # Counters are camera invocations/samples, never bounce or response rows.
        next_camera=dict(self.camera_weights)
        for pixel,samples in pixel_samples.items():
            next_camera[pixel]=next_camera.get(pixel,0)+len(samples)
        next_available=dict(self.available_weights)
        events=[]
        new_keys=set(self.sums)
        for (pixel,sample),indices in grouped.items():
            block=rows[indices]
            first=block[0]
            first_ids={(int(r[115]),int(r[124]),int(r[125])) for r in block if int(r[115])!=0xffffffff and int(r[124])!=2}
            if len(first_ids)!=1:
                # Separate spectral paths with different first receivers cannot
                # manufacture a complete RGB recipient from partial colors.
                continue
            native_hit,primitive,owner=next(iter(first_ids))
            recipient=copied_recipient(state['lineage'],native_hit,primitive,owner)
            if recipient[0]==0:continue
            stages={int(r[41]) for r in block if int(r[41]) in (1,2)}
            if len(stages)!=1:continue
            stage=next(iter(stages))
            root=(pixel[0],pixel[1],int(recipient[0]),int(recipient[1]))
            stage_keys=[root+(2,)] if stage==2 else [root+(0,),root+(1,)]
            new_keys.update(stage_keys)
            validity={key:np.ones(3,bool) for key in stage_keys}
            seen={key:0 for key in stage_keys}
            throughput_known=np.ones(3,bool)
            copied=[]
            ordered=sorted(indices,key=lambda index:(int(rows[index,3]),int(rows[index,40])))
            for index in ordered:
                row=rows[index];response=responses[index]
                if int(row[32]) in (2,4,5,6,7):
                    for key in stage_keys:validity[key][:]=False
                def source_known(kind,band):
                    state=(int(response[108+kind])>>(band*2))&3
                    value=response[kind*8+band].view(np.float32)
                    return state==3 and np.isfinite(value) and value>=0
                def apply_dependency(kind):
                    presence=int(response[96+kind]) & int(row[40]) & 7
                    for band in range(3):
                        if presence&(1<<band):throughput_known[band]&=source_known(kind,band)
                def copy_additive(kind):
                    presence=int(response[96+kind]) & int(row[40]) & 7
                    if not presence:return
                    key=root+((2,) if stage==2 else ((0,) if kind==0 else (1,)))
                    seen[key]|=presence
                    spectral=np.zeros(3,np.float32)
                    for band in range(3):
                        if presence&(1<<band):
                            validity[key][band]&=source_known(kind,band) and throughput_known[band]
                            if kind==3 and int(response[120+band])&30:validity[key][band]=False
                            spectral[band]=response[kind*8+band].view(np.float32)
                    copied.append((key,spectral))
                # Dependency order is the EXISTING camera loop order. Factors
                # are never multiplied again: only their source-knownness can
                # invalidate a later copied value that already contains them.
                apply_dependency(6)  # Native Beer occurs before the air term.
                copy_additive(0)
                apply_dependency(11) # Existing air alpha affects later terms.
                for kind in range(1,6):copy_additive(kind)
                for kind in (7,8,9,10):apply_dependency(kind)
            usable={key for key in stage_keys if seen[key]==7 and bool(np.all(validity[key]))}
            for key in usable:next_available[key]=next_available.get(key,0)+1
            for key,spectral in copied:
                if key in usable:
                    # Existing sensor response projection; no new color model.
                    events.append((key,np.asarray(projection@spectral,np.float32)))
        keys=sorted(new_keys)
        if not keys:
            self.camera_weights=defaultdict(int,next_camera);self.last_sequence=sequence
            self.latest=self._descriptor([],metadata,camera_hash,None)
            self.publish(self.latest);return
        key_indices={key:i for i,key in enumerate(keys)}
        # Prior native sums re-enter the SAME native accumulator as unit-weight
        # columns. No CPU summation or response-row averaging is introduced.
        input_rows=[(key,value) for key,value in self.sums.items()]+events
        if not input_rows:input_rows=[(keys[0],np.zeros(3,np.float32))]
        spectrum=np.asarray([value for _,value in input_rows],np.float32)
        lineage=np.zeros((len(input_rows),8),np.uint32)
        lineage[:,0]=[key_indices[key] for key,_ in input_rows]
        lineage[:,6]=np.float32(1).view(np.uint32)
        # Native node64B, sum/sumsq24B, weight4B; input lineage32B+spectrum12B.
        # Include CPU staging, native readback, and paired copied rows still live.
        needed=(len(keys)*104+len(input_rows)*44)*2+rows.nbytes+responses.nbytes+2*int(metadata.get('required_bytes',0))
        if self.reserved_bytes+needed>self.memory_budget_bytes:
            raise MemoryError('native component accumulation exceeds total working budget')
        sums=np.asarray(self.accumulator(len(keys),lineage,spectrum,self.compile_program),np.float32)
        if sums.shape!=(len(keys),3) or not np.isfinite(sums).all() or np.any(sums<0):
            raise ValueError('native accumulator returned invalid linear sensor sums')
        self.sums={key:sums[i].copy() for i,key in enumerate(keys)}
        self.camera_weights=defaultdict(int,next_camera)
        self.available_weights=defaultdict(int,next_available)
        self.last_sequence=sequence
        records=[];profile=None
        for key in keys:
            denominator=self.camera_weights[key[:2]]
            valid=self.available_weights[key]
            if valid>denominator:raise ValueError('available camera weight exceeds original weight')
            sensor=self.sums[key]/valid if valid else np.zeros(3,np.float32)
            rgb,profile=sensor_to_linear_srgb(sensor[None,:],self.film_response)
            records.append(dict(pixel=list(key[:2]),recipient_namespace=key[2],recipient_id=key[3],
                component=key[4],linear_rgb=rgb[0].tolist(),availability=valid/denominator if denominator else 0.,
                available_camera_weight=valid,original_camera_weight=denominator,
                rgb_is_availability_premultiplied=False))
        self.latest=self._descriptor(records,metadata,camera_hash,profile)
        self.publish(self.latest)

    def _descriptor(self,records,metadata,camera_hash,profile):
        return dict(schema=VERSION,scene_revision=metadata['scene_revision'],camera_revision=metadata['camera_revision'],
            camera_uniforms_sha256=camera_hash,source_sha256=metadata['dispatch']['source_sha256'],
            sensor_size=list(metadata['dispatch']['camera_uniforms'].get('sensor_size',[])),
            full_sensor_size=list(metadata['dispatch']['camera_uniforms'].get('full_sensor_size',[])),
            sensor_origin=list(metadata['dispatch']['camera_uniforms'].get('sensor_origin',[0,0])),
            publication_sequence=int(metadata['sequence']),field_generation=int(metadata.get('field_generation',0)),working_color_space='linear sRGB D65',
            color_profile=profile,records=records,components=['volumetric','reflectance','glass'],
            rgb_is_availability_premultiplied=False,emission_ownership='included once in copied native contribution; fade original primary once',
            denominator='original camera samples, not response rows',
            glass_composition='RGB=c*F+(1-c)*P; alpha=c+(1-c)*A; original first recipient only',
            specular_override_supported=False,application_ready=False,
            limits=['original selected tracer estimator only','observed source availability is not convergence',
                    'requires exact existing recipient mapping and matching browser camera','no path-factor replay'])
