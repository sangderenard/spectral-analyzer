"""Bounded, typed, scene-attached copies of native computed responses.

Storage/indexing only: no response evaluation, estimator mixing, optical replay,
normalization or pixel-image product. Exact native payload bits are retained.
"""
from collections import OrderedDict
import hashlib
import json
import numpy as np
from .studio_directional_field import stencil_directions
from .studio_response_lineage import copied_recipient, recipient_id

SCHEMA='studio-scene-responses-v1'
INDEX_WORDS=20
FAMILIES={1:'forward',2:'reactive',3:'camera_rhs_summands',4:'camera_complete_sample'}
CODECS={1:'evaluated-forward-responses-v1',2:'evaluated-forward-responses-v1',
        3:'native-camera-response-pairs-v1',4:'native-camera-scene-sample-v1'}


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))


class SceneResponseStore:
    """A column ring, not an ever-growing Python object per ray.

    Budget reserves space for publication copies and bounded context metadata.
    Eviction is explicit unavailable data, never a measured zero. New camera
    directions and population refreshes do not reset the physical scene cache.
    """
    def __init__(self, *, scene_revision, memory_budget_bytes, lineage_provider,
                 bounds=None, dimensions=None, recipient_namespaces=(1,2,3), stencil_size=12):
        if not isinstance(scene_revision,str) or not scene_revision:raise ValueError('physical scene revision required')
        if type(memory_budget_bytes) is not int or memory_budget_bytes<32768:raise ValueError('explicit scene-store memory budget required')
        self.scene_revision=scene_revision;self.memory_budget_bytes=memory_budget_bytes
        self.capacity=max(1,(memory_budget_bytes//5)//(1536+80+256))
        # Native-only events must survive even without a raster recipient.
        # Reserve a bounded part of the SAME allocation so misses/partitions
        # cannot displace all surface samples or their unavailable evidence.
        self.native_only_capacity=max(1,self.capacity//8)
        self.surface_capacity=self.capacity-self.native_only_capacity
        self._next_surface=0;self._next_native=self.surface_capacity
        self.evicted_surface_entries=0;self.evicted_native_only_entries=0
        self.native_only_context_discarded_entries=0
        self.index=np.zeros((self.capacity,20),dtype='<u4')
        self.native=np.zeros((self.capacity,384),dtype='<u4')
        self.length=np.zeros(self.capacity,np.uint16)
        self.order=np.zeros(self.capacity,np.uint64)
        self.keys=np.zeros((self.capacity,32),np.uint8)
        self.used=np.zeros(self.capacity,bool)
        # Fixed-size conservative support tombstones: false positives only
        # suppress availability; eviction can never improve confidence.
        self._evicted_support=np.zeros(max(32,self.capacity*16),np.uint8)
        self.evicted_camera_entries=0
        self.lineage_provider=lineage_provider
        self.bounds=None if bounds is None else np.asarray(bounds,np.float64)
        self.dimensions=None if dimensions is None else np.asarray(dimensions,np.int64)
        if self.bounds is not None and (self.bounds.shape!=(2,3) or self.dimensions.shape!=(3,) or np.any(self.bounds[1]<=self.bounds[0]) or np.any(self.dimensions<=0)):
            raise ValueError('invalid existing world index bounds')
        self.recipient_namespaces=frozenset(recipient_namespaces)
        if not self.recipient_namespaces.issubset({1,2,3}):raise ValueError("invalid recipient scope")
        self.omitted_out_of_scope=0
        if stencil_size not in (6,12):raise ValueError("unsupported existing directional stencil")
        self.stencil_size=stencil_size
        self.basis=stencil_directions(stencil_size)
        self._contexts={};self._context_keys={};self._context_refs={};self._context_bytes=0
        self._scalar=np.zeros((max(1,memory_budget_bytes//1280),6),dtype="<u4");self._scalar_metadata=None;self._scalar_count=0;self._scalar_serial=0;self._scalar_bytes=0
        self._context_next=0;self._seen={};self._watermarks={};self._serial=0
        self.scalar_retention=dict(mode='latest cumulative native snapshot; no cross-epoch accumulation',superseded_snapshots=0,discarded_epochs=0,discarded_entries=0,evicted_entries=0,discarded_identity_digest='',recent_discards=[],omitted_discard_identities=0)
        self.evicted=0;self.duplicates=0;self.invalidated=0;self.incomplete_dispatches=0

    @property
    def retained_entries(self):return int(np.count_nonzero(self.used))

    def preflight_capacity(self, capacity, *, live_copy_bytes):
        # Only fixed array views and one bounded per-entry copy are required.
        # Caller additionally reserves this store's declared budget in its
        # shared native/CPU budget. This check is not a coverage threshold.
        if type(capacity) is not int or capacity<0 or live_copy_bytes<0:raise ValueError('invalid dispatch capacity')
        return int(live_copy_bytes)+capacity*16+4096

    def invalidate_scene(self, scene_revision):
        if not isinstance(scene_revision,str) or not scene_revision:raise ValueError('physical scene revision required')
        if scene_revision==self.scene_revision:return
        self.invalidated+=self.retained_entries;self.scene_revision=scene_revision
        self._scalar_metadata=None;self._scalar_count=0;self._scalar_serial=0;self._scalar_bytes=0
        self.used[:]=False;self._contexts.clear();self._context_keys.clear();self._context_refs.clear()
        self._context_bytes=0;self._seen.clear();self._watermarks.clear()
        self._next_surface=0;self._next_native=self.surface_capacity
        self._evicted_support[:]=0;self.evicted_camera_entries=0

    def _begin(self,metadata,family,*,native_only=False):
        scene=metadata.get('scene_revision',metadata.get('scene_sha256'))
        if scene!=self.scene_revision:raise ValueError('stale physical scene response')
        stream=metadata.get('stream_id')
        sequence=metadata.get('sequence')
        if not isinstance(stream,str) or not stream or len(stream)>256 or type(sequence) is not int or sequence<0:
            raise ValueError('native stream ID and monotonic dispatch sequence required')
        if (stream,family) not in self._watermarks and len(self._watermarks)>=self.capacity:
            raise MemoryError('bounded stream identity capacity exhausted')
        watermark=self._watermarks.get((stream,family),-1)
        if sequence<=watermark:raise ValueError('duplicate or stale native dispatch')
        transport=metadata.get('transport_scene_sha256')
        if not isinstance(transport,str) or not transport:raise ValueError('exact assembled transport scene identity required')
        dispatch=metadata.get('dispatch',{})
        source=metadata.get('source_sha256',dispatch.get('source_sha256'))
        if not isinstance(source,str) or not source:raise ValueError('native source identity required')
        context=dict(scene_revision=scene,source_sha256=source,stream_id=stream,sequence=sequence,
            source_kind=metadata.get('source_kind'),source_population_sha256=metadata.get('source_population_sha256'),
            transport_program_sha256=metadata.get('transport_program_sha256'),
            population_id=metadata.get('population_id',metadata.get('field_generation')),
            camera_revision=metadata.get('camera_revision'),dispatch=dispatch,
            transport_scene_sha256=metadata.get('transport_scene_sha256'),
            live_revision=metadata.get('live_revision'),film_response=metadata.get('film_response'),
            reactive_queue=json.loads(_json(metadata.get('reactive_queue'))),
            semantics=('native camera summands; already path-throughput weighted; do not divide prefix or add beside parent total'
                       if family==3 else 'completed native camera sample before pixel averaging; view/prefix-conditioned, not local radiance'
                       if family==4 else 'native weighted transport packet/attribute; not automatically radiance'),
            physical_scene_identity=scene)
        encoded=_json(context).encode()
        if len(encoded)>65536:raise MemoryError('native context exceeds bounded metadata limit')
        digest=hashlib.sha256(encoded).hexdigest()
        if digest in self._context_keys:cid=self._context_keys[digest]
        else:
            charge=len(encoded)*4+512
            if charge>self.memory_budget_bytes//10:
                if native_only:
                    if metadata.get('complete_dispatch_observation') is False:self.incomplete_dispatches+=1
                    return None,(stream,family),sequence
                raise MemoryError('scene context cannot fit declared memory budget')
            while self._context_bytes+charge>self.memory_budget_bytes//10 and self.retained_entries:
                if not self._evict_oldest(native_only=native_only):break
            if self._context_bytes+charge>self.memory_budget_bytes//10:
                if native_only:
                    if metadata.get('complete_dispatch_observation') is False:self.incomplete_dispatches+=1
                    return None,(stream,family),sequence
                raise MemoryError('scene context cannot fit declared memory budget')
            cid=self._context_next;self._context_next+=1
            if cid>0xffffffff:raise OverflowError('scene context index exhausted')
            self._contexts[cid]=(context,digest,charge);self._context_keys[digest]=cid;self._context_refs[cid]=0;self._context_bytes+=charge
        if metadata.get('complete_dispatch_observation') is False:self.incomplete_dispatches+=1
        return cid,(stream,family),sequence

    def _support_hash_positions(self, slot):
        idx=self.index[slot];row=self.native[slot]
        context=self._contexts[int(idx[2])][0]
        dispatch=context.get('dispatch',{})
        camera=dispatch.get('camera_uniforms',{})
        lens={k:v for k,v in camera.items() if k!='eye'}
        # Deliberately includes every renderable component. If any member of
        # this support was discarded, no remaining component may claim its
        # complete original sample denominator. Camera/population sequence is
        # provenance, never an opportunity to erase an eviction.
        token=dict(recipient=idx[4:6].tolist(),cell_bin=idx[16:20].tolist(),
            base_eye=camera.get('eye'),lens=lens,film_response=context.get('film_response'),layer_specs=dispatch.get('layer_specs'),
            scene_revision=self.scene_revision,source_sha256=context.get('source_sha256'),source_kind=context.get('source_kind'),
            transport_program_sha256=context.get('transport_program_sha256'),
            transport_scene_sha256=context.get('transport_scene_sha256'),reactive_queue=context.get('reactive_queue'))
        digest=hashlib.sha256(_json(token).encode()).digest()
        bits=len(self._evicted_support)*8
        return [int.from_bytes(digest[i:i+8],'little')%bits for i in (0,8,16,24)]

    def support_has_eviction(self, slot):
        return all(int(self._evicted_support[bit//8]) & (1<<(bit%8)) for bit in self._support_hash_positions(slot))

    def _drop(self,slot):
        if not self.used[slot]:return
        native_only=slot>=self.surface_capacity
        if self.index[slot,0]==4 and not native_only:
            for bit in self._support_hash_positions(slot):self._evicted_support[bit//8]|=np.uint8(1<<(bit%8))
            self.evicted_camera_entries+=1
        self._seen.pop(self.keys[slot].tobytes(),None)
        cid=int(self.index[slot,2]);self._context_refs[cid]-=1
        if self._context_refs[cid]==0:
            _,digest,charge=self._contexts.pop(cid);self._context_keys.pop(digest,None)
            self._context_refs.pop(cid);self._context_bytes-=charge
        self.used[slot]=False;self.evicted+=1
        if native_only:self.evicted_native_only_entries+=1
        else:self.evicted_surface_entries+=1

    def _evict_oldest(self,*,native_only=False):
        slots=np.flatnonzero(self.used)
        if native_only:slots=slots[slots>=self.surface_capacity]
        if not len(slots):return False
        self._drop(int(slots[np.argmin(self.order[slots])]))
        return True

    def _retention_class(self,recipient,position,position_present):
        # Keep the existing explicit category/endpoint ROI restriction. A
        # missing endpoint is not a point at zero or evidence inside the ROI.
        # Its raw path fields remain intact, with no spatial support claim.
        if recipient[0]!=0 and recipient[0] not in self.recipient_namespaces:return None
        if position_present:
            p=np.asarray(position,np.float64)
            if not np.isfinite(p).all():raise ValueError('invalid present native position')
            if self.bounds is not None and not (np.all(p>=self.bounds[0]) and np.all(p<self.bounds[1])):return None
        return 'native_only' if recipient[0]==0 or not position_present else 'surface'

    def _put(self, *, family,semantic,cid,payload,event_key,recipient,primitive,
             position,position_present,direction,direction_semantics,event_present):
        retention_class=self._retention_class(recipient,position,position_present)
        if retention_class is None:
            self.omitted_out_of_scope+=1;return
        if cid is None:
            self.native_only_context_discarded_entries+=1;return
        digest=hashlib.sha256(event_key).digest()
        if digest in self._seen:self.duplicates+=1;return
        if retention_class=='native_only':
            slot=self._next_native
            self._next_native=self.surface_capacity+(slot-self.surface_capacity+1)%self.native_only_capacity
        else:
            slot=self._next_surface;self._next_surface=(slot+1)%self.surface_capacity
        # Keep the new dispatch context alive while replacing its last old row.
        self._context_refs[cid]+=1
        self._drop(slot)
        words=np.asarray(payload,dtype='<u4').reshape(-1)
        if len(words) not in (128,384):raise ValueError('unsupported native scene payload stride')
        index=np.zeros(20,dtype='<u4');index[:4]=[family,semantic,cid,0]
        index[4]=recipient[0];index[5]=np.asarray(recipient[1],np.int32).view(np.uint32)
        index[6]=np.asarray(primitive,np.int32).view(np.uint32);index[7]=int(event_present)
        index[8:11]=np.asarray(position,np.float32).view(np.uint32)
        finite_position=np.isfinite(np.asarray(position,np.float32)).all()
        index[11]=bool(position_present and finite_position)
        index[12:15]=np.asarray(direction,np.float32).view(np.uint32);index[15]=direction_semantics
        index[16:20]=0xffffffff
        if index[11] and self.bounds is not None:
            cell=np.floor((np.asarray(position)-self.bounds[0])/(self.bounds[1]-self.bounds[0])*self.dimensions).astype(np.int64)
            if np.all(cell>=0) and np.all(cell<self.dimensions):index[16:19]=cell.astype(np.uint32)
        d=np.asarray(direction,np.float64)
        if np.isfinite(d).all() and np.any(d!=0):index[19]=int(np.argmax(self.basis@d))
        self.index[slot]=index;self.native[slot,:len(words)]=words;self.native[slot,len(words):]=0
        self.length[slot]=len(words);self.keys[slot]=np.frombuffer(digest,np.uint8)
        self._serial+=1;self.order[slot]=self._serial;self.used[slot]=True;self._seen[digest]=slot

    def consume_forward(self,metadata,rows):
        array=np.frombuffer(rows,dtype='<u4')
        if array.size%128:raise ValueError('truncated native forward rows')
        array=array.reshape(-1,128)
        if not len(array):return
        if np.any(~np.isin(array[:,1],[1,2])) or not np.isfinite(array[:,21:30].view(np.float32)).all():raise ValueError('invalid forward batch')
        lineage=self.lineage_provider();recipients=np.zeros((len(array),2),np.int64)
        for ordinal,row in enumerate(array):
            primitive=int(row[112]) if metadata.get('primitive_identity_offsets')==[112,113] else -1
            owner=int(row[113].view(np.int32)) if primitive>=0 else -1
            hit=int(row[5].view(np.int32))
            rec=copied_recipient(lineage,hit,primitive,owner) if primitive>=0 else (0,-1)
            recipient_id(*rec);recipients[ordinal]=rec
        native_only=not any(self._retention_class(rec,row[21:24].view(np.float32),int(row[5].view(np.int32))>=0)=='surface'
                            for rec,row in zip(recipients,array))
        cid,watermark,sequence=self._begin(metadata,1,native_only=native_only)
        for ordinal,row in enumerate(array):
            family=int(row[1])
            if family not in (1,2):raise ValueError('unknown original transport path family')
            # These fields are copied by the producer when supplied; unknown
            # primitive lineage stays unmatched rather than guessed by material.
            primitive=int(row[112]) if metadata.get('primitive_identity_offsets')==[112,113] else -1
            owner=int(row[113].view(np.int32)) if primitive>=0 else -1
            hit=int(row[5].view(np.int32))
            recipient=recipients[ordinal]
            self._put(family=family,semantic=int(row[0]),cid=cid,payload=row,
                event_key=f'forward:{metadata["stream_id"]}:{sequence}:{ordinal}'.encode(),recipient=recipient,primitive=primitive,
                position=row[21:24].view(np.float32),position_present=hit>=0,
                direction=row[27:30].view(np.float32),direction_semantics=1,event_present=bool(row[30]))
        self._watermarks[watermark]=sequence
        if cid is not None and self._context_refs.get(cid)==0:
            _,digest,charge=self._contexts.pop(cid);self._context_keys.pop(digest,None)
            self._context_refs.pop(cid);self._context_bytes-=charge

    def consume_camera_samples(self,metadata,rows):
        array=np.frombuffer(rows,dtype='<u4')
        if array.size%128:raise ValueError('truncated completed camera sample rows')
        array=array.reshape(-1,128)
        if not len(array):return
        if not np.isfinite(array[:,8:11].view(np.float32)).all() or not np.isfinite(array[:,12:15].view(np.float32)).all():raise ValueError('invalid camera batch')
        lineage=self.lineage_provider();recipients=np.zeros((len(array),2),np.int64)
        for ordinal,row in enumerate(array):
            hit=int(row[2].view(np.int32));primitive=int(row[3].view(np.int32));owner=int(row[4].view(np.int32))
            rec=copied_recipient(lineage,hit,primitive,owner) if hit>=0 and primitive>=0 else (0,-1)
            recipient_id(*rec);recipients[ordinal]=rec
        native_only=not any(self._retention_class(rec,row[8:11].view(np.float32),bool(row[11]))=='surface'
                            for rec,row in zip(recipients,array))
        cid,watermark,sequence=self._begin(metadata,4,native_only=native_only)
        for ordinal,row in enumerate(array):
            hit=int(row[2].view(np.int32));primitive=int(row[3].view(np.int32));owner=int(row[4].view(np.int32))
            recipient=recipients[ordinal]
            self._put(family=4,semantic=1,cid=cid,payload=row,
                event_key=f'camera:{metadata["stream_id"]}:{sequence}:{int(row[0])}:{int(row[1])}'.encode(),
                recipient=recipient,primitive=primitive,position=row[8:11].view(np.float32),position_present=bool(row[11]),
                direction=row[12:15].view(np.float32),direction_semantics=2,event_present=True)
        self._watermarks[watermark]=sequence
        if cid is not None and self._context_refs.get(cid)==0:
            _,digest,charge=self._contexts.pop(cid);self._context_keys.pop(digest,None)
            self._context_refs.pop(cid);self._context_bytes-=charge

    def __call__(self, metadata, rows):
        return self.consume_forward(metadata, rows)

    def consume_scalar_cells(self, metadata, cells):
        """Consume exact original scalar cell slabs, no sums across snapshots."""
        if metadata.get('scene_revision',metadata.get('scene_sha256'))!=self.scene_revision:
            raise ValueError('stale physical scene scalar snapshot')
        for key in ('scalar_clear_epoch','snapshot_id','cell_offset','total_cells'):
            if key not in metadata:raise ValueError('missing scalar snapshot '+key)
        array=np.asarray(cells)
        if array.dtype!=np.dtype('<u4') or array.ndim!=3 or array.shape[2]!=4 or not 1<=array.shape[1]<=8:
            raise ValueError('scalar cells require exact u32[cell,band,4]')
        offset=metadata['cell_offset'];total=metadata['total_cells']
        if type(offset) is not int or type(total) is not int or offset<0 or offset+len(array)>total:
            raise ValueError('invalid scalar slab range')
        if len(_json(metadata).encode())>min(65536,self.memory_budget_bytes//10):raise MemoryError('scalar snapshot metadata exceeds bounded budget')
        identity=[metadata['scalar_clear_epoch'],metadata['snapshot_id']]
        if self._scalar_metadata is None or self._scalar_metadata['identity']!=identity:
            if offset!=0:raise ValueError('scalar snapshot must start at original cell zero')
            if self._scalar_metadata is not None:
                old=self._scalar_metadata;previous=old['metadata'];r=self.scalar_retention
                changed_epoch=old['identity'][0]!=identity[0]
                r['superseded_snapshots']+=1;r['discarded_epochs']+=int(changed_epoch)
                r['discarded_entries']+=self._scalar_count
                entry=dict(identity=old['identity'],entries=self._scalar_count,source_population_sha256=previous.get('source_population_sha256'),
                    source_sha256=previous.get('source_sha256'),transport_scene_sha256=previous.get('transport_scene_sha256'),epoch_changed=changed_epoch)
                r['discarded_identity_digest']=hashlib.sha256((r['discarded_identity_digest']+_json(entry)).encode()).hexdigest()
                if len(r['recent_discards'])>=8:r['recent_discards'].pop(0);r['omitted_discard_identities']+=1
                r['recent_discards'].append(entry)
            self._scalar_count=0;self._scalar_serial=0;self._scalar_bytes=0
            self._scalar_metadata=dict(identity=identity,total_cells=total,bands=array.shape[1],scanned_cells=0,
                omitted_unvisited=0,evicted_entries=0,complete_snapshot=False,metadata=dict(metadata))
        m=self._scalar_metadata
        if offset!=m['scanned_cells'] or total!=m['total_cells'] or array.shape[1]!=m['bands']:
            raise ValueError('scalar slabs must exactly follow original cell order')
        touched=np.any(array!=0,axis=2)
        m['omitted_unvisited']+=int(touched.size-np.count_nonzero(touched))
        for local,band in zip(*np.nonzero(touched)):
            values=array[local,band];slot=self._scalar_serial%len(self._scalar)
            self._scalar[slot,:2]=[offset+local,band];self._scalar[slot,2:]=values
            self._scalar_serial+=1
            if self._scalar_count==len(self._scalar):
                m['evicted_entries']+=1;self.scalar_retention['evicted_entries']+=1
            else:self._scalar_count+=1
        m['scanned_cells']+=len(array);self._scalar_bytes=self._scalar_count*24
        if metadata.get('final_slab'):
            if m['scanned_cells']!=total:raise ValueError('incomplete final scalar slab')
            m['complete_snapshot']=True

    def snapshot(self, application=None):
        slots=np.flatnonzero(self.used);slots=slots[np.argsort(self.order[slots])]
        index=self.index[slots].copy();native_count=0;rhs_count=0
        for i,slot in enumerate(slots):
            if self.length[slot]==128:index[i,3]=native_count;native_count+=1
            else:index[i,3]=rhs_count;rhs_count+=1
        total=index.nbytes+native_count*512+rhs_count*1536+self._scalar_bytes
        body=bytearray(total);body[:index.nbytes]=index.tobytes()
        native_offset=index.nbytes;rhs_offset=native_offset+native_count*512
        a=native_offset;b=rhs_offset
        for slot in slots:
            size=int(self.length[slot])*4;data=self.native[slot,:int(self.length[slot])].tobytes()
            if size==512:body[a:a+size]=data;a+=size
            else:body[b:b+size]=data;b+=size
        scalar_offset=rhs_offset+rhs_count*1536
        if self._scalar_count:body[scalar_offset:]=self._scalar[:self._scalar_count].tobytes()
        payload=bytes(body)
        contexts=[dict(id=cid,**value[0]) for cid,value in sorted(self._contexts.items()) if self._context_refs[cid]>0]
        manifest=dict(schema=SCHEMA,scene_revision=self.scene_revision,publication_sequence=self._serial,
            response_lineage=json.loads(_json(self.lineage_provider())),
            response_lineage_scene_revision=self.scene_revision,
            byte_order='little',payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
            families={str(k):dict(name=v,codec=CODECS[k]) for k,v in FAMILIES.items()},contexts=contexts,
            sections=[dict(name='scene_index',offset=0,stride=80,count=len(slots)),
                      dict(name='native_rows',offset=native_offset,stride=512,count=native_count),
                      dict(name='camera_rhs_rows',offset=rhs_offset,stride=1536,count=rhs_count),
                      dict(name='scalar_cells',offset=scalar_offset,stride=24,count=self._scalar_count)],
            scalar_snapshot=None if self._scalar_metadata is None else dict(metadata=self._scalar_metadata,retention=self.scalar_retention,
                semantics='original cumulative scalar deposit columns; no directional or material-radiance claim'),
            indexing=dict(semantics='lookup buckets only; exact positions/directions retained',stencil='icosahedron12-v1' if self.stencil_size==12 else 'axes6-v1',
                          bounds=None if self.bounds is None else self.bounds.tolist(),
                          dimensions=None if self.dimensions is None else self.dimensions.tolist()),
            retention=dict(mode='bounded-fifo',capacity=self.capacity,retained_entries=len(slots),
                evicted_entries=self.evicted,duplicate_entries=self.duplicates,invalidated_entries=self.invalidated,
                incomplete_dispatches=self.incomplete_dispatches,evicted_camera_entries=self.evicted_camera_entries,
                application_eviction_policy="bounded conservative support tombstones; false positives unavailable",
                partitions=dict(scheme='surface-native-fifo-v1',surface_capacity=self.surface_capacity,
                    native_only_capacity=self.native_only_capacity,
                    surface_retained_entries=int(np.count_nonzero(self.used[:self.surface_capacity])),
                    native_only_retained_entries=int(np.count_nonzero(self.used[self.surface_capacity:])),
                    surface_evicted_entries=self.evicted_surface_entries,native_only_evicted_entries=self.evicted_native_only_entries,
                    native_only_context_discarded_entries=self.native_only_context_discarded_entries,
                    native_only_application_available=False,
                    scope='known endpoints obey configured bounds/categories; unlocated rows have no claimed spatial support'),
                omitted_out_of_scope=self.omitted_out_of_scope,unavailable_on_eviction=True),
            complete_retained_snapshot=True,physical_completeness_claim=False,application_ready=False,
            rules=['native families and summands are not summed together','presence is not coverage or radiance validity',
                   'camera prefix throughput is retained, not divided out','pixel coordinates are optional provenance, never the scene index key'])
        if application is not None:
            appmeta,approws=application
            apparray=np.asarray(approws)
            if apparray.dtype!=np.dtype('<u4') or apparray.ndim!=2 or apparray.shape[1]!=12:
                raise ValueError('application rows require exact twelve-word ABI')
            manifest['application']=appmeta
            manifest['sections'].append(dict(name='application_rows',offset=len(payload),stride=48,count=len(apparray)))
            payload+=apparray.tobytes();manifest['payload_bytes']=len(payload)
            manifest['payload_sha256']=hashlib.sha256(payload).hexdigest()
        validate_scene_response_snapshot(manifest,payload)
        return manifest,payload


def validate_scene_response_snapshot(manifest,payload):
    if manifest.get('schema')!=SCHEMA or manifest.get('byte_order')!='little':raise ValueError('unsupported scene response envelope')
    if manifest.get('payload_bytes')!=len(payload) or manifest.get('payload_sha256')!=hashlib.sha256(payload).hexdigest():
        raise ValueError('scene response length/hash mismatch')
    sections=manifest.get('sections',[])
    expected={'scene_index':80,'native_rows':512,'camera_rhs_rows':1536,'scalar_cells':24}
    if 'application' in manifest:expected['application_rows']=48
    if [s.get('name') for s in sections]!=list(expected):raise ValueError('scene response section order mismatch')
    if manifest.get('application_ready') is not False:raise ValueError('raw typed rows are not application-ready')
    if manifest.get('families')!={str(k):dict(name=v,codec=CODECS[k]) for k,v in FAMILIES.items()}:raise ValueError('invalid family codecs')
    found={};end=0
    for section in sections:
        name=section.get('name');stride=section.get('stride');count=section.get('count');offset=section.get('offset')
        if name not in expected or name in found or type(offset) is not int or type(stride) is not int or stride!=expected[name] or type(count) is not int or count<0 or offset!=end:
            raise ValueError('invalid/overlapping scene response sections')
        end+=stride*count;found[name]=section
    if set(found)!=set(expected) or end!=len(payload):raise ValueError('incomplete scene response sections')
    if len(_json(manifest).encode())>8*1024*1024:raise ValueError('scene metadata too large')
    for c in manifest.get('contexts',[]):
        if type(c.get('id')) is not int or c['id']<0 or c.get('scene_revision')!=manifest.get('scene_revision') or not isinstance(c.get('source_sha256'),str) or not c['source_sha256'] or not isinstance(c.get('transport_scene_sha256'),str) or not c['transport_scene_sha256']:
            raise ValueError('invalid scene context')
    context_ids={c['id'] for c in manifest.get('contexts',[])}
    if len(context_ids)!=len(manifest.get('contexts',[])):raise ValueError('duplicate context identity')
    index=np.frombuffer(payload,dtype='<u4',count=found['scene_index']['count']*20).reshape(-1,20)
    if manifest.get('retention',{}).get('retained_entries')!=len(index):raise ValueError('retention count mismatch')
    retention=manifest.get('retention',{});partitions=retention.get('partitions')
    if partitions is not None:
        counts=('surface_capacity','native_only_capacity','surface_retained_entries','native_only_retained_entries',
                'surface_evicted_entries','native_only_evicted_entries','native_only_context_discarded_entries')
        if (partitions.get('scheme')!='surface-native-fifo-v1' or
            any(type(partitions.get(k)) is not int or partitions[k]<0 for k in counts) or
            min(partitions['surface_capacity'],partitions['native_only_capacity'])<1 or
            partitions['surface_capacity']+partitions['native_only_capacity']!=retention.get('capacity') or
            partitions['surface_retained_entries']>partitions['surface_capacity'] or
            partitions['native_only_retained_entries']>partitions['native_only_capacity'] or
            partitions['surface_retained_entries']+partitions['native_only_retained_entries']!=len(index) or
            partitions['surface_evicted_entries']+partitions['native_only_evicted_entries']!=retention.get('evicted_entries') or
            partitions.get('native_only_application_available') is not False):
            raise ValueError('invalid bounded native-only retention receipt')
        native_only=(index[:,4]==0)|(index[:,11]==0)
        if int(np.count_nonzero(native_only))!=partitions['native_only_retained_entries']:
            raise ValueError('native-only retention identity mismatch')
    references=set()
    for row in index:
        family=int(row[0]);section='camera_rhs_rows' if family==3 else 'native_rows'
        if family not in FAMILIES or int(row[2]) not in context_ids or int(row[3])>=found[section]['count']:
            raise ValueError('invalid scene entry reference')
        reference=(section,int(row[3]))
        if reference in references:raise ValueError('duplicate native payload reference')
        references.add(reference)
        if int(row[4])>3 or (int(row[4])==0 and row[5]!=0xffffffff) or (int(row[4])!=0 and int(row[5].view(np.int32))<0):raise ValueError('invalid recipient')
        if row[11] and not np.isfinite(row[8:11].view(np.float32)).all():raise ValueError('invalid present position')
        if not np.isfinite(row[12:15].view(np.float32)).all():raise ValueError('invalid direction')
        if int(row[15]) not in (1,2) or int(row[7]) not in (0,1):raise ValueError('invalid scene response semantics')
    if len(references)!=found['native_rows']['count']+found['camera_rhs_rows']['count']:raise ValueError('orphan native payload')
    scalar=manifest.get('scalar_snapshot')
    if scalar is None:
        if found['scalar_cells']['count']:raise ValueError('missing scalar metadata')
    elif not isinstance(scalar.get('metadata'),dict):raise ValueError('scalar metadata missing')
    if 'application' in manifest:
        app=manifest['application']
        if app.get('version')!='native-evaluated-rgb-v1' or app.get('ready') is not False or app.get('rgb_premultiplied') is not False:
            raise ValueError('invalid unactivated native application metadata')
        if app.get('support')!='recipient-cell-stencil-v1' or app.get('rgb')!='linear-srgb':raise ValueError('unsupported application semantics')
        queries={q['id'] for q in app.get('query_contexts',[])}
        if len(queries)!=len(app.get('query_contexts',[])):raise ValueError('duplicate query context')
        sec=found['application_rows']
        app_rows=np.frombuffer(payload,dtype='<u4',offset=sec['offset'],count=sec['count']*12).reshape(-1,12)
        for row in app_rows:
            if int(row[5]) not in queries or row[6]>3 or row[7]>row[8] or row[8]==0 or not np.isfinite(row[9:12].view(np.float32)).all():
                raise ValueError('invalid native application row')
    if manifest.get('physical_completeness_claim') is not False:raise ValueError('scene samples cannot certify physical completeness')
    return True
