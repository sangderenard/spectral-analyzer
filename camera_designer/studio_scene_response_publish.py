"""Bounded publication of retained typed scene responses, not ray history."""
import hashlib,json,os,re,time,uuid
from pathlib import Path
import numpy as np
from .studio_response_product import response_lock,_atomic
from .studio_scene_responses import validate_scene_response_snapshot,SCHEMA,PAIR_SCHEMA,ANCESTRY_SCHEMA
NAME=re.compile(r'^light-scene-response-([0-9a-f]{32})\.bin$')
MAX_BYTES=64*1024*1024
MAX_FILES=3

def lease_path(binary_path):return Path(binary_path).with_suffix('.leases.json')
def sidecar_path(binary_path):return Path(binary_path).with_suffix('.scene.json')

class SceneResponsePublisher:
    def __init__(self,output,digest,*,interval=1.):
        self.output=Path(output);self.digest=digest;self.interval=interval;self.latest=None;self.last=float('-inf');self.sequence=0

    def _room_locked(self):
        entries=[];protected={self.latest['name']} if self.latest else set()
        try:
            status=json.loads((self.output/'status.json').read_text(encoding='utf-8'))
            protected.add(status.get('scene_responses',{}).get('name'))
        except FileNotFoundError:pass
        except (OSError,ValueError):return False
        names=set()
        paths=list(self.output.glob('light-scene-response-*.scene.json'))+list(self.output.glob('light-scene-response-*.scene.retired'))
        for path in paths:
            try:
                if path.is_symlink() or path.stat().st_size>8*1024*1024:return False
                receipt=json.loads(path.read_text(encoding='utf-8'));name=receipt['name']
                if (not isinstance(name,str) or not NAME.fullmatch(name) or name in names
                        or receipt.get('schema') not in (SCHEMA,PAIR_SCHEMA,ANCESTRY_SCHEMA) or receipt.get('digest')!=self.digest
                        or not isinstance(receipt.get('metadata'),dict)
                        or receipt.get('metadata',{}).get('schema')!=receipt.get('schema')
                        or path not in (sidecar_path(self.output/name),sidecar_path(self.output/name).with_suffix('.retired'))
                        or type(receipt.get('byte_length')) is not int or not 0<=receipt['byte_length']<=MAX_BYTES
                        or not isinstance(receipt.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',receipt['sha256'])):
                    return False
                names.add(name)
                entries.append((path.stat().st_mtime_ns,path,receipt))
            except (OSError,ValueError,KeyError,TypeError):return False
        # A binary without its verified receipt is never ours to delete. It also
        # cannot disappear from storage accounting after an interrupted publish.
        if any(path.name not in names for path in self.output.glob('light-scene-response-*.bin')):return False
        for _,sidecar,receipt in sorted(entries):
            retiring=sidecar.suffix=='.retired'
            if len(entries)<MAX_FILES and not retiring:continue
            if receipt['name'] in protected:continue
            target=self.output/receipt['name'];lease=lease_path(target)
            try:
                if lease.is_symlink() or (lease.exists() and lease.stat().st_size>8192):return False
                data=json.loads(lease.read_text(encoding='utf-8'))
                if data.get('version')!=1 or data.get('name')!=receipt['name'] or not isinstance(data.get('transfers'),dict):return False
                if len(data['transfers'])>8 or any(not re.fullmatch(r'[0-9a-f]{32}',key) for key in data['transfers']):return False
                if any(type(t) not in (float,int) or not np.isfinite(t) for t in data['transfers'].values()):return False
                if any(t>time.time() for t in data['transfers'].values()):continue
            except FileNotFoundError:pass
            except (OSError,ValueError,TypeError):return False
            retired=sidecar if retiring else sidecar.with_suffix('.retired')
            try:
                if target.is_symlink():return False
                if target.exists():
                    if target.stat().st_size!=receipt['byte_length']:return False
                    digest=hashlib.sha256()
                    with target.open('rb') as stream:
                        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
                    if digest.hexdigest()!=receipt['sha256']:return False
                elif not retiring:return False
                if not retiring:os.replace(sidecar,retired)
                target.unlink(missing_ok=True);lease.unlink(missing_ok=True);retired.unlink(missing_ok=True)
            except OSError:return False
            entries=[item for item in entries if item[1]!=sidecar]
        return len(entries)<MAX_FILES

    def publish(self,store,*,live_revision=0,now=None):
        now=time.monotonic() if now is None else now
        if now-self.last<self.interval:return self.latest
        # Admission before snapshot allocation; no diagnostic-history spool.
        with response_lock(self.output):
            if not self._room_locked():return self.latest
        metadata,payload=store.snapshot()
        validate_scene_response_snapshot(metadata,payload)
        if len(payload)>MAX_BYTES:raise MemoryError('retained scene snapshot exceeds64MiB; no silent truncation')
        if metadata.get('application_ready') is not False:raise ValueError('unproved scene application must remain explicit')
        with response_lock(self.output):
            if not self._room_locked():return self.latest
            name='light-scene-response-'+uuid.uuid4().hex+'.bin';path=self.output/name
            receipt=dict(schema=metadata['schema'],digest=self.digest,live_revision=int(live_revision),
                sequence=self.sequence+1,application_ready=False,name=name,byte_length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),metadata=metadata)
            receipt_bytes=json.dumps(receipt,separators=(',',':'),allow_nan=False).encode()
            if len(receipt_bytes)>8*1024*1024:raise MemoryError('scene metadata exceeds8MiB admission')
            try:
                _atomic(path,payload)
                _atomic(sidecar_path(path),receipt_bytes)
            except BaseException:path.unlink(missing_ok=True);raise
            self.latest=receipt;self.sequence+=1;self.last=now
            return receipt
