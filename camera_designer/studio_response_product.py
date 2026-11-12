"""Bounded publication of native copied component results, never raw history."""
import contextlib,hashlib,json,os,re,time,uuid
from pathlib import Path
import numpy as np
from .studio_response_lineage import recipient_id
ROW=np.dtype([('pixel','<u4',(2,)),('recipient','<u4'),('component','<u4'),('rgb','<f4',(3,)),('availability','<f4'),('available','<u4'),('samples','<u4')])
assert ROW.itemsize==40
NAME=re.compile(r'^light-response-([0-9a-f]{32})\.bin$')
MAX_BYTES=64*1024*1024
MAX_FILES=3
LEASE_SECONDS=600


@contextlib.contextmanager
def response_lock(directory,timeout=5):
    import msvcrt
    path=Path(directory)/'light-response.lock'
    with path.open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        deadline=time.monotonic()+timeout
        while True:
            try:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                if time.monotonic()>=deadline:raise TimeoutError('response artifact lock busy')
                time.sleep(.01)
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def encode_product(product):
    if product.get('schema')!='studio-copied-components-v1' or product.get('application_ready') is not False:
        raise ValueError('unsupported or unproved native component product')
    records=product['records']
    if len(records)*40>MAX_BYTES:raise ValueError('complete component product exceeds64MiB; no rows dropped')
    rows=np.zeros(len(records),ROW);components={'volumetric':0,'reflectance':1,'glass':2}
    for i,r in enumerate(records):
        pixel=r['pixel'];weights=(r['available_camera_weight'],r['original_camera_weight'])
        if len(pixel)!=2 or any(type(v) is not int or not 0<=v<=0xffffffff for v in [*pixel,*weights]):raise ValueError('invalid copied component counters')
        if weights[0]>weights[1] or r.get('rgb_is_availability_premultiplied') is not False:raise ValueError('invalid copied component weighting')
        rgb=np.asarray(r['linear_rgb'],np.float32);alpha=float(r['availability'])
        if rgb.shape!=(3,) or not np.isfinite(rgb).all() or not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('invalid copied component values')
        expected=weights[0]/weights[1] if weights[1] else 0.
        if abs(alpha-expected)>1e-12:raise ValueError('availability does not match copied camera weights')
        rows[i]=(pixel,recipient_id(r['recipient_namespace'],r['recipient_id']),components[r['component']],rgb,alpha,*weights)
    return rows.tobytes(),{key:value for key,value in product.items() if key!='records'}


def _atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    try:
        with temp.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


class ResponseProductPublisher:
    def __init__(self,output,digest,*,interval=1.):
        self.output=Path(output);self.digest=digest;self.interval=interval;self.latest=None;self.last=float('-inf');self.sequence=0

    def _room_locked(self):
        entries=[];protected={self.latest['name']} if self.latest else set()
        try:
            status=json.loads((self.output/'status.json').read_text(encoding='utf-8'))
            protected.add(status.get('response_components',{}).get('name'))
        except FileNotFoundError:pass
        except (OSError,ValueError):return False
        for path in self.output.glob('light-response-*.components.json'):
            if path.is_symlink() or path.stat().st_size>1024*1024:return False
            try:
                receipt=json.loads(path.read_text(encoding='utf-8'));name=receipt['name']
                if not NAME.fullmatch(name) or receipt['digest']!=self.digest:return False
                entries.append((path.stat().st_mtime_ns,path,receipt))
            except (OSError,ValueError,KeyError):return False
        for _,sidecar,receipt in sorted(entries):
            if len(entries)<MAX_FILES:return True
            if receipt['name'] in protected:continue
            target=self.output/receipt['name'];lease=target.with_suffix('.leases.json')
            try:
                data=json.loads(lease.read_text(encoding='utf-8'))
                if data.get('version')!=1 or data.get('name')!=receipt['name'] or not isinstance(data.get('transfers'),dict):return False
                if any(type(t) not in (float,int) or not np.isfinite(t) for t in data['transfers'].values()):return False
                if any(t>time.time() for t in data['transfers'].values()):continue
            except FileNotFoundError:pass
            except (OSError,ValueError):return False
            retired=sidecar.with_suffix('.retired')
            try:
                os.replace(sidecar,retired);target.unlink(missing_ok=True);lease.unlink(missing_ok=True);retired.unlink(missing_ok=True)
            except OSError:return False
            entries=[item for item in entries if item[1]!=sidecar]
        return len(entries)<MAX_FILES

    def publish(self,product,*,live_revision=0,now=None):
        now=time.monotonic() if now is None else now
        if now-self.last<self.interval:return self.latest
        payload,metadata=encode_product(product)
        with response_lock(self.output):
            if not self._room_locked():return self.latest
            name='light-response-'+uuid.uuid4().hex+'.bin';path=self.output/name
            receipt=dict(schema='studio.response-components.v1',digest=self.digest,live_revision=int(live_revision),sequence=self.sequence+1,
                application_ready=False,name=name,byte_length=len(payload),sha256=hashlib.sha256(payload).hexdigest(),count=len(payload)//40,
                stride_bytes=40,format='copied-components-f32-u32le',metadata=metadata)
            try:
                _atomic(path,payload);_atomic(path.with_suffix('.components.json'),json.dumps(receipt,separators=(',',':'),allow_nan=False).encode())
            except BaseException:path.unlink(missing_ok=True);raise
            self.latest=receipt;self.sequence+=1;self.last=now
            return receipt
