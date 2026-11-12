"""Atomic opt-in field package publication; uses existing authenticated artifacts."""
import hashlib,json,os,time,uuid
from pathlib import Path
from camera_designer.studio_field_package import (SCHEMA,MAX_TOTAL_BYTES,MAX_CHUNK_BYTES,MAX_PACKAGES,
    MAX_MANIFEST_BYTES,chunk_name,manifest_name,lease_name,validate_manifest)


def _atomic(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def package_room(directory,protected=()):
    """No transfer lease may appear between retirement decision and retirement."""
    from camera_designer.studio_field_package import package_lock
    directory=Path(directory);protected=set(protected);packages=[]
    try:
        status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
        protected.add(status.get('light_field',{}).get('package_id'))
    except FileNotFoundError:pass
    except (ValueError,OSError):return False
    for path in directory.glob('light-field-*.manifest.json'):
        try:
            if path.is_symlink() or path.stat().st_size>MAX_MANIFEST_BYTES:return False
            data=json.loads(path.read_text(encoding='utf-8'));validate_manifest(data)
            packages.append((path.stat().st_mtime_ns,path,data))
        except (OSError,ValueError):return False
    for _,path,manifest in sorted(packages):
        if len(packages)<MAX_PACKAGES:return True
        package=manifest['package_id']
        if package in protected:continue
        with package_lock(directory,package):
            lease=directory/lease_name(package)
            try:
                value=json.loads(lease.read_text(encoding='utf-8'))
                if value.get('version')!=1 or value.get('package_id')!=package or not isinstance(value.get('transfers'),dict):return False
                if any(type(expiry) not in (int,float) or not __import__('math').isfinite(expiry) for expiry in value['transfers'].values()):return False
                if any(expiry>time.time() for expiry in value['transfers'].values()):continue
            except FileNotFoundError:pass
            except (OSError,ValueError):return False
            # Retirement is atomic before deleting any member; new transfers
            # under this same lock cannot acquire a partially removed package.
            retired=path.with_suffix('.retired')
            try:
                os.replace(path,retired)
                for member in manifest['chunks']:(directory/member['name']).unlink(missing_ok=True)
                lease.unlink(missing_ok=True);retired.unlink(missing_ok=True)
            except OSError:return False
            packages=[item for item in packages if item[1]!=path]
    return len(packages)<MAX_PACKAGES


def publish_package(directory,*,digest,live_revision,sequence,scene_sha256,config_sha256,sections,protected=()):
    """Write bounded immutable chunks; advertise only an entirely validated manifest."""
    directory=Path(directory)
    if not package_room(directory,protected):return None
    total=sum(s['byte_length'] for s in sections)
    if total<=0 or total>MAX_TOTAL_BYTES:raise ValueError('complete package exceeds512MiB; nothing truncated')
    package=uuid.uuid4().hex;chunks=[];descriptors=[];written=[];offset=0
    try:
        for section in sections:
            stride=section['stride_bytes'];capacity=(MAX_CHUNK_BYTES//stride)*stride
            descriptors.append({k:section[k] for k in ('id','format','stride_bytes','count','byte_length','metadata')})
            descriptors[-1]['offset_bytes']=offset;offset+=section['byte_length']
            pending=bytearray();local=0
            def write_chunk(data):
                nonlocal local
                name=chunk_name(package,len(chunks));path=directory/name
                _atomic(path,data);written.append(path)
                chunks.append(dict(name=name,section=section['id'],offset_bytes=local,byte_length=len(data),sha256=hashlib.sha256(data).hexdigest(),first_record=local//stride,record_count=len(data)//stride))
                local+=len(data)
            for block in section['blocks']:
                if not isinstance(block,(bytes,bytearray,memoryview)) or len(block)%stride:raise ValueError('section block is not record aligned')
                cursor=0
                while cursor<len(block):
                    n=min(capacity-len(pending),len(block)-cursor);pending.extend(block[cursor:cursor+n]);cursor+=n
                    if len(pending)==capacity:write_chunk(bytes(pending));pending.clear()
            if pending:write_chunk(bytes(pending))
            if local!=section['byte_length']:raise ValueError('incomplete section stream')
        manifest=dict(schema=SCHEMA,package_id=package,digest=digest,live_revision=int(live_revision),sequence=int(sequence),scene_sha256=scene_sha256,config_sha256=config_sha256,byte_length=total,sections=descriptors,chunks=chunks)
        validate_manifest(manifest)
        _atomic(directory/manifest_name(package),json.dumps(manifest,separators=(',',':'),allow_nan=False).encode())
        return manifest
    except BaseException:
        # No manifest exists, hence these newly created members were never
        # advertised or leasable. Never touch another publication's members.
        for path in written:path.unlink(missing_ok=True)
        raise
