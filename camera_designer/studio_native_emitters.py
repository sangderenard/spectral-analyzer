"""Adapt original native emissive-triangle roots to the existing Studio ABI.

No light list or sampling implementation lives here. Positions, directions,
wavelengths and spectral beta come from RayTracer.submit_emissive_triangles.
Submit triangles separately because native domains otherwise allocate by UV
area, which is not world area for arbitrary uploaded meshes. The existing
MatBuf is shared unchanged with the camera and native sampler.
"""
from __future__ import annotations
import time
import numpy as np
from bdpt_records import BDPT_VERTEX_RECORD_DTYPE, BDPT_SPECTRAL_RECORD_DTYPE

VERSION = 'native-emissive-triangle-roots-v1'


def sampling_settings(config):
    value=config.get('emitter_sampling')
    if value is None:return dict(version=1,ray_budget=4096,seed=42,refresh_each_sweep=False)
    if not isinstance(value,dict) or set(value)!={'version','ray_budget','seed','refresh_each_sweep'}:raise ValueError('invalid emitter_sampling keys')
    if type(value['version']) is not int or value['version']!=1:raise ValueError('invalid emitter_sampling version')
    if type(value['ray_budget']) is not int or not 128<=value['ray_budget']<=65536:raise ValueError('emitter ray_budget must be128..65536')
    if type(value['seed']) is not int or not 0<=value['seed']<=2147483647:raise ValueError('emitter seed must be0..2147483647')
    if type(value['refresh_each_sweep']) is not bool:raise ValueError('emitter refresh_each_sweep must be boolean')
    if value['refresh_each_sweep'] and not (config.get('live') and config.get('transport')=='fixed:3' and 'fixtures' in config and config.get('receive_light_field') and config.get('light_field_estimator')=='track-length-root-sweep-v1'):
        raise ValueError('emitter refresh requires live fixed:3 fixture root-sweep field')
    return dict(value)


def native_emissive_sources(scene, wavelengths_um, *, ray_budget=4096, seed=42):
    """Return SourceRec12 rows; reserved dir_kind.w is scene triangle ID + 1.

    A minimum complete spectral cycle is allocated per emitting triangle.
    All counts are multiples of the band count. Requested budget is a target;
    the returned count is explicit and is not used as a radiometric gain.
    Native beta already contains area / count and wavelength compensation.
    """
    import _spectral_kernels as native
    geom, shade, materials, triangles = scene[:4]
    frequencies = 299792458. / (np.asarray(wavelengths_um, np.float64) * 1e-6)
    bands = len(frequencies)
    if bands not in (1, 3, 4, 8):
        raise ValueError('Studio native emitter adapter supports fixed:1, fixed:3, fixed:4 and fixed:8')
    mat_ids = np.ascontiguousarray(shade[:,14]).view(np.uint32)
    flags = np.ascontiguousarray(geom[:,15]).view(np.uint32)
    selected = np.flatnonzero((flags & int(native.MAT_FLAG_EMISSIVE)) != 0)
    if not len(selected):
        return np.zeros((0,12), np.float32)
    table = np.asarray(materials, np.float32).reshape(-1,32,12)
    emission = table[mat_ids[selected],:bands,5].astype(np.float64)
    native_triangles=np.asarray(triangles,np.float64)
    area = .5 * np.linalg.norm(np.cross(native_triangles[selected,1]-native_triangles[selected,0],
                                      native_triangles[selected,2]-native_triangles[selected,0]),axis=1)
    active = (area > 0) & (emission.sum(axis=1) > 0)
    selected, emission, area = selected[active], emission[active], area[active]
    if not len(selected):
        return np.zeros((0,12), np.float32)
    weights = area * emission.sum(axis=1)
    counts = np.maximum(1, np.ceil(int(ray_budget)*weights/weights.sum()/bands)).astype(int)*bands
    if counts.sum() > 262144:
        raise ValueError('native emitter sample budget exceeds 262144 roots')
    tracer = native.RayTracer(len(selected), np.asarray(triangles[selected],np.float64).reshape(-1,9),
        np.asarray(geom[selected,12:15],np.float64), mat_ids[selected].astype(np.int32),
        np.asarray(materials,np.float64),len(table),frequencies,299792458.,np.zeros(bands))
    try:
        tracer.set_tri_ior(0,len(selected),int(native.MAT_FLAG_EMISSIVE))
        expected = 0
        for i, count in enumerate(counts):
            expected += tracer.submit_emissive_triangles(np.array([i],np.int32),int(count),
                max_bounces=1,min_amplitude=0.,max_children=1,seed=int(seed),
                use_gpu_compute=False,gpu_all_stages=False)
        deadline=time.monotonic()+10.
        while tracer.in_flight_count() and time.monotonic()<deadline:
            time.sleep(.001)
        if tracer.in_flight_count():
            raise RuntimeError('native emissive sample preparation exceeded 10 seconds')
        vertices=np.frombuffer(np.asarray(tracer.drain_bdpt_vertices(expected*3+1)).tobytes(),
                               dtype=BDPT_VERTEX_RECORD_DTYPE)
        spectra=np.frombuffer(np.asarray(tracer.drain_bdpt_spectral(expected*bands*3+1)).tobytes(),
                              dtype=BDPT_SPECTRAL_RECORD_DTYPE)
        roots=vertices[(vertices['vertex_index']==0)&(vertices['stream']==0)]
        spectra=spectra[(spectra['vertex_index']==0)&(spectra['beta_re']>0)]
        if len(roots)!=expected or len(spectra)!=expected:
            raise RuntimeError('native emitter root records are incomplete')
        roots=np.sort(roots,order='subpath_id');spectra=np.sort(spectra,order='subpath_id')
        if not np.array_equal(roots['subpath_id'],spectra['subpath_id']):
            raise RuntimeError('native emitter spectral/root identity mismatch')
        totals=np.zeros((len(selected),bands),np.float64)
        np.add.at(totals,(roots['tri_id'],spectra['band_id']),spectra['beta_re'])
        expected_energy=area[:,None]*emission
        tolerance=64*np.finfo(np.float32).eps*np.maximum(expected_energy,np.finfo(np.float32).tiny)
        if np.any(abs(totals-expected_energy)>tolerance):
            raise RuntimeError('native emitter spectral energy check failed; rebuild the corrected native emitter module')
        rows=np.zeros((expected,12),np.float32)
        rows[:,:3]=roots['pos'];rows[:,3]=1.;rows[:,4:7]=roots['dir_out']
        rows[:,7]=selected[roots['tri_id']]+1
        rows[:,8]=spectra['frequency_hz'];rows[:,10]=spectra['beta_re']
        if not np.isfinite(rows).all():
            raise RuntimeError('native emitter produced nonfinite roots')
        return rows
    finally:
        tracer.stop_pipeline()


def reweight_native_sources(roots, shade, baseline_materials, current_materials):
    """Retain native root identity while applying a slice's actual light state.

    Native source preparation fixes the spatial and wavelength sample set.  A
    camera slice may change the source energy without changing those samples:
    each root's original beta is multiplied by current/baseline MatBuf emission
    at its own triangle and frequency.  The caller must also install the same
    current MatBuf for camera paths and area NEE before dispatching that slice.
    A newly emitting band absent from the prepared population is refused.
    """
    rows=np.asarray(roots,np.float32)
    if rows.ndim!=2 or rows.shape[1]!=12 or not np.isfinite(rows).all():
        raise ValueError('native source roots must be finite SourceRec12 rows')
    base=np.asarray(baseline_materials,np.float32)
    current=np.asarray(current_materials,np.float32)
    if base.shape!=current.shape or base.ndim!=2 or base.shape[1]!=12 or len(base)%32:
        raise ValueError('slice MatBuf must retain the prepared material layout')
    if not np.isfinite(base).all() or not np.isfinite(current).all():
        raise ValueError('slice MatBuf must be finite')
    out=rows.copy()
    if not len(rows):return out
    ids=rows[:,7].astype(np.int64)-1
    if np.any(ids<0) or np.any(ids>=len(shade)) or not np.array_equal(ids+1,rows[:,7]):
        raise ValueError('native roots must retain triangle identities')
    mids=np.ascontiguousarray(np.asarray(shade)[ids,14]).view(np.uint32)
    if np.any(mids>=len(base)//32):raise ValueError('native root material is outside MatBuf')
    base_table=base.reshape(-1,32,12)[mids]
    current_table=current.reshape(-1,32,12)[mids]
    valid=base_table[:,:,0]>0
    nearest=np.argmin(np.where(valid,abs(base_table[:,:,0]-rows[:,None,8]),np.inf),axis=1)
    indices=np.arange(len(rows))
    old=base_table[indices,nearest,5]
    new=current_table[indices,nearest,5]
    if np.any(~valid[indices,nearest]) or np.any(old<=0) or np.any(new<0):
        raise ValueError('slice emission needs prepared positive spectral roots')
    if np.any((current_table[:,:,5]>0)&(base_table[:,:,5]<=0)&valid):
        raise ValueError('slice introduced an unprepared emitting band')
    out[:,10]=(rows[:,10].astype(np.float64)*(new.astype(np.float64)/old.astype(np.float64))).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('slice source beta is nonfinite')
    return out


def timed_light_buffers(baseline_scene, current_scene):
    """Map the current emitter powers onto stable prepared native material IDs."""
    base_geom,base_shade,base_mat,base_triangles=baseline_scene[:4]
    geom,shade,mat,triangles=current_scene[:4]
    if geom.shape!=base_geom.shape or triangles.shape!=base_triangles.shape:
        raise ValueError('timed geometry slice changed packed triangle ownership')
    base_ids=np.ascontiguousarray(base_shade[:,14]).view(np.uint32)
    current_ids=np.ascontiguousarray(shade[:,14]).view(np.uint32)
    adjusted_mat=np.array(base_mat,copy=True).reshape(-1,32,12)
    current_table=np.asarray(mat,np.float32).reshape(-1,32,12)
    for base_id in np.unique(base_ids):
        peers=np.unique(current_ids[base_ids==base_id])
        if len(peers)!=1 or peers[0]>=len(current_table):
            raise ValueError('timed light frame split a prepared material identity')
        old=adjusted_mat[base_id]
        new=current_table[peers[0]]
        if not np.array_equal(old[:,0],new[:,0]):
            raise ValueError('timed light frame changed spectral band identity')
        old[:,5]=new[:,5]
    adjusted_mat=adjusted_mat.reshape(-1,12)
    adjusted_shade=np.array(shade,copy=True)
    adjusted_shade[:,14]=base_shade[:,14]
    roots=reweight_native_sources(baseline_scene[7],base_shade,base_mat,adjusted_mat)
    if not np.array_equal(triangles,base_triangles):
        roots=relocate_native_sources(roots,base_triangles,triangles)
    return adjusted_shade,adjusted_mat,roots


def relocate_native_sources(roots, old_triangles, new_triangles):
    """Carry prepared emitter samples through a triangle slice's local frame."""
    out=np.array(roots,copy=True)
    if not len(out):return out
    ids=out[:,7].astype(np.int64)-1
    if np.any(ids<0) or np.any(ids>=len(old_triangles)):
        raise ValueError('timed source triangle identity left the geometry slice')
    old=np.asarray(old_triangles,np.float64)[ids]
    new=np.asarray(new_triangles,np.float64)[ids]
    def basis(tris):
        edge=tris[:,1]-tris[:,0]
        other=tris[:,2]-tris[:,0]
        normal=np.cross(edge,other)
        area2=np.linalg.norm(normal,axis=1)
        edge_length=np.linalg.norm(edge,axis=1)
        if np.any(area2<=1e-12) or np.any(edge_length<=1e-12):
            raise ValueError('timed emitter triangle became degenerate')
        tangent=edge/edge_length[:,None]
        normal=normal/area2[:,None]
        return tangent,np.cross(normal,tangent),normal,area2
    old_u,old_v,old_n,old_area=basis(old)
    new_u,new_v,new_n,new_area=basis(new)
    offset=out[:,:3].astype(np.float64)-old[:,0]
    edge1=old[:,1]-old[:,0]
    edge2=old[:,2]-old[:,0]
    a=np.einsum('ij,ij->i',edge1,edge1)
    b=np.einsum('ij,ij->i',edge1,edge2)
    c=np.einsum('ij,ij->i',edge2,edge2)
    d=np.einsum('ij,ij->i',offset,edge1)
    e=np.einsum('ij,ij->i',offset,edge2)
    determinant=a*c-b*b
    if np.any(determinant<=1e-24):raise ValueError('timed emitter barycentric frame is singular')
    v=(d*c-e*b)/determinant
    w=(e*a-d*b)/determinant
    out[:,:3]=(new[:,0]+v[:,None]*(new[:,1]-new[:,0])+w[:,None]*(new[:,2]-new[:,0])).astype(np.float32)
    direction=out[:,4:7].astype(np.float64)
    direction=(np.einsum('ij,ij->i',direction,old_u)[:,None]*new_u+
               np.einsum('ij,ij->i',direction,old_v)[:,None]*new_v+
               np.einsum('ij,ij->i',direction,old_n)[:,None]*new_n)
    out[:,4:7]=(direction/np.linalg.norm(direction,axis=1)[:,None]).astype(np.float32)
    out[:,10]=(out[:,10].astype(np.float64)*new_area/old_area).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('timed emitter relocation is nonfinite')
    return out


def install_timed_light_frame(field, sensor, baseline_scene, current_scene, *,
                              installed_scene=None, renderer=None, sphere_regions=None):
    """Install one captured light state in both native readers at a sweep boundary.

    The sensor texture and its pass counter remain owned by the exposure.  The
    forward field is rebuilt by the caller for this interval only.
    """
    if sensor._row_off != 0:
        raise RuntimeError('timed light state requires a complete sensor sweep')
    base_geom,_,_,base_triangles=baseline_scene[:4]
    adjusted_shade,adjusted_mat,roots=timed_light_buffers(baseline_scene,current_scene)
    if len(roots)!=field['n_sources'] or len(roots)!=sensor._nee_source_count:
        raise ValueError('timed light frame changed the prepared source population')
    installed=installed_scene if installed_scene is not None else baseline_scene
    geometry_changed=(not np.array_equal(current_scene[0][:,:15],installed[0][:,:15]) or
                      not np.array_equal(current_scene[3],installed[3]))
    if geometry_changed:
        if renderer is None or sphere_regions is None:
            raise ValueError('timed geometry slice needs the native scene uploader')
        from material_db import MaterialDatabase
        nodes,ids=renderer._build_gpu_bvh(current_scene[3])
        margin=np.asarray([.02,.02,.04],np.float32)
        bounds=(current_scene[6][0]-margin,current_scene[6][1]+margin)
        renderer._update_prebuilt_geometry(field,current_scene[0],adjusted_shade,adjusted_mat,
            nodes,ids,current_scene[4],bounds,sphere_regions,
            native_pbr=MaterialDatabase.instance().build_tensors()['pbr'])
        sensor.set_scene_geometry(n_tris=len(current_scene[3]),n_bvh_nodes=len(nodes),
            bmin=bounds[0],bmax=bounds[1],n_spheres=len(sphere_regions))
    from OpenGL import GL
    GL.glFinish()
    for binding,data in ((9,adjusted_shade),(10,adjusted_mat),(5,roots)):
        data=np.ascontiguousarray(data,np.float32)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,field['ssbo'][binding])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,0)
    sensor.set_nee_spectral_sources(roots,field['ssbo'][5],area_scene=(current_scene[0],adjusted_shade,adjusted_mat,current_scene[3]))
    field['source_buf']=roots
    return roots


def area_nee_records(sources, geom, shade, materials, triangles):
    """Camera view of the same native roots, not another source population.

    Camera-only row: xyz position; [3,7,11] actual emission normal;
    [4:7] native MatBuf emission at 650/550/450 nm; [10] area / roots on
    this triangle. Uniform root selection then has density count/(N*area).
    Never pass this camera view to forward transport.
    """
    rows=np.asarray(sources,np.float32)
    out=rows.copy()
    if not len(rows):return out
    ids=rows[:,7].astype(np.int64)-1
    if np.any(ids<0) or np.any(ids>=len(geom)) or not np.array_equal(ids+1,rows[:,7]):
        raise ValueError('area NEE requires native source triangle identities')
    unique,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
    normals=geom[ids,12:15]
    out[:,3]=normals[:,0];out[:,7]=normals[:,1];out[:,11]=normals[:,2]
    native_triangles=np.asarray(triangles,np.float64)
    area=.5*np.linalg.norm(np.cross(native_triangles[unique,1]-native_triangles[unique,0],
                                   native_triangles[unique,2]-native_triangles[unique,0]),axis=1)
    out[:,10]=area[inv]/counts[inv]
    mids=np.ascontiguousarray(shade[ids,14]).view(np.uint32)
    table=np.asarray(materials,np.float32).reshape(-1,32,12)
    for band,wavelength in enumerate((650.,550.,450.)):
        frequency=299792458./(wavelength*1e-9)
        valid=table[mids,:,0]>0
        distance=np.where(valid,abs(table[mids,:,0]-frequency),np.inf)
        nearest=np.argmin(distance,axis=1)
        out[:,4+band]=table[mids,nearest,5]
    out[:,9]=ids+1  # camera-only packet.y: native emitting triangle identity
    return np.ascontiguousarray(out)


def install_source_population(renderer,field,sensor,scene,wavelengths,settings,index,*,ray_jobs):
    """Refresh the original forward SSBO and its camera-only NEE view atomically at a sweep boundary."""
    from OpenGL import GL
    from .studio_directional_field import scene_digest
    seed=(settings['seed']+index)%2147483648
    rows=native_emissive_sources(scene,wavelengths,ray_budget=settings['ray_budget'],seed=seed)
    if len(rows)!=field['n_sources']:raise ValueError('refreshed source count changed; new candidate required')
    if len(rows)>ray_jobs:raise ValueError('live batch does not cover refreshed source population')
    # Complete the old sweep before changing either consumer. Failure propagates;
    # no new pump or purported complete publication follows a partial install.
    GL.glFinish()
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,field['ssbo'][5])
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,rows.nbytes,rows,GL.GL_STATIC_DRAW)
    sensor.set_nee_spectral_sources(rows,field['ssbo'][5],area_scene=scene)
    field['source_buf']=rows
    if field.get('forward_response_owner') is not None:
        field['response_identity']['population_index']=int(index)
        field['response_identity']['source_population_seed']=int(seed)
    record=field['directional_record']
    # Stable geometry + current population; aggregate provenance is separate.
    import hashlib
    base=record.setdefault('source_population_scene_base',record['scene_sha256'])
    record['scene_sha256']=hashlib.sha256(bytes.fromhex(base)+rows.tobytes()).hexdigest()
    return rows
