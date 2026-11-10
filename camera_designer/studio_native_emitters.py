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
    if bands not in (1, 3):
        raise ValueError('Studio native emitter adapter supports fixed:1 and fixed:3')
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
    return np.ascontiguousarray(out)
