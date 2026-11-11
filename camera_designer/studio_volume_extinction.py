"""Explicit world-space extinction input to the existing native ray transport.

This is an optical field binding, not a fluid solver or a humidity-to-fog law.
The cell values are additional achromatic extinction in inverse metres. No
scattering, emission, refractive index, surface or material is manufactured.
"""
from dataclasses import dataclass
import hashlib
import json
import copy
import numpy as np

BINDING = 20
MAX_AXIS = 64
SCHEMA = "studio.volume-extinction.v1"


@dataclass(frozen=True)
class ExtinctionGrid:
    """Piecewise-constant, cell-centred z/y/x coefficients in a world AABB."""
    sigma_t_m_inv: np.ndarray
    bounds_m: np.ndarray
    material_provenance: dict | None = None

    def __post_init__(self):
        sigma = np.array(self.sigma_t_m_inv, dtype="<f4", order="C", copy=True)
        bounds = np.array(self.bounds_m, dtype="<f4", order="C", copy=True)
        if sigma.ndim != 3 or any(not 1 <= n <= MAX_AXIS for n in sigma.shape):
            raise ValueError("volume extinction must have z/y/x dimensions in [1, 64]")
        if not np.isfinite(sigma).all() or np.any(sigma < 0):
            raise ValueError("volume extinction coefficients must be finite and nonnegative")
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("volume extinction bounds must be two finite world xyz points")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            cell = (bounds[1] - bounds[0]) / np.array(sigma.shape[::-1], np.float32)
        if not np.isfinite(cell).all() or np.any(cell <= 0):
            raise ValueError("volume extinction needs positive representable world cell sizes")
        sigma.setflags(write=False)
        bounds.setflags(write=False)
        object.__setattr__(self, "sigma_t_m_inv", sigma)
        object.__setattr__(self, "bounds_m", bounds)
        if self.material_provenance is not None:
            if not isinstance(self.material_provenance, dict):
                raise ValueError('material provenance must be a mapping')
            object.__setattr__(self, 'material_provenance', json.loads(json.dumps(self.material_provenance, allow_nan=False)))

    @classmethod
    def from_density_sampler(cls, sample_at, bounds_m, dimensions_xyz, *, mass_extinction_m2_kg):
        """Publish an existing SI density field using an explicit material law.

        For Turing SPH pass ``fluid.sample_at``. This calls that actual kernel
        reconstruction; it does not interpolate a second particle model. The
        caller must ensure the sampler uses world metres and returns rho in
        kg/m3. Cell-centre publication is not a conservative mass remap.
        """
        dimensions = tuple(dimensions_xyz)
        if len(dimensions) != 3 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 1 <= n <= MAX_AXIS for n in dimensions):
            raise ValueError("density publication dimensions_xyz must be three integers in [1, 64]")
        coefficient = float(mass_extinction_m2_kg)
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("an explicit finite nonnegative mass extinction coefficient is required")
        grid = cls(np.zeros(dimensions[::-1], np.float32), bounds_m)
        axes = [grid.bounds_m[0, k] + (np.arange(n, dtype=np.float64) + .5) *
                (float(grid.bounds_m[1, k]) - float(grid.bounds_m[0, k])) / n
                for k, n in enumerate(dimensions)]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        sigma = np.empty(len(points), np.float64)
        for first in range(0, len(points), 4096):
            chunk = points[first:first + 4096]
            rho = np.asarray(sample_at(chunk)["rho"], np.float64)
            if rho.shape != (len(chunk),) or not np.isfinite(rho).all() or np.any(rho < 0):
                raise ValueError("density sampler must return finite nonnegative rho in kg/m3")
            sigma[first:first + len(chunk)] = rho * coefficient
        return cls(sigma.reshape(dimensions[::-1]), grid.bounds_m)

    def metadata(self):
        h = hashlib.sha256()
        for data in (np.asarray(self.sigma_t_m_inv.shape, "<u4"), self.bounds_m, self.sigma_t_m_inv):
            h.update(data.tobytes())
        metadata = dict(schema=SCHEMA, dimensions_xyz=list(self.sigma_t_m_inv.shape[::-1]),
                    bounds_m=self.bounds_m.tolist(), layout="zyx-f32le", units="m^-1",
                    sha256=h.hexdigest(), reconstruction="piecewise-constant cells",
                    semantics="additional achromatic extinction; zero outside bounds; no added scattering, emission or refractive interface")
        if self.material_provenance is not None:
            metadata['material_provenance'] = copy.deepcopy(self.material_provenance)
        return metadata

    @classmethod
    def from_material_density_sampler(cls, database, material_name, sample_at, bounds_m,
                                      dimensions_xyz, *, wavelengths_m, allow_approximation=False):
        """Use a profile from the central MaterialDatabase, never sprite alpha.

        The supplied sampler must return mass concentration of THIS species in
        kg/m3 of carrier volume. Gas spectra with missing data fail explicitly.
        """
        from material_medium import mass_extinction_from_medium, medium_profile_digest
        profile = database.medium_profile(material_name)
        if profile is None:
            raise ValueError('material has no declared participating-medium profile')
        coefficient = mass_extinction_from_medium(profile, wavelengths_m=wavelengths_m,
                                                  allow_approximation=allow_approximation)
        grid = cls.from_density_sampler(sample_at, bounds_m, dimensions_xyz,
                                       mass_extinction_m2_kg=coefficient)
        provenance = dict(material_name=material_name, profile_sha256=medium_profile_digest(profile),
                          optical_model=profile['optics']['model'], mass_extinction_m2_kg=coefficient,
                          wavelengths_m=list(wavelengths_m), approximation=profile['optics']['approximation'])
        return cls(grid.sigma_t_m_inv, grid.bounds_m, provenance)


def from_payload(config, payload):
    """Local worker contract: declaration and both NPZ arrays must agree."""
    declaration = config.get("volume_extinction")
    keys = {"volume_extinction_sigma_t_m_inv", "volume_extinction_bounds_m"}
    present = keys.intersection(payload.keys())
    if declaration is None:
        if present:
            raise ValueError("volume extinction payload requires an explicit declaration")
        return None
    if declaration != {"schema": SCHEMA}:
        raise ValueError("volume_extinction declaration must specify only its supported schema")
    if present != keys:
        raise ValueError("declared volume extinction requires coefficient and bounds arrays")
    return ExtinctionGrid(payload["volume_extinction_sigma_t_m_inv"], payload["volume_extinction_bounds_m"])


GLSL = r'''
layout(std430,binding=20) readonly buffer StudioVolumeExtinction { float studio_sigma_t[]; };
uniform vec3 uVolumeExtinctionMin;
uniform vec3 uVolumeExtinctionMax;
uniform ivec3 uVolumeExtinctionDims;
// Returns (transmittance, integral of transmittance over the segment in metres).
// Exact for the supplied piecewise-constant cells, including vacuum outside.
vec2 studio_volume_path(vec3 a, vec3 b) {
    vec3 delta=b-a;
    float meters=length(delta);
    if(!(meters>0.0))return vec2(1.0,0.0);
    vec3 d=delta/meters;
    float first=0.0,last=meters;
    for(int k=0;k<3;++k){
        if(d[k]==0.0){
            if(a[k]<uVolumeExtinctionMin[k] || a[k]>=uVolumeExtinctionMax[k])return vec2(1.0,meters);
        }else{
            float ta=(uVolumeExtinctionMin[k]-a[k])/d[k];
            float tb=(uVolumeExtinctionMax[k]-a[k])/d[k];
            first=max(first,min(ta,tb));last=min(last,max(ta,tb));
        }
    }
    if(!(last>first))return vec2(1.0,meters);
    vec3 cell=(uVolumeExtinctionMax-uVolumeExtinctionMin)/vec3(uVolumeExtinctionDims);
    vec3 grid=(a+d*first-uVolumeExtinctionMin)/cell;
    ivec3 q;
    for(int k=0;k<3;++k)q[k]=clamp(int(d[k]<0.0 ? ceil(grid[k])-1.0 : floor(grid[k])),0,uVolumeExtinctionDims[k]-1);
    float cursor=first,transmit=1.0,integral=first;
    // <=64 crossed cells per axis; ties advance all crossed axes.
    for(int iteration=0;iteration<196;++iteration){
        if(any(lessThan(q,ivec3(0))) || any(greaterThanEqual(q,uVolumeExtinctionDims)))break;
        vec3 crossing=vec3(1e30);
        for(int k=0;k<3;++k)if(d[k]!=0.0){
            float edge=uVolumeExtinctionMin[k]+float(q[k]+(d[k]>0.0 ? 1 : 0))*cell[k];
            crossing[k]=(edge-a[k])/d[k];
        }
        float next=max(cursor,min(last,min(crossing.x,min(crossing.y,crossing.z))));
        float span=next-cursor;
        float sigma=studio_sigma_t[(q.z*uVolumeExtinctionDims.y+q.y)*uVolumeExtinctionDims.x+q.x];
        float tau=sigma*span;
        float attenuation=exp(-tau);
        // Stable small-depth integral, with exact zero coefficient limit.
        float weighted= tau<0.001 ? span*(1.0-tau*0.5+tau*tau/6.0) : (1.0-attenuation)/sigma;
        integral+=transmit*weighted;
        transmit*=attenuation;
        if(next>=last)return vec2(transmit,integral+transmit*(meters-last));
        for(int k=0;k<3;++k)if(crossing[k]<=next)q[k]+=(d[k]>0.0 ? 1 : -1);
        cursor=next;
    }
    return vec2(transmit,integral+transmit*(meters-cursor));
}
float studio_volume_transmit(vec3 a,vec3 b){return studio_volume_path(a,b).x;}
'''


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError("canonical volume extinction shader seam changed: " + old[:90])
    return source.replace(old, new)


def transport_shader(source, *, sensor=False):
    """Specialize only explicit-volume jobs. Omitted jobs never call this."""
    # Helpers depend only on their own uniforms, and must precede every caller.
    newline = source.index("\n", source.index("#version")) + 1
    source = source[:newline] + GLSL + source[newline:]
    if sensor:
        source = _replace(source, "float weight=cos_surf*cos_light/dist2*float(uSourceCount)*src.packet.z/PI;",
                          "float weight=cos_surf*cos_light/dist2*float(uSourceCount)*src.packet.z/PI*studio_volume_transmit(p,src.pos_weight.xyz);")
        source = _replace(source, "float geometry = cos_surf / (1.0 + dist2) * float(uSourceCount);",
                          "float geometry = cos_surf / (1.0 + dist2) * float(uSourceCount) * studio_volume_transmit(p,src.pos_weight.xyz);")
        # Attenuate incoming radiance without changing the legacy alpha law.
        source = _replace(source, "out_lum[i] += transmit * a * bv[i];",
                          "out_lum[i] += transmit * a * bv[i] * studio_volume_transmit(ro,pos);")
        source = _replace(source, "float tp_max = 0.0;",
                          "if(hit_id>=0) { float vt=studio_volume_transmit(ro,ro+rd*surf_t); for(int i=0;i<nl;++i)throughput[i]*=vt; }\n            float tp_max = 0.0;")
    else:
        # Deposits use segment-entry energy attenuated only as far as the sample.
        for call, point in (("splat_spectral(p, scatter);", "p"),
                            ("splat_spectral(p, geo_pkt);", "p"),
                            ("splat_spectral(p_mid, wave_pkt);", "p_mid")):
            packet = call.split(", ")[1].split(")")[0]
            source = _replace(source, call, packet + ".z*=studio_volume_transmit(a," + point + "); " + call)
        source = _replace(source, "packet = apply_wave_context_physics(ro, rd, best, packet);",
                          "packet.z*=studio_volume_transmit(ro,hp);\n        packet = apply_wave_context_physics(ro, rd, best, packet);")
        source = _replace(source, "Tri   tri            = ",
                          "packet.z*=studio_volume_transmit(ro,hp);\n        Tri   tri            = ")
        source = _replace(source, "return emitted * geom * bsdf_weight * mis * float(max(1, uSourceCount));",
                          "return emitted * geom * bsdf_weight * mis * float(max(1, uSourceCount)) * studio_volume_transmit(p,src.pos_weight.xyz);")
        source = _replace(source, "float path_weight = 1.0;",
                          "float path_weight = 1.0; float volume_path_weight = 1.0;")
        source = _replace(source, "float trans = exp(-uMediumExtinction * t * march_len);",
                          "float trans = exp(-uMediumExtinction * t * march_len) * volume_path_weight * studio_volume_transmit(ro,ap);")
        source = _replace(source, "splat_activation(ap, air_conn * uSensorGain);",
                          "splat_activation(ap, air_conn * uSensorGain * volume_path_weight * studio_volume_transmit(ro,ap));")
        source = _replace(source, "vec4 fwd = sample_activation_field(hp);",
                          "volume_path_weight*=studio_volume_transmit(ro,hp);\n        vec4 fwd = sample_activation_field(hp);")
        source = _replace(source, "splat_activation(hp, contribution);",
                          "splat_activation(hp, contribution * volume_path_weight);")
        source = _replace(source, "splat_activation(hp, direct * uSensorGain);",
                          "splat_activation(hp, direct * uSensorGain * volume_path_weight);")
        if "void studio_record_segment" in source:
            # Keep entry throughput, integrate attenuation within each recorder cell.
            source = _replace(source, "studio_record_track(q,d,packet,next-cursor);",
                "studio_record_track(q,d,packet,studio_volume_transmit(a,a+d*cursor)*studio_volume_path(a+d*cursor,a+d*next).y);")
    return source


def allocate(grid):
    from OpenGL import GL
    if not isinstance(grid, ExtinctionGrid):
        raise TypeError("volume_extinction must be an ExtinctionGrid or None")
    if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)) <= BINDING:
        raise RuntimeError("volume extinction requires 21 SSBO binding slots")
    handle = int(GL.glGenBuffers(1))
    try:
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, handle)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, grid.sigma_t_m_inv.nbytes, grid.sigma_t_m_inv, GL.GL_STATIC_DRAW)
    except Exception:
        GL.glDeleteBuffers(1, [handle])
        raise
    return {"buffer": handle, "grid": grid}


def bind(record, program):
    """Bind the borrowed field on every dispatch; never rely on ambient state."""
    from OpenGL import GL
    grid = record["grid"]
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, BINDING, record["buffer"])
    GL.glUniform3f(GL.glGetUniformLocation(program, "uVolumeExtinctionMin"), *grid.bounds_m[0])
    GL.glUniform3f(GL.glGetUniformLocation(program, "uVolumeExtinctionMax"), *grid.bounds_m[1])
    GL.glUniform3i(GL.glGetUniformLocation(program, "uVolumeExtinctionDims"), *grid.sigma_t_m_inv.shape[::-1])
