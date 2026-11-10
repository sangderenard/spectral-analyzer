"""NumPy mirrors of the native BDPT record ABI, without loading a UI."""
import numpy as np

BDPT_VERTEX_RECORD_DTYPE = np.dtype([
    ('subpath_id','u4'), ('vertex_index','u2'), ('stream','u1'), ('sample_domain','u1'),
    ('flags','u4'), ('strategy_id','u4'), ('tri_id','i4'), ('group_id','i4'), ('mat_idx','i4'),
    ('pos','f4',(3,)), ('normal','f4',(3,)), ('dir_in','f4',(3,)), ('dir_out','f4',(3,)),
    ('path_len','f4'), ('path_at_seg_start','f4'), ('pdf_fwd','f4'), ('pdf_rev','f4'),
    ('pdf_area','f4'), ('pdf_solid_angle','f4'), ('throughput_scalar','f4'),
    ('sensor_origin_y','f4'), ('sensor_origin_z','f4'),
])
BDPT_SPECTRAL_RECORD_DTYPE = np.dtype([
    ('subpath_id','u4'), ('vertex_index','u2'), ('band_id','u2'),
    ('beta_re','f4'), ('beta_im','f4'), ('frequency_hz','f4'), ('band_pdf','f4'),
    ('sensor_rgb_weight','f4'), ('spectral_sample_id','u4'),
])
assert BDPT_VERTEX_RECORD_DTYPE.itemsize == 112
assert BDPT_SPECTRAL_RECORD_DTYPE.itemsize == 32
