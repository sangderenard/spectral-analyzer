import ctypes

class FilmRecord(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ('iso', ctypes.c_float),  # 1
        ('exposure_time_s', ctypes.c_float),  # 2
        ('quantum_efficiency', ctypes.c_float),  # 3
        ('target_grey_point', ctypes.c_float),  # 4
        ('layer0_shadow_r', ctypes.c_float),  # 5
        ('layer0_shadow_g', ctypes.c_float),  # 6
        ('layer0_shadow_b', ctypes.c_float),  # 7
        ('layer0_light_r', ctypes.c_float),  # 8
        ('layer0_light_g', ctypes.c_float),  # 9
        ('layer0_light_b', ctypes.c_float),  # 10
        ('layer0_shadow_point', ctypes.c_float),  # 11
        ('layer0_highlight_point', ctypes.c_float),  # 12
        ('layer1_shadow_r', ctypes.c_float),  # 13
        ('layer1_shadow_g', ctypes.c_float),  # 14
        ('layer1_shadow_b', ctypes.c_float),  # 15
        ('layer1_light_r', ctypes.c_float),  # 16
        ('layer1_light_g', ctypes.c_float),  # 17
        ('layer1_light_b', ctypes.c_float),  # 18
        ('layer1_shadow_point', ctypes.c_float),  # 19
        ('layer1_highlight_point', ctypes.c_float),  # 20
        ('_layer_future_0', ctypes.c_float * 36),  # Test with 36
        ('n_layers', ctypes.c_float),  # +1 = 57
        ('_layer_config_1', ctypes.c_float),  # +1 = 58
        ('_layer_config_2', ctypes.c_float),  # +1 = 59
        ('_layer_config_3', ctypes.c_float),  # +1 = 60
        ('peak_sensitivity_nm', ctypes.c_float),  # +1 = 61
        ('spectral_fwhm_nm', ctypes.c_float),  # +1 = 62
        ('_spectral_2', ctypes.c_float),  # +1 = 63
        ('_spectral_3', ctypes.c_float),  # +1 = 64
    ]

size = ctypes.sizeof(FilmRecord)
floats = size // 4
print(f"FilmRecord size with 36: {size} bytes ({floats} floats)")
