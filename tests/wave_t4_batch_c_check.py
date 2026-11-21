"""Verify the native cross-arena FFT entry against separate authored T4 calls."""

import ctypes as ct
import math
import sys


def main(dll_path: str) -> None:
    dll = ct.CDLL(dll_path)
    pointer = ct.POINTER(ct.c_float)
    dll.wave_t4_batch_plan_create.argtypes = [ct.c_int, ct.c_int]
    dll.wave_t4_batch_plan_create.restype = ct.c_void_p
    dll.wave_t4_batch_plan_destroy.argtypes = [ct.c_void_p]
    dll.wave_t4_batch_angular_step.argtypes = [
        ct.c_void_p, ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_int,
        ct.POINTER(ct.c_double), ct.POINTER(pointer), ct.POINTER(pointer),
    ]
    dll.wave_t4_batch_angular_step.restype = ct.c_int
    plan = dll.wave_t4_batch_plan_create(16, 16)
    if not plan:
        raise AssertionError("T4 plan creation failed")
    try:
        source_re = [
            [(math.sin((i + 1) * (arena + 2)) + 1) * 0.25 for i in range(256)]
            for arena in range(2)
        ]
        source_im = [
            [math.cos((i + 3) * (arena + 1)) * 0.2 for i in range(256)]
            for arena in range(2)
        ]
        re = [(ct.c_float * 256)(*values) for values in source_re]
        im = [(ct.c_float * 256)(*values) for values in source_im]
        re_ptrs = (pointer * 2)(*(ct.cast(values, pointer) for values in re))
        im_ptrs = (pointer * 2)(*(ct.cast(values, pointer) for values in im))
        wavelengths = (ct.c_double * 2)(450e-9, 850e-9)
        if dll.wave_t4_batch_angular_step(
            plan, 2, 1, 0.5e-6, 0.25e-6, 1,
            wavelengths, re_ptrs, im_ptrs,
        ) != 0:
            raise AssertionError("batched T4 step failed")
        for arena in range(2):
            separate_re = (ct.c_float * 256)(*source_re[arena])
            separate_im = (ct.c_float * 256)(*source_im[arena])
            single_re = (pointer * 1)(ct.cast(separate_re, pointer))
            single_im = (pointer * 1)(ct.cast(separate_im, pointer))
            one_wavelength = (ct.c_double * 1)(wavelengths[arena])
            if dll.wave_t4_batch_angular_step(
                plan, 1, 1, 0.5e-6, 0.25e-6, 1,
                one_wavelength, single_re, single_im,
            ) != 0:
                raise AssertionError("single-arena T4 step failed")
            for i in range(256):
                if abs(re[arena][i] - separate_re[i]) > 2e-5 or abs(
                    im[arena][i] - separate_im[i]
                ) > 2e-5:
                    raise AssertionError(f"arena {arena} cell {i} differs")
        print("two-arena T4 FFT batch matches separate authored calls")
    finally:
        dll.wave_t4_batch_plan_destroy(plan)


if __name__ == "__main__":
    main(sys.argv[1])
