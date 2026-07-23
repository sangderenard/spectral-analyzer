import numpy as np

from camera_software.qr_optical_validator import (
    QR_MATRIX_ROWS, QR_PAYLOAD, qr_target_spec, triangulate_qr_target,
)


def test_optical_qr_is_fixed_valid_shape_with_physical_quiet_zone():
    assert QR_PAYLOAD == "SPECTRAL-BENCH-V1"
    assert len(QR_MATRIX_ROWS) == 25
    assert {len(row) for row in QR_MATRIX_ROWS} == {25}
    groups = triangulate_qr_target(qr_target_spec())
    assert [group[0] for group in groups] == [False, True]
    assert groups[0][1].shape == (2, 9)
    assert groups[1][1].shape[0] == 2 * sum(row.count("1") for row in QR_MATRIX_ROWS)
    assert all(np.allclose(group[2], (0.0, 0.0, -1.0)) for group in groups)


def test_optical_qr_black_ink_is_in_front_of_white_backing():
    white, black = triangulate_qr_target(qr_target_spec(z=0.30))
    assert np.allclose(white[1][:, 2::3], 0.30)
    assert np.allclose(black[1][:, 2::3], 0.30 - 1.0e-6)

