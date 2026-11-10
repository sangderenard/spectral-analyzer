import numpy as np
import pytest
from camera_designer.studio_sensor_evidence import leaf_nodes


def test_native_leaf_layout_exact_pixel_uv_and_offsets():
    nodes=leaf_nodes(3,2,3)
    assert nodes.shape==(6,16) and nodes.nbytes==6*64
    np.testing.assert_allclose(nodes[:,:4].view(np.float32),[[0,0,1/3,.5],[1/3,0,2/3,.5],[2/3,0,1,.5],[0,.5,1/3,1],[1/3,.5,2/3,1],[2/3,.5,1,1]])
    np.testing.assert_array_equal(nodes[:,4:6],0xffffffff)
    np.testing.assert_array_equal(nodes[:,8],np.arange(6)*3)
    assert not nodes[:,10].any()


@pytest.mark.parametrize('dimensions',[(0,1,3),(513,1,3),(1,1,1),(True,1,3),(1.,1,3)])
def test_unsupported_dimensions_refuse(dimensions):
    with pytest.raises(ValueError):leaf_nodes(*dimensions)
