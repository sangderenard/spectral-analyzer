import numpy as np

from camera_software.sensor_mipmap import SparseSensorMipmap


def center(bounds):
    return ((bounds.u0 + bounds.u1) * 0.5, (bounds.v0 + bounds.v1) * 0.5)


def test_explicitly_refined_nonterminal_node_materializes_exactly_nine_children():
    tree = SparseSensorMipmap(3, maximum_depth=2)
    assert tree.complete_work(tree.root_id) == ()
    children = tree.complete_work(tree.root_id, subdivide=True)
    assert len(children) == 9
    assert tree.complete_work(tree.root_id, subdivide=True) == children
    assert len(tree.nodes) == 10


def test_children_exactly_partition_parent_area_without_overlap():
    tree = SparseSensorMipmap(1, maximum_depth=1)
    children = tree.complete_work(tree.root_id, subdivide=True)
    bounds = [tree.nodes[node_id].bounds for node_id in children]
    assert np.isclose(sum(item.area for item in bounds), 1.0)
    assert len({(item.u0, item.v0, item.u1, item.v1) for item in bounds}) == 9


def test_direct_coarse_evidence_is_retained_when_children_roll_up():
    tree = SparseSensorMipmap(2, maximum_depth=1)
    tree.add_sample(tree.root_id, (0.5, 0.5), np.asarray([90.0, 45.0]))
    children = tree.complete_work(tree.root_id, subdivide=True)
    for value, node_id in enumerate(children):
        node = tree.nodes[node_id]
        tree.add_sample(node_id, center(node.bounds), np.asarray([value, 2.0 * value]))

    root = tree.nodes[tree.root_id]
    assert np.allclose(root.direct.mean, [90.0, 45.0])
    assert root.rolled is not None
    assert np.allclose(root.rolled.mean, [4.0, 8.0])
    assert tree.resolved_estimate(tree.root_id).valid


def test_fine_sample_keeps_global_uv_and_explicit_node_lineage():
    tree = SparseSensorMipmap(1, maximum_depth=1)
    child_id = tree.complete_work(tree.root_id, subdivide=True)[8]
    uv = center(tree.nodes[child_id].bounds)
    tree.add_sample(child_id, uv, np.asarray([3.5]))
    sample = tree.samples[-1]
    assert sample.node_id == child_id
    assert sample.global_uv == uv
    assert np.allclose(sample.spectrum, [3.5])


def test_rollup_waits_for_all_nine_strata_instead_of_biasing_toward_top_k():
    tree = SparseSensorMipmap(1, maximum_depth=1)
    children = tree.complete_work(tree.root_id, subdivide=True)
    for node_id in children[:8]:
        node = tree.nodes[node_id]
        tree.add_sample(node_id, center(node.bounds), np.asarray([100.0]))
    assert tree.nodes[tree.root_id].rolled is None

    last = tree.nodes[children[8]]
    tree.add_sample(last.node_id, center(last.bounds), np.asarray([1.0]))
    assert np.allclose(tree.nodes[tree.root_id].rolled.mean, [(8.0 * 100.0 + 1.0) / 9.0])


def test_finest_nodes_keep_accepting_epochs_without_creating_children():
    tree = SparseSensorMipmap(1, maximum_depth=0)
    assert tree.complete_work(tree.root_id) == ()
    assert tree.complete_work(tree.root_id) == ()
    assert tree.nodes[tree.root_id].completed_epochs == 2


def test_uv_request_descends_only_requested_lineage_and_keeps_continuous_address():
    tree = SparseSensorMipmap(1, maximum_depth=4)
    leaf = tree.refine_uv(0.731, 0.214, target_level=3)
    assert leaf.level == 3
    assert leaf.bounds.contains(0.731, 0.214)
    assert len(tree.nodes) == 1 + 3 * 9
    assert tree.leaf_at_uv(0.1, 0.9).level == 1
