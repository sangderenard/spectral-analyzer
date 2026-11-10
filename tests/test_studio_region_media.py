import pytest
from camera_designer.studio_region_media import Region,RegionState,Crossing,ResolvedMedium


def enter(i):return Crossing(i,True)
def leave(i):return Crossing(i,False)


def test_non_lifo_same_material_rods_form_union():
    s=RegionState({1:Region('glass_rod',7),2:Region('glass_rod',7)},0)
    for e,expected in [(enter(1),7),(enter(2),7),(leave(1),7),(leave(2),0)]:
        assert s.cross([e],transmitted=True).base_material==expected


def test_dye_overlap_preserves_all_concentrations_in_host():
    s=RegionState({1:Region('injection_dye',7),2:Region('injection_dye',8),3:Region('injection_dye',7)},0)
    s.cross([enter(1),enter(2),enter(3)],transmitted=True)
    assert s.medium==ResolvedMedium(0,(7,7,8))
    s.cross([leave(1)],transmitted=True)
    assert s.medium==ResolvedMedium(0,(7,8))


def test_bubbles_cut_medium_without_forgetting_hidden_dye_membership():
    s=RegionState({1:Region('injection_dye',7),2:Region('air_bubble',-1),3:Region('air_bubble',-1)},0,[1])
    s.cross([enter(2),enter(3)],transmitted=True)
    assert s.medium==ResolvedMedium(-1)
    s.cross([leave(2),leave(1)],transmitted=True)
    assert s.medium==ResolvedMedium(-1)
    s.cross([leave(3)],transmitted=True)
    assert s.medium==ResolvedMedium(0)


def test_reflection_and_tir_do_not_change_membership():
    s=RegionState({1:Region('glass_rod',7)},0)
    assert s.cross([enter(1)],transmitted=False)==ResolvedMedium(0)
    s.cross([enter(1)],transmitted=True)
    assert s.cross([leave(1)],transmitted=False)==ResolvedMedium(7)


def test_coincident_replacement_group_is_atomic():
    s=RegionState({1:Region('glass_rod',7),2:Region('glass_rod',8)},0,[1])
    s.cross([leave(1),enter(2)],transmitted=True)
    assert s.medium==ResolvedMedium(8)


def test_undefined_material_mixtures_are_rejected_not_prioritized():
    s=RegionState({1:Region('glass_rod',7),2:Region('glass_rod',8),3:Region('injection_dye',9)},0,[1])
    for i in (2,3):
        with pytest.raises(ValueError,match='overlap'):s.cross([enter(i)],transmitted=True)
        assert s.active==frozenset([1])


@pytest.mark.parametrize('crossings',[[leave(1)],[enter(2)],[enter(1),enter(1)]])
def test_invalid_crossing_cannot_corrupt_state(crossings):
    s=RegionState({1:Region('glass_rod',7)},0)
    with pytest.raises(ValueError):s.cross(crossings,transmitted=True)
    assert not s.active
