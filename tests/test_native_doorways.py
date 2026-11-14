import unittest
from native_doorways import Aperture, DoorRuntime


def opening(identity, role, pair="lot/door:room/enter", **extra):
    return dict(identity=identity, kind="portal", transition="door", port_role=role, pair=pair, **extra)


class NativeDoorTests(unittest.TestCase):
    def runtime(self):
        records = [opening("in", "in", coincident_with="entry"),
                   opening("out", "out", coincident_with="gate", facing="inward")]
        geometry = {"entry": Aperture((0, 0, 0), (0, 0, 1), (1, 0, 0)),
                    "gate": Aperture((100, 0, 0), (-1, 0, 0), (0, 0, 1))}
        return DoorRuntime(records, geometry)

    def test_remote_local_door_rotates_and_preserves_offset_speed(self):
        r = self.runtime()
        c = r.step("walker", (.2, .1, 1), (.2, .1, -.1), (0, 0, -2), (0, 0, -1), now_ms=0)
        self.assertEqual(c.target, "out")
        self.assertAlmostEqual(c.position[0], 100.09)
        self.assertAlmostEqual(c.position[1], .1)
        self.assertAlmostEqual(c.position[2], -.2)
        self.assertAlmostEqual(c.velocity[0], 2)
        self.assertAlmostEqual(c.facing[0], 1)
        self.assertIsNone(r.step("walker", (.2, .1, 1), (.2, .1, -.1), now_ms=239))
        self.assertIsNotNone(r.step("walker", (.2, .1, 1), (.2, .1, -.1), now_ms=240))

    def test_aperture_and_direction_and_blocked_touch(self):
        r = self.runtime()
        self.assertIsNone(r.step("w", (2, 0, 1), (2, 0, -1), now_ms=0))
        self.assertIsNone(r.step("w", (0, 0, -1), (0, 0, 1), now_ms=0))
        self.assertIsNotNone(r.step("w", (0, 0, .3), (0, 0, .1), now_ms=0))

    def test_dial_selects_island_without_loading_it(self):
        records = [opening(f"in{i}", "in", f"pair{i}", coincident_with="entry", island=f"world{i}",
                           href=f"room{i}.html", dial="lot/dial", dial_position=i, dial_positions=2) for i in range(2)]
        r = DoorRuntime(records, {"entry": Aperture((0, 0, 0), (0, 0, 1), (1, 0, 0))})
        r.select_dial("lot/dial", 1)
        c = r.step("w", (0, 0, 1), (0, 0, -1), now_ms=0)
        self.assertEqual(c.event, dict(island="world1", pair="pair1", source="in1", href="room1.html"))
        self.assertEqual(c.cooldown_until_ms, 1920)
        self.assertIsNone(r.step("car", (0, 0, 1), (0, 0, -1), now_ms=0, is_walker=False))
        with self.assertRaises(ValueError):
            r.select_dial("lot/dial", 2)

    def test_treadmill_preserves_island_and_allows_partial_last_window(self):
        record = opening("in", "in", island="world:folder", coincident_with="entry", treadmill=dict(length=7, window=3, offset=0))
        r = DoorRuntime([record], {"entry": Aperture((0, 0, 0), (0, 0, 1), (1, 0, 0))})
        self.assertEqual(r.scroll_treadmill("in", 6)["offset"], 6)
        c = r.step("w", (0, 0, 1), (0, 0, -1), now_ms=0)
        self.assertEqual(c.event["island"], "world:folder")
        self.assertEqual(c.event["treadmill"], dict(length=7, window=3, offset=6))
        with self.assertRaises(ValueError):
            r.scroll_treadmill("in", 7)

    def test_arrival_and_ambiguous_pair(self):
        r = self.runtime()
        c = r.arrive("w", "lot/door:room/enter", now_ms=100)
        self.assertEqual(c.velocity, (0, 0, 0))
        self.assertEqual(c.facing, (1, 0, 0))
        self.assertAlmostEqual(c.position[0], 100.09)
        r.openings = [*r.openings, opening("other", "out")]
        with self.assertRaises(ValueError):
            r.step("new", (0, 0, 1), (0, 0, -1), now_ms=0)

    def test_invalid_frame_and_declaration(self):
        with self.assertRaises(ValueError):
            Aperture((0, 0, 0), (0, 0, 1), (0, 0, 1))
        with self.assertRaises(ValueError):
            DoorRuntime([opening("in", "in", treadmill=dict(length=2, window=3, offset=0))], {})


if __name__ == "__main__":
    unittest.main()
