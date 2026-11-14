"""Native procedural windows of Turing's existing repository UnitIndex.

The caller supplies an authorized RepoWorldScan/UnitIndex, not executable
metadata in a map document. Only build_unit_world materializes the requested
window; there is no second layout, scanner, or REST vocabulary here.
"""
from native_island_loader import NativeIsland, NativeIslandLoader


class RepoWorldIslands:
    def __init__(self, index, workspace_factory, *, scale=1.0):
        from src.compiler.repo_world_unit_world import island_of
        self.index = index
        self.workspace_factory = workspace_factory
        self.scale = scale
        self.regions = {island_of(region): region for region in index.regions}

    def address(self, event):
        from src.compiler.repo_world_units import Treadmill, split_address
        region = self.regions[event['island']]
        door = self.index.door(region)
        treadmill = event.get('treadmill')
        if treadmill is not None:
            if not isinstance(door, Treadmill) or treadmill['length'] != door.length or treadmill['window'] != door.window:
                raise ValueError('treadmill declaration differs from the supplied repository index')
            return door.address(treadmill['offset'])
        if isinstance(door, Treadmill):
            # Original build_unit_world href callback supplies the exact parent
            # containing-window address. Interpret as index address, NOT URL.
            address = event.get('href')
            if not isinstance(address, str) or split_address(address)[0] != region:
                raise ValueError('windowed parent requires its original unit address')
            self.index.unit(address)  # validates range and original semantics
            return address
        return region

    def build(self, address):
        from src.compiler.repo_world_unit_world import build_unit_world
        from nodus_world_geometry import from_world
        # href is an existing producer extension point. No URL is opened;
        # the explicit native factory resolves these exact UnitIndex addresses.
        world = build_unit_world(self.index, address, href=lambda address: address)
        geometry = from_world(world).scaled(self.scale)
        return NativeIsland(geometry, self.workspace_factory(geometry))

    def __call__(self, event):
        return self.build(self.address(event))

    def loader(self, initial_address):
        loader = NativeIslandLoader(self.build(initial_address), initial_cache_key=initial_address)
        for identity in self.regions:
            loader.register(identity, self, cache_key=self.address)
        return loader
