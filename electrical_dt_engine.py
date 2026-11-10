"""Managed-dt participant for the complex tensor electrical network.

The dt system owns time, acceptance, and rollback.  This engine owns one
electrical graph solve and its stored electrical state.  It deliberately does
not run an internal clock or scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from electrical_tensor_network import (
    GROUND,
    ComplexCircuitReading,
    ElectricalDeviceRegistry,
    _phasors,
)
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine
from src.common.dt_system.error_channels import empty_channels
from src.common.dt_system.time_contracts import HOLD
from src.common.tensors import AbstractTensor


def _clone_reading(reading: ComplexCircuitReading | None):
    if reading is None:
        return None
    return ComplexCircuitReading(
        {key: value.detach().clone() for key, value in reading.voltage_v.items()},
        {key: value.detach().clone()
         for key, value in reading.branch_current_a.items()},
        {key: value.detach().clone()
         for key, value in reading.source_current_a.items()},
        ({key: value.detach().clone()
          for key, value in reading.device_current_a.items()}
         if reading.device_current_a is not None else None),
        ({key: value.detach().clone()
          for key, value in reading.circuit_terminal_current_a.items()}
         if reading.circuit_terminal_current_a is not None else None),
    )


@dataclass(frozen=True)
class ElectricalSafetyEvent:
    branch_identity: str
    kind: str
    condition: str
    actual: float | str | bool
    limit: float | str | bool
    thermal_target: str | None = None


@dataclass(frozen=True)
class ElectricalEngineSnapshot:
    world_time: float
    observer_time: float
    device_state: tuple[tuple[str, object], ...]
    current_overrides: tuple[tuple[str, torch.Tensor], ...]
    reading: ComplexCircuitReading | None
    branch_loss_w: tuple[tuple[str, float], ...]
    joule_heat_w: tuple[tuple[str, float], ...]
    safety_events: tuple[ElectricalSafetyEvent, ...]


class ComplexElectricalEngine(DtCompatibleEngine):
    """One top-level dt engine around one simultaneous complex nodal solve."""

    @property
    def solve_backend(self) -> str:
        return self.registry.solve_backend

    def __init__(self, registry: ElectricalDeviceRegistry,
                 *, label: str = "electrical-system") -> None:
        super().__init__()
        self.registry = registry
        self.label = str(label)
        self._registration = set()
        self._current_overrides: dict[str, torch.Tensor] = {}
        self.reading: ComplexCircuitReading | None = None
        self.thermal_assembly = None
        self.branch_loss_w: dict[str, float] = {}
        self.joule_heat_w: dict[str, float] = {}
        self.safety_events: tuple[ElectricalSafetyEvent, ...] = ()

    def bind_thermal_assembly(self, assembly) -> None:
        """Use an existing thermal assembly for resistance and heat transfer."""
        self.thermal_assembly = assembly

    def thermal_source_w(self, _index=None, assembly=None) -> dict[str, float]:
        target = assembly or self.thermal_assembly
        if target is None:
            return dict(self.joule_heat_w)
        return {
            identity: watts for identity, watts in self.joule_heat_w.items()
            if identity in target.states
        }

    def set_current(self, identity: str, current_a) -> None:
        known = {row.identity for row in self.registry.injections}
        if identity not in known:
            raise KeyError(f"unregistered electrical current port: {identity}")
        self._current_overrides[identity] = _phasors(
            current_a, self.registry.lanes).detach().clone()

    def snapshot(self) -> ElectricalEngineSnapshot:
        return ElectricalEngineSnapshot(
            world_time=float(self.world_time),
            observer_time=float(self.observer_time),
            device_state=self.registry.snapshot(),
            current_overrides=tuple(
                (identity, value.detach().clone())
                for identity, value in sorted(self._current_overrides.items())
            ),
            reading=_clone_reading(self.reading),
            branch_loss_w=tuple(sorted(self.branch_loss_w.items())),
            joule_heat_w=tuple(sorted(self.joule_heat_w.items())),
            safety_events=self.safety_events,
        )

    def restore(self, snapshot: ElectricalEngineSnapshot) -> None:
        self.world_time = float(snapshot.world_time)
        self.observer_time = float(snapshot.observer_time)
        self.registry.restore(snapshot.device_state)
        self._current_overrides = {
            identity: value.detach().clone()
            for identity, value in snapshot.current_overrides
        }
        self.reading = _clone_reading(snapshot.reading)
        self.branch_loss_w = dict(snapshot.branch_loss_w)
        self.joule_heat_w = dict(snapshot.joule_heat_w)
        self.safety_events = snapshot.safety_events

    def _safety_events(self) -> tuple[ElectricalSafetyEvent, ...]:
        """Evaluate installation limits against RMS branch current and heat state.

        Frequency lanes are RMS phasors, so their root-sum-square is the RMS
        current carried by the one physical conductor or termination.  The
        continuous-load threshold is reported as a sizing condition; breaker
        trip-time physics remains the breaker's constitutive responsibility.
        """
        if self.reading is None:
            return ()
        events = []
        for identity, requirements in sorted(
                self.registry.safety_by_branch.items()):
            current = self.reading.branch_current_a.get(identity)
            if current is None:
                continue
            current_rms_a = float(torch.sqrt(torch.sum(torch.abs(current) ** 2)))
            target = requirements.get("thermal_target")
            common = {
                "branch_identity": identity,
                "kind": str(requirements.get("kind", "electrical-branch")),
                "thermal_target": None if target is None else str(target),
            }
            continuous_limit = float(requirements.get(
                "continuous_load_limit_a", 0.0))
            if continuous_limit > 0.0 and current_rms_a > continuous_limit:
                events.append(ElectricalSafetyEvent(
                    **common,
                    condition="continuous-load-threshold-exceeded",
                    actual=current_rms_a,
                    limit=continuous_limit,
                ))
            ampacity = float(requirements.get("ampacity_a", 0.0))
            if ampacity > 0.0 and current_rms_a > ampacity:
                events.append(ElectricalSafetyEvent(
                    **common,
                    condition="conductor-ampacity-exceeded",
                    actual=current_rms_a,
                    limit=ampacity,
                ))
            if requirements.get("kind") != "junction":
                continue
            if not bool(requirements.get("torque_verified", False)):
                events.append(ElectricalSafetyEvent(
                    **common,
                    condition="termination-torque-unverified",
                    actual=False,
                    limit=True,
                ))
            listing = str(requirements.get("connector_listing", ""))
            if not listing:
                events.append(ElectricalSafetyEvent(
                    **common,
                    condition="termination-connector-unlisted",
                    actual="",
                    limit="listed connector",
                ))
            rating_c = float(requirements.get("temperature_rating_c", 0.0))
            temperature_k = self.registry.thermal_temperature_k.get(str(target))
            if rating_c > 0.0 and temperature_k is not None:
                temperature_c = float(temperature_k) - 273.15
                if temperature_c > rating_c:
                    events.append(ElectricalSafetyEvent(
                        **common,
                        condition="termination-temperature-exceeded",
                        actual=temperature_c,
                        limit=rating_c,
                    ))
        return tuple(events)

    def _kcl_residual(self, reading: ComplexCircuitReading) -> float:
        residual = {
            node: torch.zeros(self.registry.lanes, dtype=torch.complex128)
            for node in self.registry.circuit().nodes
        }
        for source in self.registry.sources:
            residual[source.node] += reading.source_current_a[source.identity]
        for row in self.registry.injections:
            if "::offset:" in row.identity:
                continue
            residual[row.node] += self._current_overrides.get(
                row.identity, _phasors(row.current_a, self.registry.lanes))
        for branch in self.registry.branches:
            if branch.identity in self.registry.port_laws:
                continue
            current = reading.branch_current_a[branch.identity]
            if branch.a in residual:
                residual[branch.a] -= current
            if branch.b in residual:
                residual[branch.b] += current
        for identity, law in self.registry.port_laws.items():
            current = reading.device_current_a[identity]
            if law.a in residual:
                residual[law.a] -= current
            if law.b in residual:
                residual[law.b] += current
        if reading.circuit_terminal_current_a:
            for identity, law in self.registry.circuit_laws.items():
                terminal_currents = reading.circuit_terminal_current_a[identity]
                for index, terminal in enumerate(law.terminals):
                    if terminal in residual:
                        # Multiport current is positive into the device.
                        residual[terminal] -= terminal_currents[:, index]
        return max(
            (float(torch.abs(value).max()) for value in residual.values()),
            default=0.0,
        )

    def step(self, dt: float, state=None, state_table=None):
        if self.thermal_assembly is not None:
            self.registry.set_thermal_temperatures({
                identity: thermal_state.mean_temperature_k
                for identity, thermal_state in self.thermal_assembly.states.items()
            })
        self.reading = self.registry.solve(
            injection_current_a=self._current_overrides)
        self.branch_loss_w = {}
        self.joule_heat_w = {}
        zero = torch.zeros(self.registry.lanes, dtype=torch.complex128)
        for branch in self.registry.branches:
            target = self.registry.thermal_target_by_branch.get(branch.identity)
            va = zero if branch.a == GROUND \
                else self.reading.voltage_v[branch.a]
            vb = zero if branch.b == GROUND \
                else self.reading.voltage_v[branch.b]
            current = self.reading.branch_current_a[branch.identity]
            watts = max(0.0, float(torch.real(
                (va - vb) * torch.conj(current)).sum()))
            self.branch_loss_w[branch.identity] = watts
            if target is not None:
                fraction = float(self.registry.thermal_fraction_by_branch.get(
                    branch.identity, 1.0))
                heat_w = watts * fraction
                self.joule_heat_w[target] = (
                    self.joule_heat_w.get(target, 0.0) + heat_w)
        self.safety_events = self._safety_events()
        self.registry.advance(self.reading, float(dt))
        currents = tuple(self.reading.branch_current_a.values()) + tuple(
            self.reading.source_current_a.values())
        if self.reading.device_current_a:
            currents += tuple(self.reading.device_current_a.values())
        if self.reading.circuit_terminal_current_a:
            currents += tuple(self.reading.circuit_terminal_current_a.values())
        max_current = max(
            (float(torch.abs(value).max()) for value in currents), default=0.0)
        channels = empty_channels()
        metrics = Metrics(
            max_vel=0.0,
            max_flux=max_current,
            div_inf=self._kcl_residual(self.reading),
            mass_err=0.0,
            pub_tau=AbstractTensor.tensor([0.0]),
            pub_tau_present=AbstractTensor.tensor([0.0]),
            pub_contract=AbstractTensor.tensor([HOLD]),
            pub_dt_limit=AbstractTensor.tensor([0.0]),
            pub_dt_limit_present=AbstractTensor.tensor([0.0]),
            pub_values=channels.copy(),
            pub_present=AbstractTensor.zeros_like(channels),
            pub_limits=AbstractTensor.zeros_like(channels),
            pub_limits_present=AbstractTensor.zeros_like(channels),
        )
        return True, metrics, self.get_state(state)

    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out["frequencies_hz"] = self.registry.frequencies_hz
        out["device_state"] = dict(self.registry.snapshot())
        if self.reading is not None:
            out["voltage_v"] = self.reading.voltage_v
            out["branch_current_a"] = self.reading.branch_current_a
            out["source_current_a"] = self.reading.source_current_a
            out["device_current_a"] = self.reading.device_current_a
            out["circuit_terminal_current_a"] = (
                self.reading.circuit_terminal_current_a)
            out["branch_loss_w"] = dict(self.branch_loss_w)
            out["joule_heat_w"] = dict(self.joule_heat_w)
            out["electrical_safety_events"] = self.safety_events
        return out

    def publish_to_state(self, table) -> None:
        if self.reading is None:
            return
        table.set("engine", self.label, "frequencies_hz",
                  self.registry.frequencies_hz)
        table.set("engine", self.label, "voltage_phasor_v",
                  self.reading.voltage_v)
        table.set("engine", self.label, "branch_current_phasor_a",
                  self.reading.branch_current_a)
        table.set("engine", self.label, "source_current_phasor_a",
                  self.reading.source_current_a)
        table.set("engine", self.label, "device_current_phasor_a",
                  self.reading.device_current_a)
        table.set("engine", self.label, "circuit_terminal_current_phasor_a",
                  self.reading.circuit_terminal_current_a)
        table.set("engine", self.label, "device_state",
                  dict(self.registry.snapshot()))
        table.set("engine", self.label, "branch_loss_w",
                  dict(self.branch_loss_w))
        table.set("engine", self.label, "joule_heat_w",
                  dict(self.joule_heat_w))
        table.set("engine", self.label, "electrical_safety_events",
                  self.safety_events)


__all__ = [
    "ComplexElectricalEngine", "ElectricalEngineSnapshot",
    "ElectricalSafetyEvent",
]
