"""Complex electrical networks with Python and LLVM nodal backends.

Electrical state stays as complex phasor tensors.  A circuit node carries one
voltage phasor per declared lane; branches carry complex admittance and yield
complex current.  The production path performs the simultaneous nodal solve
through the compiled LLVM kernel; ``GraphSolver`` remains a Python fallback.

The graph equation is the Jacobi form of nodal analysis.  For ``Y V = I``::

    V_i = I_i / Y_ii - sum(Y_ij / Y_ii * V_j, j != i)

Each right-hand term is one ordinary complex ``TensorEdge``.  This is only a
change of representation of the nodal equations; it does not iterate or
schedule a second circuit runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import torch

_CDTYPE = torch.complex128
GROUND = "__electrical_ground__"


def _phasors(value, lanes: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=_CDTYPE)
    if tensor.ndim == 0:
        tensor = tensor.repeat(lanes)
    tensor = tensor.reshape(-1)
    if tensor.numel() != lanes:
        raise ValueError(f"expected {lanes} phasor lanes, got {tensor.numel()}")
    return tensor


def _multiport_matrix(value, lanes: int, terminals: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=_CDTYPE)
    if tensor.shape == (terminals, terminals):
        tensor = tensor.unsqueeze(0).repeat(lanes, 1, 1)
    if tensor.shape != (lanes, terminals, terminals):
        raise ValueError(
            f"expected [{lanes}, {terminals}, {terminals}] admittance, "
            f"got {list(tensor.shape)}")
    return tensor


def _multiport_vector(value, lanes: int, terminals: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=_CDTYPE)
    if tensor.shape == (terminals,):
        tensor = tensor.unsqueeze(0).repeat(lanes, 1)
    if tensor.shape != (lanes, terminals):
        raise ValueError(
            f"expected [{lanes}, {terminals}] terminal vector, "
            f"got {list(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class ImpedanceBranch:
    identity: str
    a: str
    b: str
    impedance_ohm: complex | Sequence[complex] | torch.Tensor


@dataclass(frozen=True)
class TheveninSource:
    """A voltage-producing two-terminal branch.

    ``node`` is the source terminal and ``return_node`` is its physical
    return.  A generator may therefore float; bonding its neutral or return
    to the installation reference remains a separate graph connection.
    """

    identity: str
    node: str
    voltage_v: complex | Sequence[complex] | torch.Tensor
    impedance_ohm: complex | Sequence[complex] | torch.Tensor
    return_node: str = GROUND


@dataclass(frozen=True)
class CurrentInjection:
    identity: str
    node: str
    current_a: complex | Sequence[complex] | torch.Tensor


@dataclass(frozen=True)
class PortLinearization:
    """Local Norton tangent for one empirical two-terminal law.

    Current is positive from ``a`` to ``b`` and obeys
    ``I = admittance * (V_a - V_b) + current_offset``.
    """

    admittance_s: complex | Sequence[complex] | torch.Tensor
    current_offset_a: complex | Sequence[complex] | torch.Tensor = 0.0j


@dataclass(frozen=True)
class MultiportLinearization:
    """Terminal form ``I = Y V + I_offset`` for a baked circuit.

    Currents are positive into the circuit.  Terminal order is owned by the
    corresponding :class:`EmpiricalCircuitLaw`; the leading tensor dimension
    is the spectral lane.
    """

    admittance_s: torch.Tensor
    current_offset_a: torch.Tensor


@dataclass(frozen=True)
class AdmittanceStamp:
    identity: str
    terminals: tuple[str, ...]
    admittance_s: torch.Tensor
    current_offset_a: torch.Tensor


@dataclass
class EmpiricalPortLaw:
    """One electrical ABI for wires, sources, storage, and machines."""

    identity: str
    a: str
    b: str
    linearize: Callable[[torch.Tensor, torch.Tensor], PortLinearization]
    advance: Callable[[torch.Tensor, torch.Tensor, float], None] | None = None
    snapshot_state: Callable[[], Any] | None = None
    restore_state: Callable[[Any], None] | None = None

    def tangent(self, voltage_ab: torch.Tensor,
                frequencies_hz: torch.Tensor) -> PortLinearization:
        return self.linearize(voltage_ab, frequencies_hz)


@dataclass
class EmpiricalCircuitLaw:
    """A detailed circuit baked to a stateful multi-terminal boundary."""

    identity: str
    terminals: tuple[str, ...]
    linearize: Callable[[torch.Tensor, torch.Tensor], MultiportLinearization]
    advance: Callable[[torch.Tensor, torch.Tensor, float], None] | None = None
    snapshot_state: Callable[[], Any] | None = None
    restore_state: Callable[[Any], None] | None = None

    def tangent(self, terminal_voltage: torch.Tensor,
                frequencies_hz: torch.Tensor) -> MultiportLinearization:
        return self.linearize(terminal_voltage, frequencies_hz)


def impedance_port_law(
    identity: str,
    a: str,
    b: str,
    impedance_ohm: complex | Sequence[complex] | torch.Tensor,
) -> EmpiricalPortLaw:
    def linearize(_voltage_ab, frequencies_hz):
        z = _phasors(impedance_ohm, int(frequencies_hz.numel()))
        if torch.any(torch.abs(z) == 0.0):
            raise ValueError(f"{identity}: zero impedance")
        return PortLinearization(1.0 / z)
    return EmpiricalPortLaw(identity, a, b, linearize)


def transmission_line_circuit_law(
    identity: str,
    terminals: Sequence[str],
    *,
    length_m: float,
    resistance_ohm_per_m: float,
    inductance_h_per_m: float,
    conductance_s_per_m: float,
    capacitance_f_per_m: float,
) -> EmpiricalCircuitLaw:
    """Exact uniform two-conductor RLGC line as one four-terminal law.

    Terminal order is ``(send, send_return, receive, receive_return)``.
    The law retains propagation phase, attenuation, reflections, and line
    resonances on every registered frequency lane.  It therefore shares the
    same multi-terminal ABI as a baked circuit instead of introducing a
    separate transmission scheduler.
    """
    terminal_names = tuple(str(name) for name in terminals)
    if len(terminal_names) != 4 or len(set(terminal_names)) != 4:
        raise ValueError(
            f"{identity}: a transmission line needs four distinct terminals")
    if length_m <= 0.0:
        raise ValueError(f"{identity}: transmission-line length must be positive")
    if min(resistance_ohm_per_m, inductance_h_per_m,
           conductance_s_per_m, capacitance_f_per_m) < 0.0:
        raise ValueError(f"{identity}: RLGC values must be nonnegative")

    def linearize(_terminal_voltage, frequencies_hz):
        omega = 2.0 * math.pi * frequencies_hz.to(torch.float64)
        series = torch.as_tensor(
            resistance_ohm_per_m + 1j * omega * inductance_h_per_m,
            dtype=_CDTYPE)
        shunt = torch.as_tensor(
            conductance_s_per_m + 1j * omega * capacitance_f_per_m,
            dtype=_CDTYPE)
        propagation_length = torch.sqrt(series * shunt) * float(length_m)
        sinhc = torch.ones_like(propagation_length)
        moving = torch.abs(propagation_length) > 1.0e-12
        sinhc[moving] = (
            torch.sinh(propagation_length[moving])
            / propagation_length[moving]
        )
        series_transfer = series * float(length_m) * sinhc
        if torch.any(torch.abs(series_transfer) == 0.0):
            raise ValueError(
                f"{identity}: every spectral lane needs nonzero series impedance")
        diagonal = torch.cosh(propagation_length) / series_transfer
        mutual = -1.0 / series_transfer
        port_y = torch.zeros(
            (len(omega), 2, 2), dtype=_CDTYPE)
        port_y[:, 0, 0] = diagonal
        port_y[:, 0, 1] = mutual
        port_y[:, 1, 0] = mutual
        port_y[:, 1, 1] = diagonal
        incidence = torch.tensor(
            [[1.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, -1.0]],
            dtype=_CDTYPE)
        terminal_y = torch.einsum(
            "pi,lpq,qj->lij", incidence, port_y, incidence)
        return MultiportLinearization(
            terminal_y,
            torch.zeros((len(omega), 4), dtype=_CDTYPE),
        )

    return EmpiricalCircuitLaw(identity, terminal_names, linearize)


def battery_port_law(
    identity: str,
    node: str,
    battery: object,
    return_node: str = GROUND,
) -> EmpiricalPortLaw:
    """Expose an existing battery object through the common empirical ABI."""

    def voltage() -> float:
        for name in ("terminal_v", "open_circuit_voltage_v", "nominal_voltage_v"):
            if hasattr(battery, name):
                return float(getattr(battery, name))
        raise TypeError("registered battery does not publish terminal voltage")

    def resistance() -> complex:
        for name in ("internal_r_ohm", "internal_resistance_ohm"):
            if hasattr(battery, name):
                return complex(getattr(battery, name))
        raise TypeError("registered battery does not publish internal resistance")

    def linearize(_voltage_ab, frequencies_hz):
        lanes = int(frequencies_hz.numel())
        y = _phasors(1.0 / resistance(), lanes)
        emf = torch.zeros(lanes, dtype=_CDTYPE)
        emf[0] = complex(voltage(), 0.0)
        return PortLinearization(y, -y * emf)

    def advance(voltage_ab, current_ab, dt_s):
        absorbed_w = float(torch.real(
            voltage_ab * torch.conj(current_ab)).sum())
        step = getattr(battery, "step", None)
        if callable(step):
            step(float(dt_s), charge_w=max(0.0, absorbed_w),
                 load_w=max(0.0, -absorbed_w))
            return
        integrate = getattr(battery, "integrate", None)
        if callable(integrate):
            integrate(float(torch.real(current_ab[0])), float(dt_s))
            return
        raise TypeError("registered battery has no charge integration method")

    def snapshot_state():
        for name in ("state_of_charge", "soc_frac"):
            if hasattr(battery, name):
                return float(getattr(battery, name))
        raise TypeError("registered battery does not publish state of charge")

    def restore_state(value):
        for name in ("state_of_charge", "soc_frac"):
            if hasattr(battery, name):
                setattr(battery, name, float(value))
                return
        raise TypeError("registered battery does not own mutable charge state")

    return EmpiricalPortLaw(
        identity, node, return_node, linearize, advance,
        snapshot_state, restore_state,
    )


@dataclass(frozen=True)
class SeriesRLCBranch:
    """A physical series branch realized as a complex impedance vector.

    Any omitted element is absent.  A capacitor at DC is an open circuit and
    an inductor at DC is a short; those limits are represented explicitly
    rather than hidden behind an epsilon frequency.
    """

    identity: str
    a: str
    b: str
    resistance_ohm: float = 0.0
    inductance_h: float = 0.0
    capacitance_f: float = 0.0

    def impedance(self, frequencies_hz: torch.Tensor) -> torch.Tensor:
        omega = 2.0 * math.pi * frequencies_hz.to(torch.float64)
        z = torch.full_like(omega, float(self.resistance_ohm), dtype=_CDTYPE)
        if self.inductance_h:
            z = z + 1j * omega * float(self.inductance_h)
        if self.capacitance_f:
            capacitive = torch.full_like(z, complex(float("inf"), 0.0))
            moving = omega != 0.0
            capacitive[moving] = 1.0 / (
                1j * omega[moving] * float(self.capacitance_f)
            )
            z = z + capacitive
        return z

    def branch(self, frequencies_hz: torch.Tensor) -> ImpedanceBranch:
        return ImpedanceBranch(self.identity, self.a, self.b,
                               self.impedance(frequencies_hz))


@dataclass(frozen=True)
class ComplexCircuitReading:
    voltage_v: Mapping[str, torch.Tensor]
    branch_current_a: Mapping[str, torch.Tensor]
    source_current_a: Mapping[str, torch.Tensor]
    device_current_a: Mapping[str, torch.Tensor] | None = None
    circuit_terminal_current_a: Mapping[str, torch.Tensor] | None = None

    def complex_power_va(self, node: str, current_a: torch.Tensor) -> torch.Tensor:
        """Per-lane complex power ``S = V * conj(I)`` at one boundary."""
        return self.voltage_v[node] * torch.conj(current_a)


class ComplexTensorCircuit:
    """One simultaneous complex nodal solve over an arbitrary lane vector."""

    def __init__(
        self,
        *,
        lanes: int,
        branches: Sequence[ImpedanceBranch],
        sources: Sequence[TheveninSource] = (),
        injections: Sequence[CurrentInjection] = (),
        stamps: Sequence[AdmittanceStamp] = (),
    ) -> None:
        self.lanes = int(lanes)
        if self.lanes <= 0:
            raise ValueError("an electrical circuit needs at least one phasor lane")
        self.branches = tuple(branches)
        self.sources = tuple(sources)
        self.injections = tuple(injections)
        self.stamps = tuple(stamps)
        self._native_nodal_solver = None
        names = {
            endpoint
            for branch in self.branches
            for endpoint in (branch.a, branch.b)
            if endpoint != GROUND
        }
        names.update(
            endpoint
            for source in self.sources
            for endpoint in (source.node, source.return_node)
            if endpoint != GROUND
        )
        names.update(row.node for row in self.injections if row.node != GROUND)
        names.update(
            terminal for stamp in self.stamps for terminal in stamp.terminals
            if terminal != GROUND)
        self.nodes = tuple(sorted(names))
        if not self.nodes:
            raise ValueError("an electrical circuit needs at least one non-ground node")
        self._admittance: dict[str, torch.Tensor] = {
            name: torch.zeros(self.lanes, dtype=_CDTYPE) for name in self.nodes
        }
        mutual: dict[tuple[str, str], torch.Tensor] = {}

        for branch in self.branches:
            z = _phasors(branch.impedance_ohm, self.lanes)
            if torch.any(torch.abs(z) == 0):
                raise ValueError(f"{branch.identity}: zero branch impedance")
            y = 1.0 / z
            for endpoint in (branch.a, branch.b):
                if endpoint != GROUND:
                    self._admittance[endpoint] += y
            if branch.a != GROUND and branch.b != GROUND:
                mutual[(branch.b, branch.a)] = mutual.get(
                    (branch.b, branch.a), torch.zeros_like(y)) + y
                mutual[(branch.a, branch.b)] = mutual.get(
                    (branch.a, branch.b), torch.zeros_like(y)) + y

        for source in self.sources:
            if source.node == source.return_node:
                raise ValueError(
                    f"{source.identity}: source terminals must be distinct")
            z = _phasors(source.impedance_ohm, self.lanes)
            if torch.any(torch.abs(z) == 0):
                raise ValueError(f"{source.identity}: zero source impedance")
            y = 1.0 / z
            for endpoint in (source.node, source.return_node):
                if endpoint != GROUND:
                    self._admittance[endpoint] += y
            if source.node != GROUND and source.return_node != GROUND:
                mutual[(source.return_node, source.node)] = mutual.get(
                    (source.return_node, source.node), torch.zeros_like(y)) + y
                mutual[(source.node, source.return_node)] = mutual.get(
                    (source.node, source.return_node), torch.zeros_like(y)) + y

        self._stamp_injection: dict[str, torch.Tensor] = {
            name: torch.zeros(self.lanes, dtype=_CDTYPE) for name in self.nodes
        }
        for stamp in self.stamps:
            count = len(stamp.terminals)
            if not count or len(set(stamp.terminals)) != count:
                raise ValueError(
                    f"{stamp.identity}: circuit terminals must be nonempty and unique")
            y = _multiport_matrix(stamp.admittance_s, self.lanes, count)
            offset = _multiport_vector(
                stamp.current_offset_a, self.lanes, count)
            for row, dst in enumerate(stamp.terminals):
                if dst == GROUND:
                    continue
                self._admittance[dst] += y[:, row, row]
                self._stamp_injection[dst] -= offset[:, row]
                for column, src in enumerate(stamp.terminals):
                    if column == row or src == GROUND:
                        continue
                    # GraphSolver stores the rearranged nodal coefficient
                    # -Y_ij/Y_ii rather than the raw off-diagonal stamp.
                    coupling = -y[:, row, column]
                    mutual[(src, dst)] = mutual.get(
                        (src, dst), torch.zeros_like(coupling)) + coupling

        for node, diagonal in self._admittance.items():
            if torch.any(torch.abs(diagonal) == 0):
                raise ValueError(f"{node}: floating phasor lane has zero admittance")

        # Construct the Python GraphSolver only if its fallback path is
        # actually selected.  Native circuits used to instantiate it before
        # their LLVM solver was installed, which produced a misleading
        # "GraphSolver -- no timing data" shutdown report even though every
        # nodal solve had executed in LLVM.
        self._graph_edge_specs = tuple(
            (src, dst, y / self._admittance[dst])
            for (src, dst), y in mutual.items())
        self.solver = None

        # Preserve the unnormalised nodal law as the exact circuit-bake input.
        # GraphSolver consumes the rearranged Jacobi form above; circuit baking
        # needs Y and the source vector before that representation change.
        node_index = {name: index for index, name in enumerate(self.nodes)}
        self._nodal_admittance = torch.zeros(
            (self.lanes, len(self.nodes), len(self.nodes)), dtype=_CDTYPE)
        for name, diagonal in self._admittance.items():
            index = node_index[name]
            self._nodal_admittance[:, index, index] = diagonal
        for (src, dst), rearranged in mutual.items():
            self._nodal_admittance[:, node_index[dst], node_index[src]] \
                -= rearranged

        default_injection = {
            name: value.detach().clone()
            for name, value in self._stamp_injection.items()
        }
        for source in self.sources:
            norton = (_phasors(source.voltage_v, self.lanes)
                      / _phasors(source.impedance_ohm, self.lanes))
            if source.node != GROUND:
                default_injection[source.node] += norton
            if source.return_node != GROUND:
                default_injection[source.return_node] -= norton
        for row in self.injections:
            if row.node == GROUND:
                raise ValueError(
                    f"{row.identity}: current injection cannot target reference")
            default_injection[row.node] += _phasors(row.current_a, self.lanes)
        # Device current is positive into the circuit: I = YV - J.
        self._nodal_current_offset = -torch.stack([
            default_injection[name] for name in self.nodes
        ], dim=1)

    def nodal_form(self) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
        """Return ``(nodes, Y, I_offset)`` for exact circuit reduction."""
        return (
            self.nodes,
            self._nodal_admittance.detach().clone(),
            self._nodal_current_offset.detach().clone(),
        )

    def use_native_nodal_solver(self, solver) -> None:
        """Use a compiled solver for this circuit's existing nodal equation."""
        if int(solver.lanes) != self.lanes or int(solver.nodes) != len(self.nodes):
            raise ValueError(
                "native nodal solver shape does not match circuit: "
                f"solver=({solver.lanes}, {solver.nodes}), "
                f"circuit=({self.lanes}, {len(self.nodes)})")
        self._native_nodal_solver = solver

    @property
    def solve_backend(self) -> str:
        """Name the numerical backend that will execute the nodal solve."""
        return "llvm" if self._native_nodal_solver is not None else "python"

    def solve(
        self,
        *,
        source_voltage_v: Mapping[str, complex | Sequence[complex] | torch.Tensor] | None = None,
        injection_current_a: Mapping[str, complex | Sequence[complex] | torch.Tensor] | None = None,
    ) -> ComplexCircuitReading:
        source_voltage_v = source_voltage_v or {}
        injection_current_a = injection_current_a or {}
        injection = {
            name: value.detach().clone()
            for name, value in self._stamp_injection.items()
        }
        for source in self.sources:
            voltage = _phasors(
                source_voltage_v.get(source.identity, source.voltage_v), self.lanes
            )
            impedance = _phasors(source.impedance_ohm, self.lanes)
            norton_current = voltage / impedance
            if source.node != GROUND:
                injection[source.node] += norton_current
            if source.return_node != GROUND:
                injection[source.return_node] -= norton_current
        for row in self.injections:
            injection[row.node] += _phasors(
                injection_current_a.get(row.identity, row.current_a), self.lanes
            )
        if self._native_nodal_solver is None:
            if self.solver is None:
                from graph_solver import GraphSolver, TensorEdge, TensorNode
                self.solver = GraphSolver(
                    nodes=[TensorNode(name, layer="electrical")
                           for name in self.nodes],
                    edges=[TensorEdge(
                        src, dst, weight=weight,
                        semantic_role="electrical-mutual-admittance")
                        for src, dst, weight in self._graph_edge_specs])
            normalized = {
                name: injection[name] / self._admittance[name] for name in self.nodes
            }
            solved = self.solver.step(normalized)
            voltage = {
                name: solved[name].reshape(self.lanes) for name in self.nodes
            }
        else:
            current = torch.stack([injection[name] for name in self.nodes], dim=1)
            solved = self._native_nodal_solver.solve(
                self._nodal_admittance.detach().cpu().numpy(),
                current.detach().cpu().numpy(),
            )
            voltage_tensor = torch.as_tensor(solved, dtype=_CDTYPE)
            voltage = {
                name: voltage_tensor[:, index]
                for index, name in enumerate(self.nodes)
            }
        branch_current = {}
        for branch in self.branches:
            va = (torch.zeros(self.lanes, dtype=_CDTYPE) if branch.a == GROUND
                  else voltage[branch.a])
            vb = (torch.zeros(self.lanes, dtype=_CDTYPE) if branch.b == GROUND
                  else voltage[branch.b])
            branch_current[branch.identity] = (
                (va - vb) / _phasors(branch.impedance_ohm, self.lanes)
            )
        source_current = {}
        zero = torch.zeros(self.lanes, dtype=_CDTYPE)
        for source in self.sources:
            va = zero if source.node == GROUND else voltage[source.node]
            vb = (zero if source.return_node == GROUND
                  else voltage[source.return_node])
            source_current[source.identity] = (
                _phasors(
                    source_voltage_v.get(source.identity, source.voltage_v),
                    self.lanes,
                ) - (va - vb)
            ) / _phasors(source.impedance_ohm, self.lanes)
        circuit_current = {}
        for stamp in self.stamps:
            count = len(stamp.terminals)
            terminal_voltage = torch.stack([
                zero if terminal == GROUND else voltage[terminal]
                for terminal in stamp.terminals
            ], dim=1)
            y = _multiport_matrix(stamp.admittance_s, self.lanes, count)
            offset = _multiport_vector(
                stamp.current_offset_a, self.lanes, count)
            circuit_current[stamp.identity] = (
                torch.einsum("lij,lj->li", y, terminal_voltage) + offset)
        return ComplexCircuitReading(
            voltage, branch_current, source_current,
            circuit_terminal_current_a=circuit_current,
        )


def bake_linear_circuit(
    identity: str,
    circuit: ComplexTensorCircuit,
    terminals: Sequence[str],
) -> EmpiricalCircuitLaw:
    """Exactly eliminate a linear circuit's internal nodes.

    The reduction is the Schur complement of the internal nodal block.  It
    preserves the complete complex terminal law on every spectral lane,
    including independent sources already present in ``circuit``.
    """
    terminal_names = tuple(str(name) for name in terminals)
    if not terminal_names or len(set(terminal_names)) != len(terminal_names):
        raise ValueError("baked circuit terminals must be nonempty and unique")
    nodes, admittance, offset = circuit.nodal_form()
    missing = set(terminal_names) - set(nodes)
    if missing:
        raise KeyError(f"baked circuit has unknown terminals: {sorted(missing)}")
    terminal_indices = [nodes.index(name) for name in terminal_names]
    internal_indices = [
        index for index, name in enumerate(nodes)
        if name not in terminal_names
    ]
    y_tt = admittance[:, terminal_indices][:, :, terminal_indices]
    i_t = offset[:, terminal_indices]
    if internal_indices:
        y_ti = admittance[:, terminal_indices][:, :, internal_indices]
        y_it = admittance[:, internal_indices][:, :, terminal_indices]
        y_ii = admittance[:, internal_indices][:, :, internal_indices]
        i_i = offset[:, internal_indices]
        solved_y = torch.linalg.solve(y_ii, y_it)
        solved_i = torch.linalg.solve(y_ii, i_i.unsqueeze(-1)).squeeze(-1)
        reduced_y = y_tt - torch.matmul(y_ti, solved_y)
        reduced_i = i_t - torch.einsum("lij,lj->li", y_ti, solved_i)
    else:
        reduced_y = y_tt
        reduced_i = i_t

    def linearize(_terminal_voltage, _frequencies):
        return MultiportLinearization(reduced_y, reduced_i)

    return EmpiricalCircuitLaw(identity, terminal_names, linearize)


class ElectricalDeviceRegistry:
    """Build-time registration of electrical producers and consumers.

    The registry contains topology and device laws only.  ``GraphSolver`` is
    still the one simultaneous runtime solve.
    """

    def __init__(self, frequencies_hz: Sequence[float] | torch.Tensor) -> None:
        self.frequencies_hz = torch.as_tensor(
            frequencies_hz, dtype=torch.float64).reshape(-1)
        if self.frequencies_hz.numel() == 0:
            raise ValueError("an electrical registry needs at least one lane")
        if torch.any(self.frequencies_hz < 0.0):
            raise ValueError("electrical frequencies cannot be negative")
        self.branches: list[ImpedanceBranch] = []
        self.sources: list[TheveninSource] = []
        self.injections: list[CurrentInjection] = []
        self.port_laws: dict[str, EmpiricalPortLaw] = {}
        self.circuit_laws: dict[str, EmpiricalCircuitLaw] = {}
        self.stamps: list[AdmittanceStamp] = []
        self._law_branch_index: dict[str, int] = {}
        self._law_injection_ids: dict[str, tuple[str | None, str | None]] = {}
        self._circuit_stamp_index: dict[str, int] = {}
        self._last_reading: ComplexCircuitReading | None = None
        self._identities: set[str] = set()
        self._circuit: ComplexTensorCircuit | None = None
        self.thermal_target_by_branch: dict[str, str] = {}
        self.thermal_fraction_by_branch: dict[str, float] = {}
        self.thermal_temperature_k: dict[str, float] = {}
        self.safety_by_branch: dict[str, dict[str, Any]] = {}

    @property
    def lanes(self) -> int:
        return int(self.frequencies_hz.numel())

    @property
    def solve_backend(self) -> str:
        """Backend selected for the circuit's simultaneous nodal solve."""
        return "llvm" if self._native_nodal_solver is not None else "python"

    def _claim(self, identity: str) -> None:
        if not identity or identity in self._identities:
            raise ValueError(f"duplicate or empty electrical identity: {identity!r}")
        self._identities.add(identity)
        self._circuit = None
        self._native_nodal_solver = None

    def register_impedance(self, branch: ImpedanceBranch) -> None:
        self._claim(branch.identity)
        self.branches.append(branch)

    def register_rlc(self, branch: SeriesRLCBranch) -> None:
        self.register_impedance(branch.branch(self.frequencies_hz))

    def register_source(self, source: TheveninSource) -> None:
        self._claim(source.identity)
        self.sources.append(source)

    def register_current(self, source: CurrentInjection) -> None:
        self._claim(source.identity)
        self.injections.append(source)

    def set_thermal_temperatures(self, values: Mapping[str, float]) -> None:
        """Update temperatures read by temperature-sensitive device laws."""
        self.thermal_temperature_k = {
            str(identity): float(value) for identity, value in values.items()
        }

    def register(self, law: EmpiricalPortLaw) -> None:
        """Register any two-terminal object through one constitutive ABI."""
        self._claim(law.identity)
        zero = torch.zeros(self.lanes, dtype=_CDTYPE)
        tangent = law.tangent(zero, self.frequencies_hz)
        admittance = _phasors(tangent.admittance_s, self.lanes)
        impedance = torch.full_like(admittance, complex(float("inf"), 0.0))
        conducting = torch.abs(admittance) > 0.0
        impedance[conducting] = 1.0 / admittance[conducting]
        self._law_branch_index[law.identity] = len(self.branches)
        self.branches.append(ImpedanceBranch(
            law.identity, law.a, law.b, impedance))
        offset = _phasors(tangent.current_offset_a, self.lanes)
        a_id = b_id = None
        if law.a != GROUND:
            a_id = f"{law.identity}::offset:a"
            self.injections.append(CurrentInjection(a_id, law.a, -offset))
        if law.b != GROUND:
            b_id = f"{law.identity}::offset:b"
            self.injections.append(CurrentInjection(b_id, law.b, offset))
        self._law_injection_ids[law.identity] = (a_id, b_id)
        self.port_laws[law.identity] = law

    def register_circuit(self, law: EmpiricalCircuitLaw) -> None:
        """Register one baked board or component as a multi-terminal stamp."""
        self._claim(law.identity)
        terminals = tuple(law.terminals)
        if not terminals or len(set(terminals)) != len(terminals):
            raise ValueError(
                f"{law.identity}: circuit terminals must be nonempty and unique")
        voltage = torch.zeros(
            (self.lanes, len(terminals)), dtype=_CDTYPE)
        tangent = law.tangent(voltage, self.frequencies_hz)
        stamp = AdmittanceStamp(
            law.identity, terminals,
            _multiport_matrix(tangent.admittance_s, self.lanes, len(terminals)),
            _multiport_vector(tangent.current_offset_a,
                              self.lanes, len(terminals)),
        )
        self._circuit_stamp_index[law.identity] = len(self.stamps)
        self.stamps.append(stamp)
        self.circuit_laws[law.identity] = law

    def _circuit_law_tangents(self):
        out = {}
        for identity, law in self.circuit_laws.items():
            count = len(law.terminals)
            if self._last_reading is None:
                voltage = torch.zeros((self.lanes, count), dtype=_CDTYPE)
            else:
                zero = torch.zeros(self.lanes, dtype=_CDTYPE)
                voltage = torch.stack([
                    zero if terminal == GROUND
                    else self._last_reading.voltage_v[terminal]
                    for terminal in law.terminals
                ], dim=1)
            out[identity] = law.tangent(voltage, self.frequencies_hz)
        return out

    def _refresh_circuit_stamps(self, tangents) -> None:
        changed = False
        for identity, tangent in tangents.items():
            index = self._circuit_stamp_index[identity]
            previous = self.stamps[index]
            count = len(previous.terminals)
            replacement = AdmittanceStamp(
                identity, previous.terminals,
                _multiport_matrix(tangent.admittance_s, self.lanes, count),
                _multiport_vector(tangent.current_offset_a, self.lanes, count),
            )
            if (not torch.equal(previous.admittance_s,
                                replacement.admittance_s)
                    or not torch.equal(previous.current_offset_a,
                                       replacement.current_offset_a)):
                self.stamps[index] = replacement
                changed = True
        if changed:
            self._circuit = None

    def _law_tangents(self):
        zero = torch.zeros(self.lanes, dtype=_CDTYPE)
        out = {}
        for identity, law in self.port_laws.items():
            if self._last_reading is None:
                voltage_ab = zero
            else:
                va = (zero if law.a == GROUND
                      else self._last_reading.voltage_v[law.a])
                vb = (zero if law.b == GROUND
                      else self._last_reading.voltage_v[law.b])
                voltage_ab = va - vb
            out[identity] = law.tangent(voltage_ab, self.frequencies_hz)
        return out

    def _refresh_law_admittance(self, tangents) -> None:
        changed = False
        for identity, tangent in tangents.items():
            index = self._law_branch_index[identity]
            branch = self.branches[index]
            admittance = _phasors(tangent.admittance_s, self.lanes)
            impedance = torch.full_like(
                admittance, complex(float("inf"), 0.0))
            conducting = torch.abs(admittance) > 0.0
            impedance[conducting] = 1.0 / admittance[conducting]
            previous = _phasors(branch.impedance_ohm, self.lanes)
            if not torch.equal(previous, impedance):
                self.branches[index] = ImpedanceBranch(
                    branch.identity, branch.a, branch.b, impedance)
                changed = True
        if changed:
            self._circuit = None

    def circuit(self) -> "ComplexTensorCircuit":
        if self._circuit is None:
            self._circuit = ComplexTensorCircuit(
                lanes=self.lanes,
                branches=self.branches,
                sources=self.sources,
                injections=self.injections,
                stamps=self.stamps,
            )
            if self._native_nodal_solver is not None:
                self._circuit.use_native_nodal_solver(self._native_nodal_solver)
        return self._circuit

    def compile_llvm_solver(self, directory) -> None:
        """Compile and install the circuit's fixed-shape simultaneous solve."""
        circuit = self.circuit()
        from src.compiler.electrical_llvm import compile_complex_nodal_solver
        self._native_nodal_solver = compile_complex_nodal_solver(
            directory,
            lanes=self.lanes,
            nodes=len(circuit.nodes),
            name="electrical_nodal_solve",
        )
        circuit.use_native_nodal_solver(self._native_nodal_solver)

    def solve(
        self,
        *,
        injection_current_a: Mapping[str, complex | Sequence[complex] | torch.Tensor] | None = None,
    ) -> ComplexCircuitReading:
        tangents = self._law_tangents()
        circuit_tangents = self._circuit_law_tangents()
        self._refresh_law_admittance(tangents)
        self._refresh_circuit_stamps(circuit_tangents)
        currents = dict(injection_current_a or {})
        for identity, tangent in tangents.items():
            offset = _phasors(tangent.current_offset_a, self.lanes)
            a_id, b_id = self._law_injection_ids[identity]
            if a_id is not None:
                currents[a_id] = -offset
            if b_id is not None:
                currents[b_id] = offset
        reading = self.circuit().solve(
            injection_current_a=currents,
        )
        zero = torch.zeros(self.lanes, dtype=_CDTYPE)
        device_current = {}
        for identity, law in self.port_laws.items():
            va = zero if law.a == GROUND else reading.voltage_v[law.a]
            vb = zero if law.b == GROUND else reading.voltage_v[law.b]
            tangent = tangents[identity]
            device_current[identity] = (
                _phasors(tangent.admittance_s, self.lanes) * (va - vb)
                + _phasors(tangent.current_offset_a, self.lanes)
            )
        reading = ComplexCircuitReading(
            reading.voltage_v, reading.branch_current_a,
            reading.source_current_a, device_current,
            reading.circuit_terminal_current_a,
        )
        self._last_reading = reading
        return reading

    def advance(self, reading: ComplexCircuitReading, dt_s: float) -> None:
        zero = torch.zeros(self.lanes, dtype=_CDTYPE)
        for identity, law in self.port_laws.items():
            if law.advance is None:
                continue
            va = zero if law.a == GROUND else reading.voltage_v[law.a]
            vb = zero if law.b == GROUND else reading.voltage_v[law.b]
            law.advance(va - vb, reading.device_current_a[identity], float(dt_s))
        for identity, law in self.circuit_laws.items():
            if law.advance is None:
                continue
            voltage = torch.stack([
                zero if terminal == GROUND else reading.voltage_v[terminal]
                for terminal in law.terminals
            ], dim=1)
            law.advance(
                voltage, reading.circuit_terminal_current_a[identity], float(dt_s))

    def snapshot(self):
        return tuple(
            (identity, law.snapshot_state())
            for identity, law in sorted(
                {**self.port_laws, **self.circuit_laws}.items())
            if law.snapshot_state is not None
        )

    def restore(self, snapshot) -> None:
        for identity, value in snapshot:
            law = self.port_laws.get(identity) or self.circuit_laws[identity]
            restore = law.restore_state
            if restore is None:
                raise TypeError(f"{identity}: electrical state is not restorable")
            restore(value)
        self._last_reading = None
