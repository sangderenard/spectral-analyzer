"""Integration test: score generation → ScoreSequencerNode → VoiceArchetype → audio.

Full pipeline exercised:
  SparseScoreTensor (2 notes)
    → dispatch_before_start → ScoreSequencerNode._hook
    → envelope_jobs_from_sparse_score + performance_atoms_from_jobs
    → ObjectFifoSlot
    → VoiceArchetype.forward (atom drain path)
    → synthesize_atoms (per-sample NoteStateMachine evaluation)
    → run_schedule output tensor

All nodes are real TensorNodes, all edges are real TensorEdges, FIFO bank
carries real ObjectFifoSlots.  No mocks.
"""
import os
import sys
import types
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from audio_projector_node import AudioProjectorNode
from edge_fifo_bank import EdgeFifoBank
from graph_solver import GraphSolver, TensorEdge, TensorNode
from parametric_curve import default_chirp, default_envelope
from score_sequencer_node import ScoreSequencerNode
from torch_composer_engine import (
    EVENT_NOTE,
    PARAM_DURATION,
    PARAM_HZ,
    PARAM_PATTERN,
    PARAM_START,
    PARAM_VELOCITY,
    empty_score_tensor,
    envelope_jobs_from_sparse_score,
    performance_atoms_from_jobs,
    slice_score_for_consumer,
    sparse_score_tensor_from_score,
)
from voice_graph_node import MetaVoiceNode, build_voice_mixer_network

# Use a small sample rate for unit/integration tests: per-sample Python
# synthesis in synthesize_atoms is intentionally simple (no vectorisation).
# At SR=1000 the note loop is ~200 iterations — runs in milliseconds.
SR = 1_000.0
N_FRAMES = 600          # 0.6 seconds
NOTE1_HZ = 261.63       # C4
NOTE2_HZ = 329.63       # E4
NOTE1_START = 0.0
NOTE2_START = 0.2       # 200 ms gap
NOTE_DUR = 0.12         # 120 ms sustain + release
RELEASE = 0.04


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_voice_ns(key: str = "v1", freq_hz: float = NOTE1_HZ) -> object:
    """Minimal voice namespace accepted by MetaVoiceNode.from_voice."""
    return types.SimpleNamespace(
        key=key,
        freq_hz=freq_hz,
        amplitude=1.0,
        phase_origin=0.0,
        semitone_offset=0.0,
        harmonic_brightness=1.0,
        harmonic_warp_strength=0.0,
        fm=None,
        am=None,
        manifold_type="pure",
        harmonic_count=1,
        loop_enabled=False,
        loop_start=0.1,
        loop_end=0.9,
        emission_mode="single",
        pre_delay=0.0,
        envelope=None,
        chirp=None,
        # score contract fields
        adsr=types.SimpleNamespace(release=RELEASE),
        score_groups=(),
        score_subtypes=(),
        score_pages=(),
        page_key="",
    )


def _two_note_sparse_score():
    """SparseScoreTensor with C4 at t=0 and E4 at t=0.5 (batch=1)."""
    score = empty_score_tensor(batch_size=1, max_events=2)
    score.labels[:] = EVENT_NOTE
    score.mask[:] = True
    score.params[0, 0, PARAM_START]    = NOTE1_START
    score.params[0, 0, PARAM_DURATION] = NOTE_DUR
    score.params[0, 0, PARAM_HZ]       = NOTE1_HZ
    score.params[0, 0, PARAM_VELOCITY] = 1.0
    score.params[0, 1, PARAM_START]    = NOTE2_START
    score.params[0, 1, PARAM_DURATION] = NOTE_DUR
    score.params[0, 1, PARAM_HZ]       = NOTE2_HZ
    score.params[0, 1, PARAM_VELOCITY] = 0.8
    return sparse_score_tensor_from_score(score)


def _flat_envelope(name: str = "flat") -> object:
    """Default envelope with all control-point values set to 1 (flat sustain)."""
    curve = default_envelope(name)
    for pt in curve.points:
        pt.v = 1.0
    return curve


# ─────────────────────────────────────────────────────────────────────────────
# Unit: synthesize_atoms produces signal at the right sample offsets
# ─────────────────────────────────────────────────────────────────────────────

def test_synthesize_atoms_note_at_onset_sample() -> None:
    """synthesize_atoms places signal at the correct onset sample."""
    from voice_graph_node import synthesize_atoms

    score = empty_score_tensor(batch_size=1, max_events=1)
    score.labels[0, 0] = EVENT_NOTE
    score.mask[0, 0] = True
    score.params[0, 0, PARAM_START]    = 0.25
    score.params[0, 0, PARAM_DURATION] = 0.1
    score.params[0, 0, PARAM_HZ]       = 440.0
    score.params[0, 0, PARAM_VELOCITY] = 1.0
    packet = sparse_score_tensor_from_score(score)
    jobs   = envelope_jobs_from_sparse_score(packet, sample_rate=SR, release_tail_s=0.05)

    env_curve = _flat_envelope()
    chirp_curve = default_chirp("chirp")

    atoms = performance_atoms_from_jobs(
        jobs,
        envelope_curves=[env_curve],
        chirp_curves=[chirp_curve],
        voice_key="v1",
        sample_rate=SR,
    )
    assert len(atoms) == 1
    onset = atoms[0].onset_sample  # should be 0.25 * 48000 = 12000

    n_frames = onset + atoms[0].sample_count + 1000
    audio = synthesize_atoms(atoms, n_frames, SR)

    assert audio.dtype == torch.complex128
    assert audio.shape[0] == n_frames
    # samples before onset should be exactly zero
    assert audio[:onset].abs().max().item() == 0.0
    # samples in the note body should be non-zero
    mid = onset + atoms[0].sample_count // 4
    assert audio[mid].abs().item() > 0.0


def test_synthesize_atoms_gap_between_notes_is_silent() -> None:
    """Gap between two atoms must be exactly zero."""
    from voice_graph_node import synthesize_atoms

    packet = _two_note_sparse_score()
    jobs   = envelope_jobs_from_sparse_score(packet, sample_rate=SR, release_tail_s=RELEASE)
    env    = _flat_envelope()
    chirp  = default_chirp("c")
    n      = jobs.job_count
    atoms  = performance_atoms_from_jobs(
        jobs,
        envelope_curves=[env] * n,
        chirp_curves=[chirp] * n,
        voice_key="v1",
        sample_rate=SR,
    )

    assert len(atoms) == 2

    audio = synthesize_atoms(atoms, N_FRAMES, SR)

    note1_end   = atoms[0].onset_sample + atoms[0].sample_count   # ~23040
    note2_onset = atoms[1].onset_sample                            # ~24000
    gap = audio[note1_end : note2_onset]

    assert gap.abs().max().item() == 0.0, "gap between notes must be silent"


# ─────────────────────────────────────────────────────────────────────────────
# Unit: ScoreSequencerNode writes atoms into the FIFO bank
# ─────────────────────────────────────────────────────────────────────────────

def test_sequencer_hook_writes_atoms_to_fifo_bank() -> None:
    """dispatch_before_start fires the hook and atoms appear in the bank."""
    voice_ns    = _make_voice_ns()
    voice_solver, voice_nodes, _ = build_voice_mixer_network(
        [voice_ns], sample_rate=SR, duration=2.0
    )
    vnode: MetaVoiceNode = voice_nodes["v1"]
    # Use a flat envelope so atoms will have deterministic output
    for pt in vnode.oscillator.envelope_curve.points:
        pt.v = 1.0

    bank  = EdgeFifoBank(stride=1, fifo_size=32)
    score = _two_note_sparse_score()
    seq   = ScoreSequencerNode("seq", score, sample_rate=SR, fifo_bank=bank)

    seq_node   = seq.build_node()
    score_edge = TensorEdge(
        "seq", "v1_out",
        weight=0.0 + 0.0j,
        semantic_role="score",
        src_port="score_out",
        dst_port="score_in",
    )

    solver = GraphSolver(
        [seq_node] + list(voice_solver.nodes),
        [score_edge] + list(voice_solver.edges),
        sample_rate=SR,
    )

    assert len(solver.contract_edges) == 1
    assert solver.contract_edges[0].src_key == "seq"
    assert solver.contract_edges[0].dst_key == "v1_out"
    assert solver.contract_edges[0].contract["edge"]["transport"] == "object_fifo"
    assert solver.contract_edges[0].contract["edge"]["object_fifo_key"] == "seq_v1_out_atoms"
    assert all(
        (edge.src_key, edge.dst_key) != ("seq", "v1_out")
        for edge in solver.solve_edges
    )
    assert vnode._fifo_bank is bank
    assert vnode._fifo_slot_key == "seq_v1_out_atoms"
    assert bank.has_slot("seq_v1_out_atoms")
    assert bank.count("seq_v1_out_atoms") == 0

    solver.dispatch_before_start()

    slot_key = "seq_v1_out_atoms"
    assert bank.has_slot(slot_key), "bank must contain the atom slot after hook fires"
    assert bank.count(slot_key) == 1, "exactly one batch written"

    atoms = bank.try_read(slot_key)
    assert isinstance(atoms, list)
    assert len(atoms) == 2, "two notes → two atoms"
    assert abs(atoms[0].fundamental_hz - NOTE1_HZ) < 0.1
    assert abs(atoms[1].fundamental_hz - NOTE2_HZ) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full solver pipeline produces non-zero audio at note positions
# ─────────────────────────────────────────────────────────────────────────────

def test_full_sequencer_voice_pipeline_produces_audio_at_note_positions() -> None:
    """End-to-end: score → FIFO → VoiceArchetype → run_schedule → audio signal.

    Two notes are scheduled.  The audio output must be:
      - non-zero during both note bodies
      - exactly zero in the silent gap between them
    """
    # Build voice network
    voice_ns = _make_voice_ns()
    voice_solver, voice_nodes, _ = build_voice_mixer_network(
        [voice_ns], sample_rate=SR, duration=2.0
    )
    vnode: MetaVoiceNode = voice_nodes["v1"]
    # Flat envelope ensures deterministic non-zero signal
    for pt in vnode.oscillator.envelope_curve.points:
        pt.v = 1.0

    # Build score + sequencer
    bank  = EdgeFifoBank(stride=1, fifo_size=32)
    score = _two_note_sparse_score()
    seq   = ScoreSequencerNode("seq", score, sample_rate=SR, fifo_bank=bank)

    seq_node   = seq.build_node()
    score_edge = TensorEdge(
        "seq", "v1_out",
        weight=0.0 + 0.0j,
        semantic_role="score",
        src_port="score_out",
        dst_port="score_in",
    )

    # Build full solver: sequencer + voice + mixer
    full_solver = GraphSolver(
        [seq_node] + list(voice_solver.nodes),
        [score_edge] + list(voice_solver.edges),
        sample_rate=SR,
    )

    # Fire sequencer hook → atoms land in FIFO; GraphSolver has already attached
    # the voice to this object FIFO from the score contract metadata.
    full_solver.dispatch_before_start()

    # Run the full schedule
    outputs = full_solver.run_schedule({}, n_frames=N_FRAMES)

    # ── Output shape and dtype ────────────────────────────────────────────────
    assert "mix_out" in outputs
    out = outputs["mix_out"]
    assert out.dtype == torch.complex128
    # Shape: (B, n_frames, C) — B=1, C=1
    assert out.dim() == 3
    assert out.shape[1] == N_FRAMES

    audio = out[0, :, 0]   # (N_FRAMES,) complex128

    # ── Determine expected sample boundaries ─────────────────────────────────
    # re-derive from jobs so the test is self-consistent
    packet = _two_note_sparse_score()
    jobs   = envelope_jobs_from_sparse_score(packet, sample_rate=SR, release_tail_s=RELEASE)
    onset1 = int(jobs.sample_start[0].item())
    count1 = int(jobs.sample_count[0].item())
    onset2 = int(jobs.sample_start[1].item())

    gap_start = onset1 + count1    # first sample after note 1 ends
    gap_end   = onset2             # first sample of note 2

    # ── Signal assertions ─────────────────────────────────────────────────────
    # Note 1 body — quarter-way through to avoid potential near-zero attack edge
    n1_body_start = onset1 + max(1, count1 // 8)
    n1_body_end   = onset1 + count1 // 2
    n1_rms = audio[n1_body_start:n1_body_end].abs().mean().item()
    assert n1_rms > 1e-4, f"Note 1 body is silent: rms={n1_rms:.3e}"

    # Note 2 body
    count2 = int(jobs.sample_count[1].item())
    n2_body_start = onset2 + max(1, count2 // 8)
    n2_body_end   = onset2 + count2 // 2
    n2_rms = audio[n2_body_start:n2_body_end].abs().mean().item()
    assert n2_rms > 1e-4, f"Note 2 body is silent: rms={n2_rms:.3e}"

    # Gap between notes must be exactly zero
    if gap_start < gap_end:
        gap_max = audio[gap_start:gap_end].abs().max().item()
        assert gap_max == 0.0, f"Gap is not silent: max_abs={gap_max:.3e}"


def test_audio_projector_node_writes_wav_from_mixer_output(tmp_path) -> None:
    """Projector owns complex->PCM conversion and optional WAV output."""
    voice_ns = _make_voice_ns()
    voice_solver, voice_nodes, _ = build_voice_mixer_network(
        [voice_ns], sample_rate=SR, duration=2.0
    )
    vnode: MetaVoiceNode = voice_nodes["v1"]
    for pt in vnode.oscillator.envelope_curve.points:
        pt.v = 1.0

    bank = EdgeFifoBank(stride=1, fifo_size=32)
    seq = ScoreSequencerNode("seq", _two_note_sparse_score(), sample_rate=SR, fifo_bank=bank)
    projector_path = tmp_path / "projected_pipeline.wav"
    projector = AudioProjectorNode(
        "audio_out",
        sample_rate=SR,
        projection="real",
        wav_path=projector_path,
        normalize=False,
    )

    score_edge = TensorEdge(
        "seq",
        "v1_out",
        weight=0.0 + 0.0j,
        semantic_role="score",
        src_port="score_out",
        dst_port="score_in",
    )
    projector_edge = TensorEdge(
        "mix_out",
        "audio_out",
        weight=1.0 + 0.0j,
        semantic_role="audio_projection_source",
        src_port="mix_out",
        dst_port="audio_in",
    )
    solver = GraphSolver(
        [seq.build_node()] + list(voice_solver.nodes) + [projector.build_node()],
        [score_edge] + list(voice_solver.edges) + [projector_edge],
        sample_rate=SR,
    )

    solver.dispatch_before_start()
    outputs = solver.run_schedule({}, n_frames=N_FRAMES)

    assert "audio_out" in outputs
    projected = outputs["audio_out"]
    assert projected.dtype == torch.complex128
    assert projected.imag.abs().max().item() == 0.0
    assert projected.real.abs().max().item() > 0.0
    assert projector.last_write_path == projector_path

    with wave.open(str(projector_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == int(SR)
        assert wf.getnframes() == N_FRAMES
        frames = wf.readframes(N_FRAMES)
    assert any(byte != 0 for byte in frames)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: contract edge is not injected into the tensor solve
# ─────────────────────────────────────────────────────────────────────────────

def test_score_edge_does_not_pollute_tensor_solve_output() -> None:
    """A zero-weight score edge must not add any value to the voice accumulator."""
    voice_ns    = _make_voice_ns()
    voice_solver, voice_nodes, _ = build_voice_mixer_network(
        [voice_ns], sample_rate=SR, duration=2.0
    )

    bank  = EdgeFifoBank(stride=1, fifo_size=8)
    score = _two_note_sparse_score()
    seq   = ScoreSequencerNode("seq", score, sample_rate=SR, fifo_bank=bank)

    seq_node   = seq.build_node()
    score_edge = TensorEdge(
        "seq", "v1_out",
        weight=0.0 + 0.0j,
        semantic_role="score",
        src_port="score_out",
        dst_port="score_in",
    )

    solver = GraphSolver(
        [seq_node] + list(voice_solver.nodes),
        [score_edge] + list(voice_solver.edges),
        sample_rate=SR,
    )

    # Do NOT dispatch_before_start — no atoms in FIFO
    # Do NOT attach_fifo — voice uses oscillator path

    outputs = solver.run_schedule({}, n_frames=4)

    # seq node output must be zero (no inputs, no transform)
    assert outputs["seq"].abs().max().item() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: FIFO bank is the only channel — direct read/write roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_fifo_bank_object_slot_roundtrip_with_atoms() -> None:
    """EdgeFifoBank object slot stores and retrieves a list[PerformanceAtom]."""
    packet = _two_note_sparse_score()
    jobs   = envelope_jobs_from_sparse_score(packet, sample_rate=SR, release_tail_s=RELEASE)
    n      = jobs.job_count
    atoms  = performance_atoms_from_jobs(
        jobs,
        envelope_curves=[_flat_envelope()] * n,
        chirp_curves=[default_chirp("c")] * n,
        voice_key="v1",
        sample_rate=SR,
    )

    bank = EdgeFifoBank(stride=1, fifo_size=8)
    slot_key = "test_atoms"
    bank.claim_object(slot_key, fifo_size=4)
    bank.write_batch(slot_key, atoms)

    assert bank.count(slot_key) == 1

    retrieved = bank.try_read(slot_key)
    assert retrieved is atoms          # same list object
    assert len(retrieved) == n
    assert bank.count(slot_key) == 0  # consumed
    assert bank.try_read(slot_key) is None


def test_subscription_edge_can_bind_generic_object_fifo_transport() -> None:
    """Any declared subscription-port edge can use object FIFO metadata transport."""

    class _Producer:
        def __init__(self, bank: EdgeFifoBank) -> None:
            self.fifo_bank = bank
            self.subscription_ports = {}

    class _Consumer:
        def __init__(self) -> None:
            self.subscription_ports = {}
            self.attached = None

        def attach_fifo(self, bank: EdgeFifoBank, slot_key: str) -> None:
            self.attached = (bank, slot_key)

    bank = EdgeFifoBank(stride=1, fifo_size=4)
    producer = _Producer(bank)
    consumer = _Consumer()
    src = TensorNode(
        "producer",
        analytic_module=producer,
        subscription_ports=("meta_out",),
        subscription_contracts={
            "meta_out": {
                "kind": "generic_metadata",
                "transport": "object_fifo",
                "object_fifo_suffix": "payloads",
            }
        },
    )
    dst = TensorNode(
        "consumer",
        analytic_module=consumer,
        subscription_ports=("meta_in",),
        subscription_contracts={"meta_in": {"kind": "generic_consumer"}},
    )
    edge = TensorEdge(
        "producer",
        "consumer",
        weight=0.0 + 0.0j,
        semantic_role="metadata",
        src_port="meta_out",
        dst_port="meta_in",
    )

    solver = GraphSolver([src, dst], [edge], sample_rate=SR)

    assert len(solver.contract_edges) == 1
    assert not solver.solve_edges
    assert bank.has_slot("producer_consumer_payloads")
    assert bank.count("producer_consumer_payloads") == 0
    assert consumer.attached == (bank, "producer_consumer_payloads")


def test_object_fifo_transport_requires_one_shared_bank() -> None:
    """Graph construction rejects fragmented FIFO ownership."""

    class _Producer:
        def __init__(self, bank: EdgeFifoBank) -> None:
            self.fifo_bank = bank
            self.subscription_ports = {}

    class _Consumer:
        def __init__(self, bank: EdgeFifoBank) -> None:
            self.fifo_bank = bank
            self.subscription_ports = {}

        def attach_fifo(self, bank: EdgeFifoBank, slot_key: str) -> None:
            pass

    src = TensorNode(
        "producer",
        analytic_module=_Producer(EdgeFifoBank()),
        subscription_ports=("meta_out",),
        subscription_contracts={"meta_out": {"transport": "object_fifo"}},
    )
    dst = TensorNode(
        "consumer",
        analytic_module=_Consumer(EdgeFifoBank()),
        subscription_ports=("meta_in",),
        subscription_contracts={"meta_in": {}},
    )
    edge = TensorEdge(
        "producer",
        "consumer",
        weight=0.0 + 0.0j,
        src_port="meta_out",
        dst_port="meta_in",
    )

    import pytest
    with pytest.raises(ValueError, match="one shared EdgeFifoBank"):
        GraphSolver([src, dst], [edge], sample_rate=SR)


def _run_direct_demo() -> None:
    """Run the sequencer->object FIFO->voice pipeline when executed as a script."""
    voice_ns = _make_voice_ns()
    voice_solver, voice_nodes, _ = build_voice_mixer_network(
        [voice_ns], sample_rate=SR, duration=2.0
    )
    vnode: MetaVoiceNode = voice_nodes["v1"]
    for pt in vnode.oscillator.envelope_curve.points:
        pt.v = 1.0

    bank = EdgeFifoBank(stride=1, fifo_size=32)
    seq = ScoreSequencerNode("seq", _two_note_sparse_score(), sample_rate=SR, fifo_bank=bank)
    projector_path = os.path.abspath("sequencer_voice_pipeline_demo.wav")
    projector = AudioProjectorNode(
        "audio_out",
        sample_rate=SR,
        projection="real",
        wav_path=projector_path,
        normalize=False,
    )
    seq_node = seq.build_node()
    score_edge = TensorEdge(
        "seq",
        "v1_out",
        weight=0.0 + 0.0j,
        semantic_role="score",
        src_port="score_out",
        dst_port="score_in",
    )
    projector_edge = TensorEdge(
        "mix_out",
        "audio_out",
        weight=1.0 + 0.0j,
        semantic_role="audio_projection_source",
        src_port="mix_out",
        dst_port="audio_in",
    )
    solver = GraphSolver(
        [seq_node] + list(voice_solver.nodes) + [projector.build_node()],
        [score_edge] + list(voice_solver.edges) + [projector_edge],
        sample_rate=SR,
    )

    slot_key = "seq_v1_out_atoms"
    assert bank.has_slot(slot_key), "GraphSolver did not preclaim the object FIFO"
    assert vnode._fifo_bank is bank and vnode._fifo_slot_key == slot_key
    print(f"[demo] object FIFO preclaimed: {slot_key} capacity={bank.fifo_size}")

    solver.dispatch_before_start()
    atoms = bank.peek(slot_key)
    assert isinstance(atoms, list) and len(atoms) == 2
    print(
        "[demo] sequencer wrote atoms: "
        f"count={len(atoms)} hz={[round(a.fundamental_hz, 2) for a in atoms]}"
    )

    outputs = solver.run_schedule({}, n_frames=N_FRAMES)
    complex_audio = outputs["mix_out"][0, :, 0]
    audio = outputs["audio_out"][0, :, 0].real
    jobs = envelope_jobs_from_sparse_score(
        _two_note_sparse_score(),
        sample_rate=SR,
        release_tail_s=RELEASE,
    )
    onset1 = int(jobs.sample_start[0].item())
    count1 = int(jobs.sample_count[0].item())
    onset2 = int(jobs.sample_start[1].item())
    count2 = int(jobs.sample_count[1].item())
    n1 = audio[onset1 + max(1, count1 // 8) : onset1 + count1 // 2].abs().mean().item()
    n2 = audio[onset2 + max(1, count2 // 8) : onset2 + count2 // 2].abs().mean().item()
    gap_max = audio[onset1 + count1 : onset2].abs().max().item()

    assert n1 > 1e-4 and n2 > 1e-4
    assert gap_max == 0.0
    assert complex_audio.dtype == torch.complex128
    assert bank.count(slot_key) == 0
    print(
        "[demo] oscillator output: complex128 analytic signal; projector wrote real PCM WAV"
    )
    print(
        "[demo] projected audio verified: "
        f"note1_rms={n1:.6f} note2_rms={n2:.6f} gap_max={gap_max:.6f}"
    )
    print(f"[demo] wav: {projector_path}")


def test_group_subscription_routes_events_to_correct_voice() -> None:
    """Group-subscription kernel: one sequencer routes events to two voices by group tag.

    A combined score has 2 melody events (PARAM_PATTERN=0, "melody") and
    2 bass events (PARAM_PATTERN=1, "bass").  The sequencer carries
    group_keys={0: "melody", 1: "bass"}.  v1 declares score_groups=("melody",)
    and v2 declares score_groups=("bass",).

    After dispatch_before_start:
      - seq_v1_out_atoms contains exactly the 2 melody notes.
      - seq_v2_out_atoms contains exactly the 2 bass notes.
    """
    GROUP_MELODY, GROUP_BASS = 0, 1
    GROUP_KEYS = {GROUP_MELODY: "melody", GROUP_BASS: "bass"}

    MEL_HZ1, MEL_HZ2 = 261.63, 329.63   # C4, E4
    BAS_HZ1, BAS_HZ2 = 130.81, 196.00   # C3, G3

    # Build a combined score: 2 melody + 2 bass events, all at distinct Hz
    score = empty_score_tensor(1, 4)
    for ei, (hz, grp, start) in enumerate([
        (MEL_HZ1, GROUP_MELODY, 0.0),
        (MEL_HZ2, GROUP_MELODY, 0.2),
        (BAS_HZ1, GROUP_BASS,   0.0),
        (BAS_HZ2, GROUP_BASS,   0.2),
    ]):
        score.labels[0, ei]                 = EVENT_NOTE
        score.mask[0, ei]                   = True
        score.params[0, ei, PARAM_START]    = start
        score.params[0, ei, PARAM_DURATION] = 0.1
        score.params[0, ei, PARAM_HZ]       = hz
        score.params[0, ei, PARAM_VELOCITY] = 0.8
        score.params[0, ei, PARAM_PATTERN]  = float(grp)
    sparse = sparse_score_tensor_from_score(score)

    # Two voices with distinct group declarations
    v1_ns = _make_voice_ns("v1", MEL_HZ1)
    v1_ns.score_groups = ("melody",)
    v2_ns = _make_voice_ns("v2", BAS_HZ1)
    v2_ns.score_groups = ("bass",)

    voice_solver, _, _ = build_voice_mixer_network(
        [v1_ns, v2_ns], sample_rate=SR, duration=0.6,
    )

    bank  = EdgeFifoBank(stride=1, fifo_size=16)
    seq   = ScoreSequencerNode("seq", sparse, sample_rate=SR, fifo_bank=bank,
                               group_keys=GROUP_KEYS)

    solver = GraphSolver(
        [seq.build_node()] + list(voice_solver.nodes),
        [
            TensorEdge("seq", "v1_out", weight=0.0+0.0j,
                       semantic_role="score", src_port="score_out", dst_port="score_in"),
            TensorEdge("seq", "v2_out", weight=0.0+0.0j,
                       semantic_role="score", src_port="score_out", dst_port="score_in"),
        ] + list(voice_solver.edges),
        sample_rate=SR,
    )

    solver.dispatch_before_start()

    v1_atoms = bank.peek("seq_v1_out_atoms")
    v2_atoms = bank.peek("seq_v2_out_atoms")

    assert len(v1_atoms) == 2, f"v1 should have 2 melody atoms, got {len(v1_atoms)}"
    assert len(v2_atoms) == 2, f"v2 should have 2 bass atoms, got {len(v2_atoms)}"

    v1_hzs = sorted(a.fundamental_hz for a in v1_atoms)
    v2_hzs = sorted(a.fundamental_hz for a in v2_atoms)

    assert abs(v1_hzs[0] - MEL_HZ1) < 1e-4, f"v1 got wrong note: {v1_hzs}"
    assert abs(v1_hzs[1] - MEL_HZ2) < 1e-4, f"v1 got wrong note: {v1_hzs}"
    assert abs(v2_hzs[0] - BAS_HZ1) < 1e-4, f"v2 got wrong note: {v2_hzs}"
    assert abs(v2_hzs[1] - BAS_HZ2) < 1e-4, f"v2 got wrong note: {v2_hzs}"

    # No cross-contamination: melody Hz must not appear in bass atoms and vice versa
    all_v1_hz = {round(a.fundamental_hz, 2) for a in v1_atoms}
    all_v2_hz = {round(a.fundamental_hz, 2) for a in v2_atoms}
    assert not all_v1_hz & all_v2_hz, f"Cross-contamination: {all_v1_hz & all_v2_hz}"


def test_slice_score_for_consumer_group_filter() -> None:
    """slice_score_for_consumer masks events by PARAM_PATTERN group index."""
    score = empty_score_tensor(1, 4)
    for ei, (hz, grp) in enumerate([(261.63, 0), (329.63, 0), (130.81, 1), (196.00, 1)]):
        score.labels[0, ei]                 = EVENT_NOTE
        score.mask[0, ei]                   = True
        score.params[0, ei, PARAM_HZ]       = hz
        score.params[0, ei, PARAM_PATTERN]  = float(grp)
        score.params[0, ei, PARAM_START]    = 0.0
        score.params[0, ei, PARAM_DURATION] = 0.1
    sparse = sparse_score_tensor_from_score(score)

    # Filter to group 0 only
    filtered = slice_score_for_consumer(sparse, wanted_groups=frozenset([0]))
    jobs = envelope_jobs_from_sparse_score(filtered, sample_rate=SR, release_tail_s=0.02)
    assert jobs.job_count == 2
    got_hz = sorted(float(jobs.fundamental_hz[i].item()) for i in range(jobs.job_count))
    assert abs(got_hz[0] - 261.63) < 1e-4
    assert abs(got_hz[1] - 329.63) < 1e-4

    # Filter to group 1 only
    filtered2 = slice_score_for_consumer(sparse, wanted_groups=frozenset([1]))
    jobs2 = envelope_jobs_from_sparse_score(filtered2, sample_rate=SR, release_tail_s=0.02)
    assert jobs2.job_count == 2
    got_hz2 = sorted(float(jobs2.fundamental_hz[i].item()) for i in range(jobs2.job_count))
    assert abs(got_hz2[0] - 130.81) < 1e-4
    assert abs(got_hz2[1] - 196.00) < 1e-4

    # No filter → all 4 events
    unfiltered = slice_score_for_consumer(sparse)
    jobs3 = envelope_jobs_from_sparse_score(unfiltered, sample_rate=SR, release_tail_s=0.02)
    assert jobs3.job_count == 4


if __name__ == "__main__":
    _run_direct_demo()
