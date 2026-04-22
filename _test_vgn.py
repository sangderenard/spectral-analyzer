"""Smoke test for voice_graph_node.py."""
import torch
from voice_graph_node import (
    VoiceTorchOscillator, MetaVoiceNode, MixerSumNode,
    build_voice_mixer_network, VoiceTrainer,
)

# ── 1. single forward tick ─────────────────────────────────────────
osc = VoiceTorchOscillator(freq_hz=440.0, amplitude=0.5, sample_rate=48000.0, duration=0.1)
osc._sample_idx.fill_(240)  # t = 0.005 s, mid-attack
x_in = torch.tensor(0.0 + 0.0j, dtype=torch.complex128)
sig = osc(x_in)
print(f"single forward tick: dtype={sig.dtype}, abs={sig.abs().item():.4f}")
assert sig.dtype == torch.complex128
assert sig.shape == torch.Size([])

# ── 2. build network + single step ──────────────────────────────────────────
class FakeVoice:
    key = "v1"
    freq_hz = 440.0; amplitude = 1.0; phase_origin = 0.0
    semitone_offset = 0.0; harmonic_brightness = 1.0; harmonic_warp_strength = 0.0
    fm = None; am = None; env_type = "adsr"; loop_enabled = False
    loop_start = 0.1; loop_end = 0.9; emission_mode = "single"
    pre_delay = 0.0; manifold_type = "pure"; harmonic_count = 1
    class adsr:
        attack = 0.005; decay = 0.04; sustain = 0.75; release = 0.08; peak = 1.0
    class chirp:
        chirp_type = "none"; f_delta_start = 0.0; f_delta_end = 0.0
        tau = 0.5; chirp_power = 1.0
    def active_knots(self):
        return [[0, 0], [0.01, 1], [0.1, .75], [.85, .75], [1, 0]]

solver, vnodes, mixer = build_voice_mixer_network(
    [FakeVoice()], sample_rate=48000.0, duration=0.1
)
vnodes["v1"].set_sample(0)
out = solver.step({})
print("solver step keys:", list(out.keys()))
sample0 = out["mix_out"]
print("mix_out[0] =", sample0.item())
assert "mix_out" in out
assert out["mix_out"].dtype == torch.complex128

# ── 3. gradient flows through log_freq ─────────────────────────────────────
solver.reset()
vnodes["v1"].reset()
for p in vnodes["v1"].oscillator.parameters():
    if p.grad is not None:
        p.grad.zero_()

# Use sample 480 (t=0.01 s) which sits at the ADSR attack peak — envelope is non-zero
vnodes["v1"].set_sample(480)
out2 = solver.step({})
pred = out2["mix_out"]
ref = torch.tensor(0.3 + 0.2j, dtype=torch.complex128)
loss = ((pred - ref).abs() ** 2)
loss.backward()
g = vnodes["v1"].oscillator.log_freq.grad
print("log_freq grad =", g)
assert g is not None and g.abs().item() > 0, "log_freq must have non-zero gradient"

# ── 4. knobs() returns non-empty list ───────────────────────────────────────
knobs = MetaVoiceNode.knobs()
print(f"knobs count = {len(knobs)}")
assert len(knobs) > 0

# ── 5. VoiceTrainer short run ───────────────────────────────────────────────
import torch.optim as optim
all_params = list(vnodes["v1"].oscillator.parameters())
opt = optim.Adam(all_params, lr=1e-3)
trainer = VoiceTrainer(solver, vnodes, opt, target_key="mix_out", accum_n=16)
sr = 48000.0
dur = 0.01  # 10 ms
N = int(sr * dur)
ref_wave = torch.randn(N, dtype=torch.complex128) * 0.1
losses = trainer.fit(ref_wave, n_steps=1, verbose=True)
print(f"trainer losses: {losses[:3]}")
assert len(losses) > 0
assert all(isinstance(l, float) for l in losses)

print("\nALL CHECKS PASSED")
