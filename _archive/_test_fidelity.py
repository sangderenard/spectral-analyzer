"""Quick smoke test for fidelity_curve and variable-BPO CQT."""
import numpy as np
import torch
import torch_cqt_new as tcq

SR = 22050
HOP = 512
FMIN = 27.5
FMAX = 11025.0
BPO = 48

# ── 1. fidelity_curve with constant params ──
fc = tcq.fidelity_curve(SR, HOP, FMIN, FMAX, BPO)
print(f"Constant BPO={BPO}:")
print(f"  bins: {len(fc['freqs'])}")
print(f"  Δt range: [{fc['delta_t'].min():.6f}, {fc['delta_t'].max():.6f}] s")
print(f"  Δf range: [{fc['delta_f'].min():.2f}, {fc['delta_f'].max():.2f}] Hz")
print(f"  uncertainty range: [{fc['uncertainty'].min():.4f}, {fc['uncertainty'].max():.4f}]")
print(f"  Gabor limit: {fc['gabor_limit']:.6f}")
print(f"  bpo_per_octave: {fc['bpo_per_octave']}")
print(f"  hop_per_octave: {fc['hop_per_octave']}")

# ── 2. fidelity_curve with variable BPO function ──
def my_bpo(oct_idx: int, base: int) -> int:
    """More bins in top octaves (fine detail), fewer in bottom (broad)."""
    return max(12, base >> oct_idx)

fc2 = tcq.fidelity_curve(SR, HOP, FMIN, FMAX, BPO, bpo_func=my_bpo)
print(f"\nVariable BPO (halving per octave):")
print(f"  bins: {len(fc2['freqs'])}")
print(f"  bpo_per_octave: {fc2['bpo_per_octave']}")
print(f"  hop_per_octave: {fc2['hop_per_octave']}")
print(f"  uncertainty range: [{fc2['uncertainty'].min():.4f}, {fc2['uncertainty'].max():.4f}]")

# ── 3. fidelity_curve with variable hop function ──
def my_hop(oct_idx: int, decimated_hop: int) -> int:
    """Double the hop for bottom octaves (narrower bins, can afford it)."""
    if oct_idx >= 4:
        return decimated_hop * 2
    return decimated_hop

fc3 = tcq.fidelity_curve(SR, HOP, FMIN, FMAX, BPO, hop_func=my_hop)
print(f"\nVariable hop (doubled for bottom 4 octaves):")
print(f"  hop_per_octave: {fc3['hop_per_octave']}")
print(f"  uncertainty range: [{fc3['uncertainty'].min():.4f}, {fc3['uncertainty'].max():.4f}]")

# ── 4. fidelity_curve with both ──
fc4 = tcq.fidelity_curve(SR, HOP, FMIN, FMAX, BPO, bpo_func=my_bpo, hop_func=my_hop)
print(f"\nBoth variable:")
print(f"  bins: {len(fc4['freqs'])}")
print(f"  bpo_per_octave: {fc4['bpo_per_octave']}")
print(f"  hop_per_octave: {fc4['hop_per_octave']}")

# ── 5. CQT with variable BPO (proof-of-concept forward pass) ──
duration = 1.0
t = np.arange(int(SR * duration)) / SR
sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
y = torch.from_numpy(sig)

N_BINS = BPO * 4  # 4 octaves worth at base BPO

print(f"\nCQT forward (constant BPO={BPO}, {N_BINS} bins):")
C, freqs = tcq.cqt(y, SR, hop_length=HOP, fmin=FMIN,
                     n_bins=N_BINS, bins_per_octave=BPO,
                     device=torch.device("cpu"))
print(f"  C shape: {C.shape}, freqs: {freqs.shape}")
print(f"  freq range: [{freqs[0].item():.2f}, {freqs[-1].item():.2f}] Hz")

print(f"\nCQT forward (variable BPO):")
C2, freqs2 = tcq.cqt(y, SR, hop_length=HOP, fmin=FMIN,
                       n_bins=N_BINS, bins_per_octave=BPO,
                       bpo_func=my_bpo,
                       device=torch.device("cpu"))
print(f"  C shape: {C2.shape}, freqs: {freqs2.shape}")
print(f"  freq range: [{freqs2[0].item():.2f}, {freqs2[-1].item():.2f}] Hz")

print("\nAll tests passed.")
