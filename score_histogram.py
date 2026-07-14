#!/usr/bin/env python3
"""
score_histogram.py — eyeball the batch_score.comp.glsl output.

Reads scores.bin (appended per tile by batch_score_host.cpp.inc):
  repeated records of
    header : 4x uint32  = [n_cbatch, n_lbatch, tx, ty]
    body   : n_cbatch*n_lbatch*4 float32 = [A, B, C, S] per (c,b) unit

Prints, for each signal:
  - range / mean / median / the fat tail (p90, p99)
  - an ASCII histogram (log-count) so you can see how many units are "nothing"
  - a now/idle split sweep: at each percentile cut on S (combined), how much
    work runs NOW vs falls to the never-discarded idle pile, and what fraction
    of *total predicted energy* (sum of A) the NOW set captures.

Usage:
  python score_histogram.py [scores.bin]

No deps beyond numpy.
"""
import sys, struct
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "scores.bin"

def load(path):
    rows = []
    with open(path, "rb") as f:
        blob = f.read()
    off, n = 0, len(blob)
    tiles = 0
    while off + 16 <= n:
        n_cb, n_lb, tx, ty = struct.unpack_from("<4I", blob, off); off += 16
        cnt = n_cb * n_lb * 4
        need = cnt * 4
        if off + need > n:
            break
        arr = np.frombuffer(blob, dtype="<f4", count=cnt, offset=off).reshape(-1, 4)
        off += need
        rows.append(arr)
        tiles += 1
    if not rows:
        sys.exit(f"no records parsed from {path}")
    data = np.concatenate(rows, axis=0)
    print(f"loaded {data.shape[0]:,} work units from {tiles} tile record(s) in {path}\n")
    return data

def ascii_hist(x, label, bins=24, log_x=True, width=50):
    x = x[np.isfinite(x)]
    pos = x[x > 0]
    print(f"── {label}  (n={x.size:,}, zero/neg={x.size - pos.size:,}) ──")
    if pos.size == 0:
        print("  (all zero)\n"); return
    if log_x:
        lo, hi = np.log10(pos.min()), np.log10(pos.max() + 1e-30)
        if hi - lo < 1e-9: hi = lo + 1.0
        edges = np.logspace(lo, hi, bins + 1)
    else:
        edges = np.linspace(pos.min(), pos.max(), bins + 1)
    counts, _ = np.histogram(pos, bins=edges)
    cmax = max(counts.max(), 1)
    for i in range(bins):
        bar = "#" * int(round(width * counts[i] / cmax))
        print(f"  {edges[i]:11.3g} | {bar:<{width}} {counts[i]:>8,}")
    print(f"  stats: min={pos.min():.3g} p50={np.median(pos):.3g} "
          f"p90={np.percentile(pos,90):.3g} p99={np.percentile(pos,99):.3g} "
          f"max={pos.max():.3g}\n")

def split_sweep(S, A):
    """Show NOW/IDLE split + captured-energy at percentile cuts on combined S."""
    print("── run-order sweep on combined S (never discarded — this is sort order) ──")
    print(f"  {'cut pct':>8} {'S cut':>11} {'NOW units':>10} {'IDLE units':>11} "
          f"{'NOW % energy':>13}")
    total_E = float(A.sum()) or 1.0
    order = np.argsort(-S)            # descending
    A_sorted = A[order]
    cum_E = np.cumsum(A_sorted) / total_E
    n = S.size
    for pct in (50, 75, 90, 95, 99):
        k = int(round(n * pct / 100.0))         # top-k run NOW
        k = max(1, min(n, k))
        s_cut = float(np.sort(S)[::-1][k - 1])
        now_E = float(cum_E[k - 1])
        print(f"  {pct:>7}% {s_cut:>11.3g} {k:>10,} {n-k:>11,} {now_E*100:>12.1f}%")
    print("  (read: 'top 50% of units by S already carry NOW% of predicted energy')\n")

def main():
    d = load(PATH)
    A, B, C, S = d[:,0], d[:,1], d[:,2], d[:,3]
    ascii_hist(A, "A  predicted contribution (mean beta_c*beta_l*geom)", log_x=True)
    ascii_hist(B, "B  connectable density (fraction)", log_x=False)
    ascii_hist(C, "C  peak geom potential (max geom)", log_x=True)
    ascii_hist(S, "S  combined (wA*A + wB*B + wC*C)", log_x=True)
    # If S is flat (e.g. only wA used and equals A), the sweep still works.
    split_sweep(S if np.ptp(S) > 0 else A, A)
    # Quick "how much is nothing" headline.
    dead = np.mean((A <= 0) & (C <= 0)) * 100.0
    print(f"headline: {dead:.1f}% of work units sampled as pure nothing "
          f"(no connectable pair with positive geom). These are exactly the "
          f"cells the idle pile defers to last.")

if __name__ == "__main__":
    main()
