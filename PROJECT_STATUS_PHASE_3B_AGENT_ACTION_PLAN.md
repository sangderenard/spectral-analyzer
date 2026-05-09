# Phase 3B Agent Kickoff And Three-Scale Action Plan

Date: May 9, 2026  
Repository: spectral-analyzer  
Companion source: PROJECT_STATUS_PHASE_3B_COMPANION_REVISION.md

## 1) Introduction For The Incoming Agent
You are entering a project that already has a meaningful Phase 3B foundation for sensor/film registry and tensor packing, but runtime integration into exposure execution is incomplete. Your mission is to close the loop from registered sensor/film data to physically interpretable per-pixel spectral outputs, with calibration scenes that make correctness visible immediately.

### Working definition of success
A run is successful when it can:
1. Select sensor/film pairs by slot.
2. Push slot data through runtime interfaces without schema ambiguity.
3. Accumulate endpoint-derived photons and electrons per pixel.
4. Compute SNR from explicit noise terms.
5. Validate expected behavior in calibration scenes, including prism-spectrum backplate comparisons.

### Primary constraints
1. Keep dataflow explicit and reproducible.
2. Prefer deterministic tests over qualitative judgments.
3. Treat code constants as source of truth until replaced by cited calibrations.
4. Avoid hidden assumptions in slot indexing, layer semantics, and tensor contracts.

---

## 2) Agent Instruction Block (Use As Tasking Header)

Copy this section into the operating prompt for the next implementation agent.

### Agent directive
You are tasked with implementing Phase 3C through first-pass 3E in a physically traceable manner.

Do this in order:
1. Freeze API contracts between Python and C++ for sensor/film payloads and slot metadata.
2. Integrate slot-aware sensor/film tensors into ExposureSession.
3. Replace placeholder sensor integral with endpoint-derived accumulation for slot 0.
4. Extend to multi-slot accumulation with isolation checks.
5. Add calibration scene suite including grid, markers, eye-chart frequency patterns, and prism-spectrum backplate comparison.
6. Produce reproducibility artifacts and acceptance evidence.

Required outputs:
1. Code changes with deterministic tests.
2. A runbook with exact commands and expected artifacts.
3. A short verification report listing pass/fail against acceptance criteria.

Definition of done:
1. No placeholder sensor integral path in primary execution route.
2. At least one slot and one multi-slot run produce valid photons/electrons/SNR artifacts.
3. Calibration scenes produce measurable comparisons against expected outputs.

---

## 3) Strict Task Board (Owners, Tests, Duration)

### Legend
- Owner suggestions are role-based; map to actual names.
- Estimate format: engineering days.
- Priority: P0 blocking, P1 high, P2 follow-up.

| ID | Priority | Task | Owner | Estimate | Dependencies | Acceptance Test |
|---|---|---|---|---:|---|---|
| T1 | P0 | Freeze Python-C++ sensor/film payload schema and slot metadata contract | Integration Lead + Kernel Lead | 0.5 | none | Contract note approved; payload keys/shapes stable in one integration test |
| T2 | P0 | Add sensor_film_slots and tensor loading to ExposureSession init path | Python Runtime | 0.5 | T1 | Session startup logs active slots and ids; no runtime errors |
| T3 | P0 | Implement binding helper from session to tracer payload upload API | Python Runtime + C++ Bindings | 1.0 | T1,T2 | Payload arrives in tracer with expected row count and dtype |
| T4 | P0 | Replace placeholder slot-0 sensor integral with endpoint-derived accumulation | Python Runtime + Integrator | 1.5 | T2,T3 | One-slot run outputs photons/electrons/SNR maps with non-trivial spatial variation |
| T5 | P0 | Add explicit SNR and noise model terms to saved metadata | Science + Runtime | 0.5 | T4 | Saved metadata includes read noise, dark current, shot-noise model references |
| T6 | P1 | Extend accumulation from slot 0 to active slots 0-7 | Runtime + Integrator | 1.0 | T4 | Multi-slot run writes per-slot artifacts and shows slot isolation |
| T7 | P1 | Implement slot isolation regression test (cross-slot bleed guard) | QA/Validation | 0.5 | T6 | Deliberate slot perturbation affects only targeted slot |
| T8 | P1 | Draft and implement calibration Scene A: fine grid and marker lattice | Validation + Scene Authoring | 1.0 | T4 | Distortion and keypoint residual reports generated |
| T9 | P1 | Implement calibration Scene B: eye-chart and radial frequency target | Validation + Scene Authoring | 1.0 | T4 | Resolution trend report generated vs expected frequency response |
| T10 | P0 | Implement calibration Scene C: prism colinear beam with expected-spectrum backplate | Validation + Rendering | 2.0 | T4 | Peak position and spread-width error metrics produced |
| T11 | P1 | Add deterministic run harness (seed, scene hash, artifact manifest) | Reliability/Tooling | 1.0 | T4,T8,T9,T10 | Re-running same config reproduces artifact checksums or bounded diffs |
| T12 | P1 | Add first-pass dashboard/HUD labels for slot metrics | Runtime/UI | 0.75 | T6 | Slot labels and summary metrics visible and match artifact metadata |
| T13 | P2 | Begin shader-side multi-slot vectorized unpack parity checks | Kernel/Shader | 1.5 | T1,T6 | Known-value probe confirms layout and unpack parity |
| T14 | P2 | Performance baseline and budget report (1-slot vs N-slot) | Perf Lead | 1.0 | T6,T13 | Report includes timing table and memory envelope |

### Suggested ownership mapping
1. Integration Lead: API contracts, sequencing, review gate.
2. Python Runtime: ExposureSession and artifact generation.
3. C++/Kernel: binding surfaces and shader parity.
4. Science/Validation: noise model correctness and calibration criteria.
5. Reliability: reproducibility harness and regression automation.

---

## 4) Three-Scale Action Plan

## Immediate Scale (0-3 days)
Goal: remove ambiguity and get one physically grounded path working now.

### Objectives
1. Finalize payload schema and slot metadata contract.
2. Wire ExposureSession to sensor/film tensors.
3. Replace placeholder slot-0 sensor integral with endpoint-derived accumulation.
4. Generate first valid photons/electrons/SNR artifacts.

### Immediate deliverables
1. Merged tasks: T1, T2, T3, T4, T5.
2. One command that runs deterministic one-slot test and writes artifacts.
3. One verification note showing formulas, constants, and artifact paths.

### Immediate risks and mitigation
1. Risk: payload shape mismatch.
Mitigation: enforce schema assertions at both Python and C++ boundaries.
2. Risk: endpoint mapping uncertainty.
Mitigation: instrument with debug projections and pixel-hit sanity histograms.

---

## Horizon Scale (1-3 weeks)
Goal: make the system robust, comparable, and calibration-ready for iteration.

### Objectives
1. Expand to multi-slot accumulation and isolation guarantees.
2. Introduce calibration scene suite (grid, markers, eye chart, prism backplate).
3. Add reproducibility harness and baseline regression checks.
4. Add first operational HUD/reporting for slot outputs.

### Horizon deliverables
1. Merged tasks: T6, T7, T8, T9, T10, T11, T12.
2. Calibration report package with numeric pass/fail against tolerances.
3. Repeatable CI-friendly runbook with scene manifests.

### Horizon risks and mitigation
1. Risk: calibration scenes too qualitative.
Mitigation: require numeric error metrics and thresholded pass/fail.
2. Risk: cross-slot contamination hidden by visualization.
Mitigation: explicit synthetic perturbation tests and statistical isolation checks.

---

## Total Project Scale (1-3 months)
Goal: evolve from functional pipeline to trusted scientific imaging platform.

### Objectives
1. Complete shader parity and multi-slot performance optimization.
2. Establish validated sensor/film parameter provenance with citations.
3. Expand calibration taxonomy and long-run regression corpus.
4. Formalize release gates around reproducibility, scientific validity, and performance.

### Total project deliverables
1. Merged tasks: T13, T14 plus parameter-provenance and governance milestones.
2. Quarterly benchmark suite with trend tracking.
3. Team-level playbook for adding new sensors/films and calibration scenes safely.

### Total project governance gates
1. Correctness gate: parity checks and calibration pass rates.
2. Reproducibility gate: deterministic rerun agreement.
3. Performance gate: memory and runtime envelope adherence.
4. Documentation gate: every new model parameter has origin, units, and test coverage.

---

## 5) Calibration Scene Design Notes (Requested Prism/Backplate Program)

### Prism spectral comparator scene blueprint
1. Build a corridor scene with controlled colinear source beam.
2. Place prism at fixed transform and define expected dispersion path on rear plate.
3. Use backplate texture that encodes expected wavelength track and tolerance bands.
4. Render measured spectral projection and compute:
- peak position delta
- dispersion spread error
- channel energy ratio error
5. Output overlay image and machine-readable metric JSON.

### Computer-science framing for scene set
1. Fine grids and marker lattices validate geometric transforms and sampling behavior.
2. Eye-chart/radial patterns validate frequency response and resolution falloff.
3. Prism backplate validates spectral ordering and dispersion plausibility.

---

## 6) Execution Checklist For Day 1 Start

1. Read companion and this action plan fully.
2. Confirm schema contract decision in writing.
3. Implement T2 and T3 behind assertions.
4. Replace placeholder path for slot 0 (T4).
5. Run one deterministic scene and verify artifacts.
6. Publish short evidence note with links to artifacts.

---

## 7) Reporting Template For The Agent

Use this exact structure at each checkpoint:
1. What changed.
2. What evidence proves it.
3. What failed or remains unknown.
4. What is next in priority order.

Minimum evidence each report must include:
1. Active slot list and ids.
2. Artifact paths for photons/electrons/SNR.
3. Numeric summary metrics.
4. Seed and scene hash.

---

## 8) Final Note To The Team
The program should optimize for evidence velocity: every implementation step should produce measurable output that either confirms or falsifies assumptions. This is how we move quickly without sacrificing scientific integrity.
