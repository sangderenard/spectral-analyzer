# Native camera interval observer v1

This is an explicit diagnostic side channel of the existing native sensor continuation, not a render, a field estimator, a confidence score, or an image overlay. It does not evaluate the field for the screen. Ordinary callers pass no observer and retain the exact original shader source.

`CameraIntervalObserver` borrows caller-owned SSBO storage at binding 22. The caller must initialize/reset its two-word header before dispatch, read back after completion, and retain the exact scene revision, camera revision, shader identity, sample seed and dispatch identity with the bytes. Retained cooperative loan replay is rejected because recording both capture and resumed execution would duplicate observations. Forward recorder binding 22 is a separate program; owners must rebind each program correctly.

Header uint32 words: attempted rows, overflow flag. Storage is at least `8 + capacity * 512` bytes. Capacity exhaustion is explicit; truncated observations never certify absence. Rows are 128 raw uint32 words; floats use IEEE bit representation. Signed IDs retain two's-complement representation.

- 0–3: global sensor pixel x/y, sample index within dispatch, bounce.
- 4–7: canonical geometry hit ID, entry native medium, band count, fixed3 mode.
- 8–10: entry origin. 11: actual hit distance, NaN on miss.
- 12–14: actual entry direction.
- 16–23: entry throughput per active band, before canonical Beer attenuation.
- 24–31: actual band frequencies in Hz.
- 32: observed outcome: 0 ordinary surface/not observed as boundary; 1 accepted boundary; 2 refused boundary; 3 geometry miss; 4 canonical throughput cutoff; 5 invalid entry medium. This is not a complete termination taxonomy.
- 33: boundary output medium, entry medium when no boundary outcome.
- 34–36: boundary output direction, meaningful only for boundary outcomes.
- 37: original boundary camera factor, meaningful only for boundary outcomes.
- 38: boundary frequency actually used by native continuation.
- 39: active region count, zero for non-region source.
- 40: active wavelength bitmask. In original shared-path mode, all slots below band count are active. In separate-field mode, exactly the current wavelength bit is active; fixed3 has three slots, not eight.
- 48–79: complete entry active-region bitset.
- 80–111: complete post-boundary active-region bitset, meaningful only for boundary outcomes.
- Remaining words reserved zero.

Geometry IDs resolve through the authoritative native scene-domain table, not browser material IDs. Native medium alone is insufficient for additive dyes; preserve the full bitset. Refused/invalid/unvisited and truncated observations are unknown, never physical black.

Known limitation retained deliberately: canonical sensor uses band 0 to choose a shared refracted path while attenuating all active bands separately. The observer reports that actual path honestly and draws no additional RNG. It adds no scattering or emission to clear absorbing glass and does not reuse the air energy-opacity marcher.

Implemented only when `field_continuation=True`, separately from observation: an explicit field-continuation variant would run each actual spectral band through the same existing geometry/region/boundary functions with that band's frequency, retain its own direction/medium/active-region state and apply the existing Beer law. This changes the shared-path spectral estimator and must be reviewed separately. Interior forward arrivals must be evaluated as transmitted/reflected radiance with matching material and angular availability, not added as absorption-induced glow. Availability is not inferred from energy or direction coherence.


## Opt-in spectral specialization and verification

`studio_field_continuation.separate_field_continuation` scopes one unchanged native continuation per actual band. The aperture/jitter camera ray is shared. Each wavelength restores that ray and independently calls `studio_origin_medium`, resetting canonical active-region state. Boundary frequency uses the active wavelength. Original RNG calls consume one continuous stream in wavelength order; there is no new seeding or random draw. Each spectral slot receives one full estimator per camera sample, with no division by band count. Film filtering and sample accumulation remain outside the wavelength loop. This is an optical estimator correction, distinct from neutral observation, not a proven field-response estimator.

Ten CPU/source/offline compiler tests pass; full native plain and region shader variants with observer compile in glslang. No GL context, render, exposure, or screenshot was used. Runtime numerical equivalence or correctness is not claimed from those checks.

Raw interval evidence is bounded and optional, not the permanent screen product: 128�72�3�6�512 = 84,934,656 bytes (81 MiB) before headers; 1920�1080 with the same paths is 19,110,297,600 bytes. A full interval stream cannot silently truncate to fit a package. Overflow is explicit unknown. Canonical field/material response and its actual availability should be retained for screen consumption with scene/camera/domain identity; a pre-rendered image is not an acceptable substitute.

## Existing evidence extension: actual response copies

`studio_sensor_evidence.StudioResponseEvidence` owns optional bounded streaming storage through `SensorAccumulator(light_field_information=...)`. It extends the existing evidence side channel without changing leaf-integrated spectra at bindings13/14. Ordinary callers do not create it. It does not automatically enable separate-band optical correction.

Native sensor binding22 holds interval rows. Binding23 is reserved for the producer's original scalar-deposit visitation field. Binding24 holds copied response rows, aligned one-to-one with interval rows. Each response row is128uint32: slots0–95 contain twelve channels × eight capacity slots of exact float RHS bits;96–107 hold per-channel evaluated-presence masks;108–119 hold two-bit source states per wavelength (0unknown,1known absent,3known present). Slots120–127 are reserved for actual scalar-query flags. Presence is never an availability alpha.

Channels: air-volume addition, area-emission addition, legacy-emission addition, surface-field addition, direct-light addition, native PBR NEE addition, Beer multiplier, boundary multiplier, legacy-transmission multiplier, surface-survival multiplier, native PBR multiplier, air multiplier. Only the actual selected canonical path's evaluated expressions produce presence. Scalar field/NEE/air source availability remains unknown until its original query supplies a valid outcome. Material/emission expressions have their own evaluated-input status, which is not a certificate of complete radiance.

Each original RHS is evaluated once through an identity-return copy function, then used by the original addition/multiplication. No path, BRDF, Fresnel, Beer, normalization, color transform or light is reconstructed from a package. The original native sensor-to-linear color seam must be used when packaging display-ready material components; spectral rows retain their exact native frequencies and values. No auto-exposure or display tone mapping belongs in these rows.

The stream budgets GPU interval+response storage plus both CPU readbacks, including caller-declared other reserved memory. Capacity follows actual active pixels×sample count×path-band count×(enforced bounce bound+1), not empirical sample confidence. Existing strip size decreases if necessary; minimum supported granularity must fit or dispatch is rejected. Caller consumes/spools each batch synchronously under an explicit retention policy; no full-history retention is automatic. Overflow raises incomplete and does not deliver a misleading successful batch. Cooperative replay remains unsupported.

This is not yet the final screen product: copied spectral components still need exact existing visibility/stage lineage and canonical producer color conversion before delivery to the three secondary passes. The browser must not re-evaluate optical/material responses from these raw rows.

## Latest provenance fields and verification

Interval words112–114 retain the existing direct-light source index, branch outcome, and shadow-hit index. Outcomes:0not sampled,1no source,2source selected without classified result,3receiver back-facing,4emitter back-facing,5occluded,6reached source. These are copied at the original tests, without a second shadow ray.

Interval115 is the original first hit ID;116–118 copy the already-computed original first hit position;119 marks that position present.120–123 copy the original resolved first Tri's material-index bits, material flags, and positive/negative boundary tag bits. Subsequent bounce rows retain these unchanged instead of pretending the current hit is the visible recipient. Exact spheres/rod partitions need their existing primitive provenance too; this is not yet a proven browser-face mapping.

Original scalar flags are borrowed from prebuilt state `scalar_visit_buffer` at23 and queried at the existing scalar lookup position/band:1evaluated deposit,2positive quantized tozero,4clipped,8invalid. Response slots120–127 preserve flags verbatim. No deposit gives absent information, not black. Flags describe that scalar query only, not angular/material completeness. Native NEE copies use the actual existing source/occlusion branch outcome rather than energy as availability.

Fifteen CPU/offline compiler tests now pass, including actual plain/region, legacy/native opaque-PBR shader assembly with response copies and scalar-query provenance. A CPU fake-GL lifecycle test verifies budget/reset/readback/overflow/destruction without opening a context. Runtime visual/transport correctness has not been tested under the current no-render restriction.
