"""Fix patch_to_driver.py module docstring."""
NEW_DOC = '''\
"""patch_to_driver.py \u2014 Bridge: AnalyticPatch \u2192 DriverConfig.

Converts the high-level AnalyticPatch (voices, performers, routing) into the
flat tensor config consumed by ``performer_engine.py``.

Typical usage::

    from patch_to_driver import build_driver_config
    from performer_engine import init_driver_state, multi_level_driver_step

    cfg, performers, driver_list = build_driver_config(patch, performers, device, sr)
    state = init_driver_state(cfg)
    # per-chunk synthesis loop:
    driver_out, voice_out, state = multi_level_driver_step(cfg, state, chunk_T, sr)

Driver slots
------------
One slot is created for every (performer, voice) pair where the voice key
appears in ``PerformerPlacement.source_voice_keys``.  Muted voices are skipped.
When a performer has an empty ``source_voice_keys`` list it is expanded to all
non-muted voices (keeps compatibility with simple single-performer patches that
pre-date the placement solver).

If the *performers* list is empty (placement solver has not been run), a single
synthetic performer is created for every non-muted voice, positioned at the
origin with no delays or gain offsets.

Ownership
---------
Performers own instruments 1:1.  Each driver slot bakes in the performer\'s
timing humanization (geometric_delay_ms, humanization_ms) into
pre_delay_samples.  Performers are state-machine objects and carry no tensor
representation in DriverConfig.

``instrument_idx`` identifies which instrument (excitation point) each slot
belongs to, used by the sympathetic coupling step.

Frequency resolution
--------------------
Each voice\'s ``note_tracking`` field determines which base frequency is used:

* ``"note"``  \u2014 *note_hz* (the currently sounding note, defaults to ``seq_tonic_hz``)
* ``"root"``  \u2014 ``patch.seq_tonic_hz`` (tonal centre / scale root)
* ``"free"``  \u2014 ``voice.freq_hz`` (voice\'s own absolute frequency)

``voice.semitone_offset`` is applied after tracking resolution in all cases.

Envelope
--------
``voice.active_knots()`` returns ``[[t_frac, v], \u2026]`` with *t_frac* \u2208 [0, 1].
These are converted to absolute seconds by multiplying by *note_duration*
(``patch.duration``).
"""
'''

with open("patch_to_driver.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find start and end of existing docstring
start = content.index('"""patch_to_driver.py')
end   = content.index('"""\n\nfrom __future__') + len('"""\n')
print(f"Replacing chars {start}..{end}")
content = content[:start] + NEW_DOC + content[end:]

with open("patch_to_driver.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done.")
