"""Second rename pass: remaining partial/partials in PartialPanel and PatchPanel"""
with open("analytic_driver.py", "r", encoding="utf-8") as f:
    src = f.read()

replacements = [
    # Module docstring comments
    ('  left panel  (280 px) — voice list (partials + LFOs)',
     '  left panel  (280 px) — voice list (voices + LFOs)'),
    ('  right panel (280 px)  — parameters for the selected partial',
     '  right panel (280 px)  — parameters for the selected voice'),
    # EditorCanvas cache comment
    ('        # Cache of active partial key (drives overlay rendering)',
     '        # Cache of active voice key (drives overlay rendering)'),
    ('        # --- find active partial ---',
     '        # --- find active voice ---'),
    # PatchPanel docstring + attr
    ('    """Left panel: scrollable list of partials + LFOs."""',
     '    """Left panel: scrollable list of voices + LFOs."""'),
    ('        self.on_add_partial: Any = None',
     '        self.on_add_voice:   Any = None'),
    ('        for p in self._patch.partials:',
     '        for p in self._patch.voices:'),
    ('        p_lbl = font.render("+ Partial", True, (160, 200, 240))',
     '        p_lbl = font.render("+ Voice", True, (160, 200, 240))'),
    ('                        for p in self._patch.partials:',
     '                        for p in self._patch.voices:'),
    ('                    if self.on_add_partial:',
     '                    if self.on_add_voice:'),
    ('                        self.on_add_partial()',
     '                        self.on_add_voice()'),
    # PartialPanel section comment + class docstring
    ('# PartialPanel — right panel: parameters for the selected partial',
     '# VoicePanel — right panel: parameters for the selected voice'),
    ('class PartialPanel(Panel):\n    """Right panel: editable parameters for the active partial/LFO."""',
     'class PartialPanel(Panel):\n    """Right panel: editable parameters for the active voice/LFO."""'),
    # PartialPanel.render() – voice lookup + all partial.xxx
    ('        partial = next((p for p in self._patch.partials if p.key == self._active_key), None)',
     '        voice = next((p for p in self._patch.voices if p.key == self._active_key), None)'),
    ('        obj = partial or lfo',
     '        obj = voice or lfo'),
    ('        if partial:',
     '        if voice:'),
    ('                                 partial.freq_hz, 1.0, 20000.0, ".1f", True)',
     '                                 voice.freq_hz, 1.0, 20000.0, ".1f", True)'),
    ('                                 partial.amplitude, 0.0, 4.0, ".3f")',
     '                                 voice.amplitude, 0.0, 4.0, ".3f")'),
    ('                                 partial.phase_origin, -math.pi, math.pi, ".3f")',
     '                                 voice.phase_origin, -math.pi, math.pi, ".3f")'),
    ('            if partial.env_type == "adsr":', '            if voice.env_type == "adsr":'),
    ('                                     partial.adsr.attack,  0.001, 2.0, ".4f", True)',
     '                                     voice.adsr.attack,  0.001, 2.0, ".4f", True)'),
    ('                                     partial.adsr.decay,   0.001, 2.0, ".4f", True)',
     '                                     voice.adsr.decay,   0.001, 2.0, ".4f", True)'),
    ('                                     partial.adsr.sustain, 0.0,   1.0, ".3f")',
     '                                     voice.adsr.sustain, 0.0,   1.0, ".3f")'),
    ('                                     partial.adsr.release, 0.001, 4.0, ".4f", True)',
     '                                     voice.adsr.release, 0.001, 4.0, ".4f", True)'),
    ('                                     partial.adsr.peak,    0.0,   2.0, ".3f")',
     '                                     voice.adsr.peak,    0.0,   2.0, ".3f")'),
    ('                                 partial.chirp.f_delta_start, -5000.0, 5000.0, ".1f")',
     '                                 voice.chirp.f_delta_start, -5000.0, 5000.0, ".1f")'),
    ('                                 partial.chirp.f_delta_end,   -5000.0, 5000.0, ".1f")',
     '                                 voice.chirp.f_delta_end,   -5000.0, 5000.0, ".1f")'),
    ('                                 partial.chirp.tau, 0.01, 10.0, ".3f", True)',
     '                                 voice.chirp.tau, 0.01, 10.0, ".3f", True)'),
    ('            fm_depth = partial.fm.depth_hz if partial.fm else 0.0',
     '            fm_depth = voice.fm.depth_hz if voice.fm else 0.0'),
    ('            am_depth = partial.am.depth_amp if partial.am else 0.0',
     '            am_depth = voice.am.depth_amp if voice.am else 0.0'),
    ('                                 partial.loop_start, 0.0, 0.99, ".3f")',
     '                                 voice.loop_start, 0.0, 0.99, ".3f")'),
    ('                                 partial.loop_end, 0.01, 1.0, ".3f")',
     '                                 voice.loop_end, 0.01, 1.0, ".3f")'),
    ('        name = (partial.label if partial else lfo.label) if obj else "?"',
     '        name = (voice.label if voice else lfo.label) if obj else "?"'),
    # PartialPanel._set_slider_val() – voice lookup + all partial. assignments
    ('        partial = next((p for p in self._patch.partials if p.key == self._active_key), None)',
     '        voice = next((p for p in self._patch.voices if p.key == self._active_key), None)'),
    ('        if partial:',
     '        if voice:'),
    ('            if key == "freq_hz":         partial.freq_hz          = val',
     '            if key == "freq_hz":         voice.freq_hz            = val'),
    ('            elif key == "amplitude":     voice.amplitude          = val',  # already done
     '            elif key == "amplitude":     voice.amplitude          = val'),
    ('            elif key == "phase_origin":  partial.phase_origin     = val',
     '            elif key == "phase_origin":  voice.phase_origin       = val'),
    ('            elif key == "adsr.attack":   partial.adsr.attack      = val',
     '            elif key == "adsr.attack":   voice.adsr.attack        = val'),
    ('            elif key == "adsr.decay":    partial.adsr.decay       = val',
     '            elif key == "adsr.decay":    voice.adsr.decay         = val'),
    ('            elif key == "adsr.sustain":  partial.adsr.sustain     = val',
     '            elif key == "adsr.sustain":  voice.adsr.sustain       = val'),
    ('            elif key == "adsr.release":  partial.adsr.release     = val',
     '            elif key == "adsr.release":  voice.adsr.release       = val'),
    ('            elif key == "adsr.peak":     partial.adsr.peak        = val',
     '            elif key == "adsr.peak":     voice.adsr.peak          = val'),
    ('                partial.chirp.f_delta_start = val',
     '                voice.chirp.f_delta_start = val'),
    ('                partial.chirp.f_delta_end   = val',
     '                voice.chirp.f_delta_end   = val'),
    ('            elif key == "chirp.tau":     partial.chirp.tau        = val',
     '            elif key == "chirp.tau":     voice.chirp.tau          = val'),
    ('                if partial.fm is None:\n                    partial.fm = ModRouting()\n                partial.fm.depth_hz = val',
     '                if voice.fm is None:\n                    voice.fm = ModRouting()\n                voice.fm.depth_hz = val'),
    ('                if partial.am is None:\n                    partial.am = ModRouting()\n                partial.am.depth_amp = val',
     '                if voice.am is None:\n                    voice.am = ModRouting()\n                voice.am.depth_amp = val'),
    ('            elif key == "loop_start":    partial.loop_start       = val',
     '            elif key == "loop_start":    voice.loop_start         = val'),
    ('            elif key == "loop_end":      partial.loop_end         = val',
     '            elif key == "loop_end":      voice.loop_end           = val'),
    # AnalyticDriverViewer init: patch.partials[0].key
    ('        self.active_key: str = (self.patch.partials[0].key\n                                if self.patch.partials else "")',
     '        self.active_key: str = (self.patch.voices[0].key\n                                if self.patch.voices else "")'),
    # Wiring: on_add_partial callback
    ('        self.patch_panel.on_add_voice    = self._on_add_voice',  # already renamed, skip
     '        self.patch_panel.on_add_voice    = self._on_add_voice'),
]

count = 0
misses = []
for old, new in replacements:
    if old != new and old in src:
        src = src.replace(old, new, 1)
        count += 1
    elif old != new:
        misses.append(old[:70])

with open("analytic_driver.py", "w", encoding="utf-8") as f:
    f.write(src)

print(f"Applied: {count}")
if misses:
    print("MISSES:")
    for m in misses:
        print(" ", repr(m))
