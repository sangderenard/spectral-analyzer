"""One-shot terminology rename: partial/partials -> voice/voices throughout analytic_driver.py"""
import re

with open("analytic_driver.py", "r", encoding="utf-8") as f:
    src = f.read()

replacements = [
    # AnalyticPatch dataclass field
    ('    partials:   list  = field(default_factory=list)',
     '    voices:    list  = field(default_factory=list)'),
    # to_dict
    ('"partials":   [p.to_dict() for p in self.partials]',
     '"voices":     [v.to_dict() for v in self.voices]'),
    # from_dict
    ('p.partials   = [AnalyticVoice.from_dict(x) for x in d.get("partials", [])]',
     'p.voices     = [AnalyticVoice.from_dict(x) for x in d.get("voices", [])]'),
    # default_patch body
    ('            part = AnalyticVoice()', '            voice = AnalyticVoice()'),
    ('            part.label    = f"P{i+1}"', '            voice.label   = f"V{i+1}"'),
    ('            part.freq_hz  = hz', '            voice.freq_hz = hz'),
    ('            part.color    = col', '            voice.color   = col'),
    ('            p.partials.append(part)', '            p.voices.append(voice)'),
    # Default label string in dataclass
    ('    label:        str   = "Partial"', '    label:        str   = "Voice"'),
    ('        p.label        = d.get("label", "Partial")', '        p.label        = d.get("label", "Voice")'),
    # Function rename
    ('def _synthesize_partial(', 'def _synthesize_voice('),
    # DragState field
    ('    partial_key: str   = ""', '    voice_key:   str   = ""'),
    # _compute_envelope parameter name
    ('def _compute_envelope(partial: AnalyticVoice,', 'def _compute_envelope(voice: AnalyticVoice,'),
    ('    knots = partial.active_knots()', '    knots = voice.active_knots()'),
    # _synthesize_voice body: parameter and all partial. usages
    ('    partial: AnalyticVoice,', '    voice: AnalyticVoice,'),
    ('    f_inst = np.full(n, partial.freq_hz, dtype=np.float64)',
     '    f_inst = np.full(n, voice.freq_hz, dtype=np.float64)'),
    ('    ct = partial.chirp.chirp_type', '    ct = voice.chirp.chirp_type'),
    ('        f_inst += np.linspace(partial.chirp.f_delta_start, partial.chirp.f_delta_end, n)',
     '        f_inst += np.linspace(voice.chirp.f_delta_start, voice.chirp.f_delta_end, n)'),
    ('    elif ct == "exponential" and partial.chirp.tau > 0:',
     '    elif ct == "exponential" and voice.chirp.tau > 0:'),
    ('        decay   = np.exp(-t / partial.chirp.tau)', '        decay   = np.exp(-t / voice.chirp.tau)'),
    ('        f_inst += partial.chirp.f_delta_start * decay + partial.chirp.f_delta_end * (1 - decay)',
     '        f_inst += voice.chirp.f_delta_start * decay + voice.chirp.f_delta_end * (1 - decay)'),
    ('    if partial.fm and partial.fm.source_key:', '    if voice.fm and voice.fm.source_key:'),
    ('        sk = partial.fm.source_key', '        sk = voice.fm.source_key'),
    ('        elif sk in p_map and sk != partial.key:', '        elif sk in p_map and sk != voice.key:'),
    ('        f_inst += partial.fm.depth_hz * mod', '        f_inst += voice.fm.depth_hz * mod'),
    ('    phase = np.cumsum(2.0 * np.pi * f_inst / sr) + partial.phase_origin',
     '    phase = np.cumsum(2.0 * np.pi * f_inst / sr) + voice.phase_origin'),
    ('    amp = np.full(n, partial.amplitude, dtype=np.float64)',
     '    amp = np.full(n, voice.amplitude, dtype=np.float64)'),
    ('    if partial.am and partial.am.source_key:', '    if voice.am and voice.am.source_key:'),
    ('        sk = partial.am.source_key', '        sk = voice.am.source_key'),
    # second occurrence of same pattern (am block has same 'sk')
    ('        elif sk in p_map and sk != partial.key:', '        elif sk in p_map and sk != voice.key:'),
    ('        amp *= (1.0 + partial.am.depth_amp * mod)', '        amp *= (1.0 + voice.am.depth_amp * mod)'),
    # Return complex analytic signal
    ('    return amp * _compute_envelope(partial, n, dur) * np.cos(phase)',
     '    return (amp * _compute_envelope(voice, n, dur) * np.exp(1j * phase))'),
    # _synthesize_patch: p_map, loop, call
    ('    p_map   = {p.key: p for p in patch.partials}\n    n = int(patch.preview_sr * patch.duration)\n    mix = np.zeros(n, dtype=np.float64)\n    for partial in patch.partials:\n        if partial.muted:\n            continue\n        mix += _synthesize_partial(partial, patch, lfo_map, p_map)',
     '    p_map   = {p.key: p for p in patch.voices}\n    n = int(patch.preview_sr * patch.duration)\n    mix = np.zeros(n, dtype=np.complex128)\n    for voice in patch.voices:\n        if voice.muted:\n            continue\n        mix += _synthesize_voice(voice, patch, lfo_map, p_map)'),
    # Project to real at end of _synthesize_patch
    ('    peak = np.max(np.abs(mix))\n    if peak > 1e-9:\n        mix /= peak\n    return mix',
     '    peak = np.max(np.abs(mix))\n    if peak > 1e-9:\n        mix /= peak\n    return mix.real'),
    # rebuild() – patch.partials and partial variable
    ('        partial = next((p for p in patch.partials if p.key == active_key), None)',
     '        voice = next((p for p in patch.voices if p.key == active_key), None)'),
    ('        p_map   = {p.key: p for p in patch.partials}\n        if partial and not partial.muted:',
     '        p_map   = {p.key: p for p in patch.voices}\n        if voice and not voice.muted:'),
    ('            sig = _synthesize_partial(partial, patch, lfo_map, p_map).astype(np.float32)',
     '            sig = _synthesize_voice(voice, patch, lfo_map, p_map).real.astype(np.float32)'),
    ('        if partial:\n            env = _compute_envelope(partial, n, dur).astype(np.float32)',
     '        if voice:\n            env = _compute_envelope(voice, n, dur).astype(np.float32)'),
    ('        if partial:\n            f_base = partial.freq_hz',
     '        if voice:\n            f_base = voice.freq_hz'),
    ('            if partial.chirp.chirp_type == "linear":',
     '            if voice.chirp.chirp_type == "linear":'),
    ('                    partial.chirp.f_delta_start,\n                    partial.chirp.f_delta_end, n)).astype(np.float32)',
     '                    voice.chirp.f_delta_start,\n                    voice.chirp.f_delta_end, n)).astype(np.float32)'),
    ('            elif partial.chirp.chirp_type == "exponential":',
     '            elif voice.chirp.chirp_type == "exponential":'),
    ('                tau = max(partial.chirp.tau, 1e-9)', '                tau = max(voice.chirp.tau, 1e-9)'),
    ('                    + partial.chirp.f_delta_start * dec\n                    + partial.chirp.f_delta_end * (1 - dec)',
     '                    + voice.chirp.f_delta_start * dec\n                    + voice.chirp.f_delta_end * (1 - dec)'),
    ('        col = tuple(partial.color[:3]) if partial else (100, 160, 255)',
     '        col = tuple(voice.color[:3]) if voice else (100, 160, 255)'),
    # render() – active voice lookup + loop/envelope references
    ('        partial = next((p for p in patch.partials if p.key == self.active_key), None)',
     '        voice = next((p for p in patch.voices if p.key == self.active_key), None)'),
    ('        if partial and partial.loop_enabled and self.mode == EditorMode.WAVEFORM:',
     '        if voice and voice.loop_enabled and self.mode == EditorMode.WAVEFORM:'),
    ('            lsx, _ = self.data_to_px(partial.loop_start, 0, win_w, win_h)',
     '            lsx, _ = self.data_to_px(voice.loop_start, 0, win_w, win_h)'),
    ('            lex, _ = self.data_to_px(partial.loop_end,   0, win_w, win_h)',
     '            lex, _ = self.data_to_px(voice.loop_end,   0, win_w, win_h)'),
    ('        if self.mode == EditorMode.ENVELOPE and partial:',
     '        if self.mode == EditorMode.ENVELOPE and voice:'),
    ('            for ki, knot in enumerate(partial.active_knots()):',
     '            for ki, knot in enumerate(voice.active_knots()):'),
    # handle_event() – active voice lookup (first occurrence is in render, second here)
    ('        partial = next((p for p in patch.partials if p.key == self.active_key), None)',
     '        voice = next((p for p in patch.voices if p.key == self.active_key), None)'),
    # loop handle checks in handle_event
    ('            if partial and partial.loop_enabled and self.mode == EditorMode.WAVEFORM:',
     '            if voice and voice.loop_enabled and self.mode == EditorMode.WAVEFORM:'),
    ('                lsx, _ = self.data_to_px(partial.loop_start, 0, win_w, win_h)',
     '                lsx, _ = self.data_to_px(voice.loop_start, 0, win_w, win_h)'),
    ('                lex, _ = self.data_to_px(partial.loop_end,   0, win_w, win_h)',
     '                lex, _ = self.data_to_px(voice.loop_end,   0, win_w, win_h)'),
    ('                                          start_val=(partial.loop_start, 0))',
     '                                          start_val=(voice.loop_start, 0))'),
    ('                                          start_val=(partial.loop_end, 0))',
     '                                          start_val=(voice.loop_end, 0))'),
    ('            if self.mode == EditorMode.ENVELOPE and partial:',
     '            if self.mode == EditorMode.ENVELOPE and voice:'),
    ('                knots = partial.active_knots()',
     '                knots = voice.active_knots()'),
    ('                    knot = partial.active_knots()[best_i]',
     '                    knot = voice.active_knots()[best_i]'),
    ('                if partial.env_type != "adsr":',
     '                if voice.env_type != "adsr":'),
    ('                    partial.env_knots.append([t_n, v_n])',
     '                    voice.env_knots.append([t_n, v_n])'),
    ('                    partial.env_knots.sort(key=lambda k: k[0])',
     '                    voice.env_knots.sort(key=lambda k: k[0])'),
    # MOUSEBUTTONDOWN right click – envelope knot delete
    ('            if self.mode == EditorMode.ENVELOPE and partial and partial.env_type != "adsr":',
     '            if self.mode == EditorMode.ENVELOPE and voice and voice.env_type != "adsr":'),
    ('                knots = partial.active_knots()',
     '                knots = voice.active_knots()'),
    ('                        if len(partial.env_knots) > 2:',
     '                        if len(voice.env_knots) > 2:'),
    ('                            partial.env_knots.pop(ki)', '                            voice.env_knots.pop(ki)'),
    # MOUSEMOTION – drag handler voice_d
    ('                partial_d = next(\n                    (p for p in patch.partials if p.key == self.drag.partial_key), None)',
     '                voice_d = next(\n                    (p for p in patch.voices if p.key == self.drag.voice_key), None)'),
    ('                if partial_d and self.drag.kind == "loop_start":',
     '                if voice_d and self.drag.kind == "loop_start":'),
    ('                    partial_d.loop_start = max(0.0, min(t_now, partial_d.loop_end - 0.01))',
     '                    voice_d.loop_start = _snap_to_phase_boundary(t_now, voice_d, patch, self._phase_cycles)'),
    ('                if partial_d and self.drag.kind == "loop_end":',
     '                if voice_d and self.drag.kind == "loop_end":'),
    ('                    partial_d.loop_end = min(1.0, max(t_now, partial_d.loop_start + 0.01))',
     '                    voice_d.loop_end = _snap_to_phase_boundary(t_now, voice_d, patch, self._phase_cycles)'),
    ('                if partial_d and self.drag.kind == "knot":',
     '                if voice_d and self.drag.kind == "knot":'),
    ('                    ki = self.drag.index\n                    if partial_d.env_type == "adsr":',
     '                    ki = self.drag.index\n                    if voice_d.env_type == "adsr":'),
    ('                        knots = partial_d.adsr.to_knots(1.0)',
     '                        knots = voice_d.adsr.to_knots(1.0)'),
    ('                            if ki == 1:\n                                partial_d.adsr.attack = max(0.001, t_now * dur)\n                                partial_d.adsr.peak   = max(0.0, min(1.0, v_now))\n                            elif ki == 2:\n                                a = partial_d.adsr.attack\n                                partial_d.adsr.decay   = max(0.001, t_now * dur - a)\n                                partial_d.adsr.sustain = max(0.0, min(1.0, v_now))\n                            elif ki == 3:\n                                partial_d.adsr.release = max(0.001, (1.0 - t_now) * dur)',
     '                            if ki == 1:\n                                voice_d.adsr.attack = max(0.001, t_now * dur)\n                                voice_d.adsr.peak   = max(0.0, min(1.0, v_now))\n                            elif ki == 2:\n                                a = voice_d.adsr.attack\n                                voice_d.adsr.decay   = max(0.001, t_now * dur - a)\n                                voice_d.adsr.sustain = max(0.0, min(1.0, v_now))\n                            elif ki == 3:\n                                voice_d.adsr.release = max(0.001, (1.0 - t_now) * dur)'),
    ('                    else:\n                        if 0 <= ki < len(partial_d.env_knots):\n                            partial_d.env_knots[ki] = [\n                                max(0.0, min(1.0, t_now)),\n                                max(0.0, min(1.0, v_now)),\n                            ]\n                            # Keep sorted by time, anchoring endpoints\n                            if 0 < ki < len(partial_d.env_knots) - 1:\n                                partial_d.env_knots.sort(key=lambda k: k[0])',
     '                    else:\n                        if 0 <= ki < len(voice_d.env_knots):\n                            voice_d.env_knots[ki] = [\n                                max(0.0, min(1.0, t_now)),\n                                max(0.0, min(1.0, v_now)),\n                            ]\n                            # Keep sorted by time, anchoring endpoints\n                            if 0 < ki < len(voice_d.env_knots) - 1:\n                                voice_d.env_knots.sort(key=lambda k: k[0])'),
    # hover detection for envelope knots
    ('            if self.mode == EditorMode.ENVELOPE and partial:',
     '            if self.mode == EditorMode.ENVELOPE and voice:'),
    ('                knots = partial.active_knots()\n                self._hover_cp = -1',
     '                knots = voice.active_knots()\n                self._hover_cp = -1'),
    # DragState: partial_key= in drag state constructions
    ('                                          index=best_i, partial_key=self.active_key,',
     '                                          index=best_i, voice_key=self.active_key,'),
    ('                    self.drag = DragState(active=True, kind="loop_start",\n                                          partial_key=self.active_key,',
     '                    self.drag = DragState(active=True, kind="loop_start",\n                                          voice_key=self.active_key,'),
    ('                    self.drag = DragState(active=True, kind="loop_end",\n                                          partial_key=self.active_key,',
     '                    self.drag = DragState(active=True, kind="loop_end",\n                                          voice_key=self.active_key,'),
    # PartialPanel slider handler
    ('            elif key == "amplitude":     partial.amplitude        = val',
     '            elif key == "amplitude":     voice.amplitude          = val'),
    # on_add_partial callback wiring in AnalyticDriverViewer
    ('        self.patch_panel.on_add_partial  = self._on_add_partial',
     '        self.patch_panel.on_add_voice    = self._on_add_voice'),
    # _on_add_partial method rename
    ('    def _on_add_partial(self) -> None:', '    def _on_add_voice(self) -> None:'),
    ('        p = AnalyticVoice()\n        p.label   = f"P{len(self.patch.partials) + 1}"\n        p.freq_hz = 220.0 * (2 ** len(self.patch.partials))\n        colors = [(100,180,255),(255,130,60),(120,220,120),(220,80,160),(200,200,80)]\n        p.color = list(colors[len(self.patch.partials) % len(colors)])\n        self.patch.partials.append(p)',
     '        p = AnalyticVoice()\n        p.label   = f"V{len(self.patch.voices) + 1}"\n        p.freq_hz = 220.0 * (2 ** len(self.patch.voices))\n        colors = [(100,180,255),(255,130,60),(120,220,120),(220,80,160),(200,200,80)]\n        p.color = list(colors[len(self.patch.voices) % len(colors)])\n        self.patch.voices.append(p)'),
]

count = 0
misses = []
for old, new in replacements:
    if old in src:
        src = src.replace(old, new, 1)
        count += 1
    elif old != new:  # skip no-op entries
        misses.append(old[:70])

with open("analytic_driver.py", "w", encoding="utf-8") as f:
    f.write(src)

print(f"Applied: {count}/{len(replacements)}")
if misses:
    print("MISSES:")
    for m in misses:
        print(" ", repr(m))
