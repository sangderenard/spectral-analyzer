State machine plugin API

Folder
- Drop plugin files in `sm_plugins/` as `name.py`.
- The module selector uses the filename stem.

Required plugin API
- `STATE_VARS: list[str]`
- `step(inputs, state, dt, n_items=1, use_torch=False, **kwargs) -> dict`

Optional plugin API
- `OUTPUT_VARS: list[str]`
  Declares which trajectories become routing output nodes. When omitted, the
  host uses `STATE_VARS`.
- `PARAM_SPECS: list[dict]`
  Plugin-defined UI knobs stored on the module and passed back to `step()` as
  `params={...}`.
- `ITEM_PREFIX = "m"`
- `item_names(n_items) -> list[str]`

What the host passes
- `inputs`
  Mapping `{src_key: array(T,)}` of routed signals that feed the state-machine module's main node.
  Values are `numpy.ndarray` by default or `torch.Tensor` when the module backend is set to `torch` and torch is available.
  Complex inputs are allowed.
- `state`
  Mapping `{item_name: {var_name: float}}` with the previous scalar state for each item/variable.
- `plugin_state`
  Optional transient plugin-owned cache/state object from the previous step.
  Use this for histories, overlap buffers, tensor caches, and other non-scalar
  runtime state that should not be serialized into the patch.
- `dt`
  Seconds per sample.
- `n_items`
  The module's configured item count.
- `use_torch`
  Backend preference flag from the module UI.

What the plugin must return
- `trajectories`
  Mapping `{item_name: {var_name: array(T,)}}`.
  Arrays may be real or complex-valued sample trajectories. The host preserves
  complex outputs and routes them as complex analytic signals.
  Returned item/var names should match the active item names and `OUTPUT_VARS`
  or `STATE_VARS`.

Extended return form
- A plugin may instead return:
  `{ "outputs": {item: {var: array(T,)}}, "state": {item: {var: scalar}}, "plugin_state": {...} }`
- `outputs` drives routing output nodes.
- `state` persists scalar state variables without forcing them to become output nodes.
- `plugin_state` persists transient plugin-owned runtime caches between steps.

Host behavior and current limits
- Output node keys are generated as `"{module_key}_sm_{item}_{var}"`.
- The host now exposes plugin parameters from `PARAM_SPECS` in the module UI.
- Import failures currently fall back silently in the host.

Practical guidance
- Accept both numpy and torch inputs, but convert internally if you need one specific backend.
- Keep return shapes consistent across every item and var.
- Use deterministic item names if you want saved patches to remain stable across reloads.
