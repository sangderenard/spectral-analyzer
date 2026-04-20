import json
d = json.load(open('analytic_patch.json'))
print('voices:', len(d['voices']))
for v in d['voices']:
    print(f"  {v['key']:15s} role={v.get('voice_role','?'):10s} seq={v.get('seq_role','?'):8s} reg={v.get('register','?'):6s}")
print('modules:', len(d.get('modules',[])))
for m in d.get('modules',[]):
    print(f"  {m.get('kind','?')} label={m.get('label','?')}")
print('routing type:', type(d.get('routing')).__name__)
if isinstance(d.get('routing'), dict):
    print('routing keys:', list(d['routing'].keys())[:10])
# Check placement resonator fields
print('---')
for k in ['placement_resonator', 'placement_resonator_enabled', 'placement']:
    if k in d:
        print(f"Found top-level key: {k} = {d[k]}")
# Check modules for state_machine type
for m in d.get('modules',[]):
    print(f"  module: kind={m.get('kind')} sm_plugin={m.get('sm_plugin','N/A')}")
