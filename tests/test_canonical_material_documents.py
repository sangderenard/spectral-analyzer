import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from canonical_material_documents import (
 MATERIAL_DIRECTORY,MaterialDocument,read_material_document,read_material_documents,
 restore_material_document,native_payload)
from material_db import MaterialDatabase
from spectral_library import load_default_library


def test_all_disk_documents_roundtrip_exact_bytes_and_independent_mapping():
    docs=read_material_documents()
    assert len(docs)==47
    assert len({d.key for d in docs})==len(docs)
    for d in docs:
        raw=(MATERIAL_DIRECTORY/(d.key+'.yaml')).read_bytes()
        assert d.source_bytes==raw
        assert d.revision==hashlib.sha256(raw).hexdigest()
        snapshot=d.snapshot();json.dumps(snapshot,allow_nan=False)
        restored=restore_material_document(snapshot)
        assert restored.key==d.key and restored.source_bytes==raw
        view=d.definition;view['private-test']={'x':1}
        assert 'private-test' not in d.definition
        assert read_material_document(d.key).source_bytes==raw


def test_targeted_edit_preserves_unknown_nested_fields_and_original_source():
    raw=b'# source comment\nname: test\npbr:\n  roughness: 0.2\nfuture:\n  unknown: [3, {value: original}]\n'
    d=MaterialDocument('stable-key','memory',raw)
    assert d.with_field(['pbr','roughness'],.2) is d
    e=d.with_field(['pbr','roughness'],.7)
    assert d.source_bytes==raw and e.key==d.key and e.revision!=d.revision
    assert e.definition['future']==d.definition['future']
    assert e.definition['pbr']['roughness']==.7
    restored=restore_material_document(d.snapshot(),[{'path':['pbr','roughness'],'value':.7}])
    assert restored.definition==e.definition
    with pytest.raises(KeyError):d.with_field(['pbr','missing'],1)
    snap=d.snapshot();snap['source_yaml']+='extra: changed\n'
    with pytest.raises(ValueError,match='revision'):restore_material_document(snap)


def test_alias_edit_is_local_and_recursive_yaml_rejected():
    d=MaterialDocument('test','memory',b'a: &shared {nested: 1}\nb: *shared\n')
    e=d.with_field(['a','nested'],2)
    assert e.definition=={'a':{'nested':2},'b':{'nested':1}}
    with pytest.raises(ValueError,match='recursive'):MaterialDocument('test','memory',b'a: &loop [*loop]')


@pytest.mark.parametrize('key',['../outside','x/y','x\\y','..','',None])
def test_source_keys_are_plain_stable_library_identifiers(key):
    with pytest.raises(ValueError):read_material_document(key)


def test_native_database_preserves_legacy_rays_and_explicit_pbr():
    d=MaterialDocument('test','memory',b'''reflectivity: 0.31
diffusion: 0.22
absorption: 0.47
ior: 1.5
opacity: 0.1
pbr:
  albedo: [0.1,0.2,0.3]
  roughness: 0.06
  metallic: 0.07
  transmission: 0.82
  ior: 1.54
  opacity: 0.2
  emission_rgb: [0.4,0.5,0.6]
unknown_science: {future: preserved}
''')
    payload=native_payload(d)
    assert 'unknown_science' in payload.uncompiled_fields
    assert payload.payload['unknown_science']==d.definition['unknown_science']
    db=MaterialDatabase();db.register(d.key,payload.payload);t=db.build_tensors()
    np.testing.assert_allclose(t['raymat_compat'][0,:3],[.31,.22,.47])
    np.testing.assert_allclose(t['pbr'][0,:11],[.1,.2,.3,.06,.07,.82,1.54,.2,.4,.5,.6])
    assert read_material_document('construction_glass').definition['pbr']['roughness']==.06


def test_all47_imports_preserve_source_and_make_missing_profiles_explicit():
    load_default_library();db=MaterialDatabase();missing={};compiled=[]
    for d in read_material_documents():
        before=d.source_bytes
        try:p=native_payload(d)
        except ValueError as e:
            assert 'unresolved color_profile_name' in str(e)
            missing[d.key]=str(e);continue
        db.register(d.key,p.payload);compiled.append(d.key)
        assert d.source_bytes==before
    assert set(missing)=={'acoustic_foam_panel','acoustic_plaster_ceiling','construction_glass',
      'corrugated_steel_exterior','flashlight_reflector','flashlight_shell','painted_plaster_wall',
      'pearl_white_tile','tungsten_filament'}
    assert len(compiled)==38
    t=db.build_tensors()
    for name in ('pbr','phong_compat','raymat_compat','spectral','enamel','texture_stack'):
        assert np.isfinite(t[name]).all()
    for i,key in enumerate(compiled):
        d=read_material_document(key).definition
        if all(k in d for k in ('reflectivity','diffusion','absorption')):
            np.testing.assert_allclose(t['raymat_compat'][i,:3],[d[k] for k in ('reflectivity','diffusion','absorption')])
        for prop,index in (('roughness',3),('metallic',4),('transmission',5),('ior',6),('opacity',7)):
            if prop in d.get('pbr',{}):assert t['pbr'][i,index]==pytest.approx(d['pbr'][prop])


def test_native_bands_unknown_and_profile_alias_survive():
    load_default_library();d=read_material_document('construction_borosilicate')
    p=native_payload(d)
    assert p.payload['emit_profile_idx']>=0
    assert 'emission_profile_name' in d.definition
    with pytest.raises(ValueError,match='unresolved'):
        native_payload(MaterialDocument('x','memory',b'color_profile_name: nonexistent_exact_profile'))
