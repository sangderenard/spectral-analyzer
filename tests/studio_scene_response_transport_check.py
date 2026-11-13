"""CPU-only exact-record handoff to an existing Turing JS decoder.

Usage: python tests/studio_scene_response_transport_check.py <turing-root>
Add --require-record-index to also verify the existing optional cell lookup.
Add --camera-pairs to round-trip the exact v2 interval/response buffers instead.
Add --camera-query-segments to also check opt-in, derived query geometry.
No HTTP server, GL context, shader compilation, rendering or scene is started.
"""
import base64
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


JS = r'''
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {receiveSceneResponses}=await import(pathToFileURL(process.argv[1]).href);
let input='';for await(const chunk of process.stdin)input+=chunk;
const fixture=JSON.parse(input),bytes=Buffer.from(fixture.payload,'base64');
const payload=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
const indexed=process.argv[2]==='true';
const result=await receiveSceneResponses(fixture.manifest,payload,{
  sceneRevision:fixture.manifest.scene_revision,information:'evaluated-responses-v1',
  recordIndexBytes:indexed?1024*1024:0,cameraQuerySegments:fixture.cameraQuerySegments});
if(fixture.cameraPairs){
  assert.deepEqual(Array.from(result.cameraRhsRows),fixture.native);
  assert.equal(result.nativeRows.length,0);
}else assert.deepEqual(Array.from(result.nativeRows),fixture.native);
assert.equal(result.applicationReady,false);
assert.equal(result.metadata.physical_completeness_claim,false);
assert.equal(result.memory.gpuBytes,0);
for(let row=0;row<2;row++){
  assert.equal(result.index[row*20+11],fixture.cameraPairs?0:1);
  assert.deepEqual(Array.from(result.index.slice(row*20+16,row*20+19)),[0xffffffff,0xffffffff,0xffffffff]);
}
if(indexed){
  assert.ok(result.recordIndex,'requested existing record-index implementation is required');
  assert.equal(result.recordIndex.stats.segmentRows,fixture.cameraQuerySegments?0:2);
  if(fixture.cameraQuerySegments){
    assert.equal(result.recordIndex.stats.cameraQueryRows,2);
    assert.equal(result.recordIndex.stats.cameraQueryUnavailableRows,0);
  }
  assert.equal(result.recordIndex.stats.pointRows,0);
  assert.equal(result.recordIndex.stats.references,64);
  assert.equal(result.recordIndex.energyStatements,false);
  for(let x=0;x<32;x++)assert.deepEqual([...result.recordIndex.records([x,16,16])],
    [{row:0,support:fixture.cameraQuerySegments?4:2},{row:1,support:fixture.cameraQuerySegments?4:2}]);
}
console.log(JSON.stringify({result:'PASS',records:2,exactNativeWords:fixture.native.length,
  cameraPairs:fixture.cameraPairs,cameraQuerySegments:fixture.cameraQuerySegments,
  crossingReferences:indexed?64:null,applicationReady:false,gpuBytes:0}));
'''


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from camera_designer.studio_response_lineage import NativeResponseLineage
    from camera_designer.studio_scene_responses import SceneResponseStore
    turing = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent / 'turing'
    decoder = (turing / 'examples/meowdoku/web/marble_scene_responses.js').resolve(strict=True)
    lineage = NativeResponseLineage(0)
    store = SceneResponseStore(scene_revision='a'*64, memory_budget_bytes=1024*1024,
        lineage_provider=lambda: lineage.snapshot(), bounds=[[0]*3, [1]*3], dimensions=[32]*3)
    rows = np.zeros((2, 128), dtype='<u4')
    rows[:, 0] = 5; rows[:, 1] = [1, 2]; rows[:, 30] = 1
    rows[:, 21:24] = np.asarray([2, .5, .5], np.float32).view(np.uint32)
    rows[:, 27:30] = np.asarray([0, 1, 0], np.float32).view(np.uint32)
    rows[:, 112] = 2  # Actual virtual primitive stays unmapped, not a guessed mesh recipient.
    rows[:, 114:117] = np.asarray([1, 0, 0], np.float32).view(np.uint32)
    rows[:, 117:120] = np.asarray([-1, .5, .5], np.float32).view(np.uint32)
    rows[:, 7:9] = [4, 6]; rows[:, 32] = 0x80000001; rows[:, 104] = 0x7fc12345
    store.consume_forward(dict(scene_revision='a'*64, stream_id='fixture', sequence=0,
        source_sha256='b'*64, transport_scene_sha256='c'*64, primitive_identity_offsets=[112, 113]), rows)
    camera_query_segments='--camera-query-segments' in sys.argv
    camera_pairs='--camera-pairs' in sys.argv or camera_query_segments
    if camera_pairs:
        if '--require-record-index' in sys.argv and not camera_query_segments:
            raise ValueError('camera query geometry requires explicit --camera-query-segments')
        store.invalidate_scene('f'*64)
        intervals=np.zeros((2,128),dtype='<u4');responses=np.zeros_like(intervals)
        intervals[:,6]=3;intervals[:,40]=7;intervals[:,126]=2
        intervals[:,12:15]=np.asarray([1,0,0],np.float32).view(np.uint32)
        if camera_query_segments:
            intervals[:,4]=6  # Actual nonnegative query hit, not the first-hit field.
            intervals[:,8:11]=np.asarray([-1,.5,.5],np.float32).view(np.uint32)
            intervals[:,11]=np.float32(1.5).view(np.uint32)
            intervals[:,12:15]=np.asarray([2,0,0],np.float32).view(np.uint32)
            intervals[:,32]=[4,7]  # Geometry does not imply surface evaluation or coverage.
            intervals[:,115]=17
            intervals[:,116:119]=np.asarray([999,999,999],np.float32).view(np.uint32)
        intervals[:,5]=4;intervals[:,48]=0x80000001;intervals[:,80]=0x40000002
        responses[:,1]=0x80000000;responses[:,2]=0x7fc12345
        responses[:,96]=1;responses[:,108]=3;responses[:,120]=8
        store.consume_camera_pairs(dict(scene_revision='f'*64,stream_id='fixture-pairs',sequence=0,
            source_sha256='b'*64,transport_scene_sha256='c'*64,
            version='native-camera-intervals-v1',response_schema='native-camera-response-copies-v1',
            complete_dispatch_observation=True,application_ready=False,row_count=2),intervals,responses)
        rows=np.concatenate([intervals,responses],axis=1)
    manifest, payload = store.snapshot()
    assert store.retained_entries == 2
    completed = subprocess.run(['node', '--input-type=module', '-e', JS, str(decoder),
        str('--require-record-index' in sys.argv or camera_query_segments).lower()],
        input=json.dumps(dict(manifest=manifest, payload=base64.b64encode(payload).decode(),cameraPairs=camera_pairs,
                             cameraQuerySegments=camera_query_segments,
                             native=rows.reshape(-1).tolist())),
        text=True, capture_output=True, check=True, timeout=30)
    print(completed.stdout.strip())


if __name__ == '__main__':
    main()
