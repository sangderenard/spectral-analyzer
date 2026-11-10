"""Isolated real owner relay/native host/compiled client/GL pump integration."""
import sys,time,threading,json,subprocess
from pathlib import Path

def run(a,renderer,source_root):
 root=Path(source_root).parent;turing=root/'turing';sys.path.insert(0,str(turing))
 from examples.meowdoku.client_ray_process_bridge import ClientRayProcessBridge
 from camera_designer.studio_cooperative_sensor import CooperativeSensorPump
 bridge=ClientRayProcessBridge();secret=bridge.register('test','owner');credential=bridge.join('test','owner')['credential'];stop=threading.Event();errors=[]
 def client():
  try:
   while not stop.is_set():
    response=bridge.relay.poll('test','owner',credential)
    if response.get('state')=='pending':
     w=response['work'];scene=w.get('scene')
     # The relay sends packed scene under the work's geometry fields.
     if 'triangles' not in w:
      with bridge.relay.lock:w={**bridge.relay.jobs['test']['scene'],**w}
     folder='studio-region-compute' if a._n_regions else 'studio-compute'
     p=subprocess.run(['node',str(turing/'examples/meowdoku/studio_bvh_stdio.mjs'),str(turing/'build/dot-client-ray-20261109'/folder)],input=json.dumps(w)+'\n',text=True,capture_output=True,timeout=40)
     if p.returncode:raise RuntimeError(p.stderr)
     value=json.loads(p.stdout.splitlines()[0]);hits=value['hits'];hits=[hits[i:i+4] for i in range(0,len(hits),4)]
     result={k:w[k] for k in ('token','revision','sample_prefix','geometry_hash','kernel','query_kernel')};result['hits']=hits
     bridge.relay.submit('test','owner',credential,result)
    stop.wait(.1)
  except Exception as e:errors.append(e)
 thread=threading.Thread(target=client,daemon=True);thread.start()
 def mail(action,body):
  if errors:raise errors[0]
  return bridge.engine('test',secret,action,body)
 pump=CooperativeSensorPump(renderer,'pump-fixture',mail=mail)
 try:
  pump.step(a);return dict(pump.profile)
 finally:stop.set();thread.join(timeout=45);pump.close();bridge.close('test')
