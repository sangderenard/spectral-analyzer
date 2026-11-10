"""Isolated real HTTPS handler + native Nodus + compiled client + sensor test."""
import sys,time,threading,json,subprocess,ssl,os,tempfile,hashlib
from pathlib import Path
from urllib.request import Request,urlopen
from http.server import ThreadingHTTPServer

def run(a,renderer,source_root):
 root=Path(source_root).parent;turing=root/'turing';sys.path[:0]=[str(turing),str(turing/'examples')]
 from meowdoku import server
 from camera_designer.studio_cooperative_sensor import CooperativeSensorPump
 temporary=tempfile.TemporaryDirectory(prefix='studio-ray-https-');web=Path(temporary.name);(web/'boards.json').write_text('{"boards":[],"levels":{}}')
 sessions=server.Sessions(web,None);sessions.lifecycle=None;sessions.access.lookup=lambda url:None
 token,_,_=sessions.get(None);owner='browser:'+hashlib.sha256(token.encode()).hexdigest();bridge=sessions.client_ray_bridge
 secret=bridge.register('test',owner);credential=bridge.join('test',owner)['credential'];stop=threading.Event();errors=[];attempts=[];client_samples=[]
 tls=turing/'build/meowdoku/tls';httpd=ThreadingHTTPServer(('127.0.0.1',0),server.make_handler(sessions))
 context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(tls/'cert.pem',tls/'key.pem');httpd.socket=context.wrap_socket(httpd.socket,server_side=True)
 serving=threading.Thread(target=httpd.serve_forever,daemon=True);serving.start();base=f'https://127.0.0.1:{httpd.server_port}/api/client_rays'
 client_context=ssl.create_default_context(cafile=str(tls/'ca.crt'))
 def call(action,body):
  req=Request(base+'/'+action,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+credential,'X-Meowdoku-Rays':'1','Cookie':server.COOKIE+'='+token},method='POST')
  with urlopen(req,context=client_context,timeout=20) as r:return json.load(r)
 def client():
  resident=None;process=None;resident_key=None
  try:
   while not stop.is_set():
    response=call('poll',{'job':'test'})
    if response.get('state')=='pending':
     w=response['work']
     if 'triangles' in w:resident={k:w[k] for k in ('triangles','bvh_nodes','bvh_ids','bvh_epsilon','query_variant')}
     elif resident is not None:w={**resident,**w}
     folder='studio-region-compute' if a._n_regions else 'studio-compute'
     encode_start=time.perf_counter();encoded=(json.dumps(w)+'\n') if os.environ.get('STUDIO_RESIDENT_TRAVERSAL')!='1' else '';encode_elapsed=time.perf_counter()-encode_start;process_start=time.perf_counter()
     runner='studio_bvh_stdio_profile.mjs' if os.environ.get('STUDIO_DETAILED_PROFILE')=='1' else 'studio_bvh_stdio.mjs'
     if os.environ.get('STUDIO_RESIDENT_TRAVERSAL')=='1':
      if process is None:process=subprocess.Popen(['node',str(Path(source_root)/'tests/first_hit_resident_client.mjs'),str(turing/'build/dot-client-ray-20261109'/folder),str(turing/'examples/meowdoku/web/studio_traversal_compute.js')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
      key=(w.get('query_geometry_hash'),w['query_kernel'])
      send=w if resident_key!=key else {k:v for k,v in w.items() if k not in ('triangles','bvh_nodes','bvh_ids')}
      encode_start=time.perf_counter();encoded=json.dumps(send)+'\n';encode_elapsed=time.perf_counter()-encode_start
      process.stdin.write(encoded);process.stdin.flush();line=process.stdout.readline()
      if not line:raise RuntimeError('Resident compute process ended')
      value=json.loads(line);resident_key=key
     else:
      p=subprocess.run(['node',str(turing/'examples/meowdoku'/runner),str(turing/'build/dot-client-ray-20261109'/folder)],input=encoded,text=True,capture_output=True,timeout=40)
      if p.returncode:raise RuntimeError(p.stderr)
      value=json.loads(p.stdout.splitlines()[0])
     if 'error' in value:raise RuntimeError(value['error'])
     client_samples.append({'process_wall_seconds':time.perf_counter()-process_start,'input_encode_seconds':encode_elapsed,'input_bytes':len(encoded.encode()),'compute_ms':value.get('elapsedMs'),**value.get('stdio_profile',{}),'query_profile':value.get('profile')});hits=value['hits'];hits=[hits[i:i+4] for i in range(0,len(hits),4)]
     if response.get('scene_ack_required'):call('scene_ack',{'job':'test','scene':{k:w[k] for k in ('revision','geometry_hash','query_kernel','query_geometry_hash') if k in w}})
     result={k:w[k] for k in ('token','revision','sample_prefix','geometry_hash','kernel','query_kernel')}
     if 'query_geometry_hash' in w:result['query_geometry_hash']=w['query_geometry_hash']
     result['hits']=hits;attempts.append(1)
     if os.environ.get('STUDIO_PUMP_REJECT')=='1' and len(attempts)==1:result['hits'][0][0]*=.5
     result['profile']={'client_compute_ms':value.get('elapsedMs',0),'rays':len(hits),'backend':'wasm'}
     call('result',{'job':'test','result':result})
    stop.wait(.15)
  except Exception as e:
   errors.append(e);print('CLIENT ERROR:',repr(e),flush=True)
   try:call('cancel',{'job':'test'})
   except Exception:pass
  finally:
   if process is not None:
    process.stdin.close()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.terminate();process.wait()
 thread=threading.Thread(target=client,daemon=True);thread.start()
 env={'MEOWDOKU_RAY_BRIDGE_URL':base+'/engine','MEOWDOKU_RAY_BRIDGE_JOB':'test','MEOWDOKU_RAY_BRIDGE_SECRET':secret,'MEOWDOKU_RAY_BRIDGE_CA':str(tls/'ca.crt')};old={k:os.environ.get(k) for k in env};os.environ.update(env)
 def capture_reject(issued,result,verifier):
  with bridge.relay.lock:scene=bridge.relay.jobs['test']['scene']
  data={**scene,**issued,'rays':issued['rays'].ravel().tolist(),'expected':verifier.last_expected.ravel().tolist(),'returned':result['hits']}
  (Path(source_root)/'build/dot-first-hit-loan-20261109/rejected-strip.json').write_text(json.dumps(data))
 def capture_verified(issued,result,verifier):
  name=os.environ.get('STUDIO_QUERY_CAPTURE')
  if not name or not hasattr(verifier,'last_expected'):return
  path=Path(source_root)/'build/dot-first-hit-loan-20261109'/Path(name).name
  if path.exists():return
  with bridge.relay.lock:scene=bridge.relay.jobs['test']['scene']
  data={**scene,**issued,'rays':issued['rays'].ravel().tolist(),'expected':verifier.last_expected.ravel().tolist()}
  path.write_text(json.dumps(data))
 native_library=root/'nodus/build/Release/nodus_contract_service_diagnostic.dll' if os.environ.get('STUDIO_NATIVE_DIAGNOSTIC')=='1' else None
 pump=CooperativeSensorPump(renderer,'https-fixture',on_reject=capture_reject,on_verified=capture_verified,native_library=native_library,verification_mode=('gpu_reference' if os.environ.get('STUDIO_SERVER_VERIFY')=='1' else 'participant'))
 try:
  pump.step(a)
  while os.environ.get('STUDIO_PUMP_FULL')=='1' and a._pass<1:
   if os.environ.get('STUDIO_PUMP_FIXED_SEED')=='1':a._frame=0
   pump.step(a)
   if errors:raise errors[0]
   print(json.dumps({'completed_rays':pump.profile['committed_rays'],'pass':a._pass}),flush=True)
  if errors:raise errors[0]
  return dict(pump.profile,https_transport=True,client_stage_samples=client_samples)
 finally:
  stop.set();thread.join(timeout=45);pump.close();httpd.shutdown();httpd.server_close();serving.join();sessions.marble_photos.close();temporary.cleanup()
  for k,v in old.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
