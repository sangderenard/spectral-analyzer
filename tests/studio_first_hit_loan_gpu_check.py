"""Bounded isolated composed-engine continuation gate; no live API calls."""
from pathlib import Path
import sys,json,os,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import demo_pluck_gl as r
    from OpenGL import GL as g
    from camera_designer.studio_first_hit_loan import FirstHitLoan,loan_shader
    r._ensure_worker_gl_context()
    buffers=[];textures=[];accs=[]
    def buffer(data):
        b=int(g.glGenBuffers(1));buffers.append(b);g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,b)
        g.glBufferData(g.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,g.GL_DYNAMIC_DRAW);return b
    geom=np.zeros((1,16),np.float32)
    geom[0,:3]=[-4,-4,0];geom[0,4:7]=[8,0,0];geom[0,8:11]=[4,8,0];geom[0,12:15]=[0,0,1]
    geom.view(np.uint32)[0,15]=1
    shade=np.zeros((1,16),np.float32);shade[0,:4]=[.1,0,0,1.5];shade[0,4:8]=[.1,0,0,1]
    shade[0,8:12]=[.2,.8,.4,1];shade[0,12:15]=[1,1,1]
    nodes=np.array([[-4,-4,-1e-5,-1,4,4,1e-5,-1,0,1,0,0]],np.float32)
    multi=os.environ.get('STUDIO_LOAN_MULTI')=='1'
    if multi:
        geom=np.repeat(geom,3,axis=0);shade=np.repeat(shade,3,axis=0)
        geom[0,:3]=[-.8,-.8,0];geom[0,4:7]=[1.6,0,0];geom[0,8:11]=[.8,1.6,0]
        geom[1,:3]=[-.4,-.4,.4];geom[1,4:7]=[.8,0,0];geom[1,8:11]=[.4,.8,0]
        geom[2]=geom[1] # exact tie, first leaf candidate must win
        shade[1,8:11]=[.9,.1,.2];shade[2,8:11]=[.1,.2,.9]
        nodes[0,8:10]=[0,3];nodes[0,6]=.40001
    stage=os.environ.get('STUDIO_LOAN_STAGE')
    regions=None;spheres=None;eye=[0,0,2];target=[0,0,0];mat=np.zeros(32,np.float32);ids=np.arange(len(geom),dtype=np.uint32)
    if stage:
        from camera_designer.studio_photo_geometry import prepare_photo_geometry
        from camera_designer.studio_region_gpu import region_records_from_geometry
        from camera_designer.studio_rod_bundle_gpu import pack_bundle_records
        folder=Path(stage);config=json.loads((folder/'config.json').read_text())
        with np.load(folder/'payload.npz',allow_pickle=False) as payload:
            scene,spheres,stats,_=prepare_photo_geometry(config,payload['vertices'],payload['material_ids'],payload['pbr'])
        geom,shade,mat=scene[:3];nodes,ids=r._build_gpu_bvh(scene[3])
        regions=region_records_from_geometry(geom)
        if config.get('region_media',{}).get('axial_bundle') is not None:
            regions=pack_bundle_records(regions,config['region_media']['axial_bundle'],scale_m=config.get('scale_m',.025),center_m=config.get('object_center_m',[0,0,.18]),host_triangle=int(np.flatnonzero(geom[:,3]==-1)[0]))
        eye=config['camera']['eye_m'];target=config['camera']['target_m']
    gb=buffer(geom);nb=buffer(nodes);ib=buffer(ids);sb=buffer(shade);mb=buffer(mat);pb=buffer(np.zeros((1,16),np.float32))
    region_buffer=buffer(regions) if regions is not None else 0
    from camera_designer.studio_rod_bundle_gpu import region_record_count
    region_count=region_record_count(regions) if regions is not None else 0
    source=r._GPU_SENSOR_REGION_CS if region_count else r._GPU_SENSOR_CS
    variant='region-bvh' if region_count else ('plain-bvh' if stage else 'plain')
    for _ in range(3):
        t=int(g.glGenTextures(1));textures.append(t);g.glBindTexture(g.GL_TEXTURE_3D,t)
        g.glTexImage3D(g.GL_TEXTURE_3D,0,g.GL_R32UI,2,2,2,0,g.GL_RED_INTEGER,g.GL_UNSIGNED_INT,np.zeros(8,np.uint32))
        g.glTexParameteri(g.GL_TEXTURE_3D,g.GL_TEXTURE_MIN_FILTER,g.GL_NEAREST);g.glTexParameteri(g.GL_TEXTURE_3D,g.GL_TEXTURE_MAG_FILTER,g.GL_NEAREST)
    sphere_test=os.environ.get('STUDIO_LOAN_SPHERE')=='1'
    sphere_buffer=buffer(np.array([[0,0,.7,.35,0,0,0,0]],np.float32)) if sphere_test else 0
    if spheres is not None and len(spheres):sphere_buffer=buffer(spheres);sphere_test=True
    full_image=os.environ.get('STUDIO_PUMP_FULL')=='1'
    test_edge=int(os.environ.get("STUDIO_PUMP_EDGE",128 if full_image else 16))
    test_rows=max(1,int(os.environ.get("STUDIO_PUMP_BATCH",256))//test_edge)
    def make():
        a=r.SensorAccumulator(gb,nb,ib,sb,mb,len(geom),len(nodes),textures,[-5,-5,-1],[5,5,1],(2,2,2),eye,target,test_edge,test_edge,rows_per_frame=test_rows,max_bounces=2,aperture_radius=.01,focus_dist=2,vol_alpha=0,ssbo_spheres=sphere_buffer,n_spheres=(len(spheres) if spheres is not None else int(sphere_test)),ssbo_regions=region_buffer,n_regions=region_count,ssbo_native_pbr=pb)
        a._sprinkle=False
        if os.environ.get("STUDIO_TEST_HISTORY"):a.set_history(int(os.environ["STUDIO_TEST_HISTORY"]),checkpoint=True)
        accs.append(a)
        for _ in range(int(os.environ.get("STUDIO_PUMP_START",0))):a._dispatch_once()
        return a
    def read(a):
        g.glFinish();result=[]
        for t in a._textures:
            g.glBindTexture(g.GL_TEXTURE_2D,t)
            result.append(np.asarray(g.glGetTexImage(g.GL_TEXTURE_2D,0,g.GL_RGBA,g.GL_FLOAT)).copy())
        return result
    try:
        if '--coverage' in sys.argv:
            from camera_designer.studio_completed_hit_verifier import CompletedFirstHitVerifier
            offset=int(os.environ.get('STUDIO_COVERAGE_OFFSET',0))
            baseline=make();baseline._row_off=offset;baseline._dispatch_once();expected=read(baseline)
            a=make();a._row_off=offset;loan=FirstHitLoan(a,source,r._prog,'coverage',query_variant=variant)
            before=read(a);assert all(not np.any(x) for x in before)
            request=loan.issue(seconds=60);verifier=CompletedFirstHitVerifier(loan,source,r._prog)
            empty=np.zeros((loan.count,4),np.float32);empty[:,1]=-1
            verifier(loan.rays,empty);hits=verifier.last_expected.copy()
            loan.complete(request['token'],request['revision'],hits,verifier)
            for x,y in zip(expected,read(a)):np.testing.assert_array_equal(x,y)
            assert (a._row_off,a._pass,a._frame)==(baseline._row_off,baseline._pass,baseline._frame)
            assert loan.count==test_edge*min(test_rows,test_edge-offset)
            print(json.dumps({'width':test_edge,'offset':offset,'rays':loan.count,'parity':True,'pass':a._pass}))
            verifier.close();return
        skip_reference=os.environ.get('STUDIO_SKIP_REFERENCE')=='1'
        expected=None;baseline_seconds=None
        if not skip_reference:
            baseline=make();baseline_start=time.perf_counter();baseline._dispatch_once();
            while full_image and baseline._pass<1:
                if os.environ.get('STUDIO_PUMP_FIXED_SEED')=='1':baseline._frame=0
                baseline._dispatch_once()
            expected=read(baseline);baseline_seconds=time.perf_counter()-baseline_start
            assert any(np.any(x) for x in expected), 'Fixture rendered nothing'
            if not full_image and int(os.environ.get('STUDIO_PUMP_START',0))==0:
                for x in expected:assert np.count_nonzero(x.reshape(-1,4)[:,3])==test_edge*min(test_rows,test_edge),'Padded shader invocations escaped the requested strip'
        if len(sys.argv)>1 and sys.argv[1] in ('--pump','--https-pump'):
            if sys.argv[1]=='--https-pump':from first_hit_https_pump import run
            else:from first_hit_process_pump import run
            a=make();profile=run(a,r,ROOT)
            if expected is not None:
                for x,y in zip(expected,read(a)):np.testing.assert_array_equal(x,y)
            if full_image:
                for x in read(a):assert np.count_nonzero(x.reshape(-1,4)[:,3])==test_edge*test_edge,'Incomplete image coverage'
            out=ROOT/'build/dot-first-hit-loan-20261109';out.mkdir(parents=True,exist_ok=True)
            receipt={'sensor_sha256':hashlib.sha256(b''.join(x.tobytes() for x in read(a))).hexdigest(),'fixed_dispatch_seed':os.environ.get('STUDIO_PUMP_FIXED_SEED')=='1','ray_batch':test_edge*test_rows,'full_image_edge':test_edge,'baseline_sensor_and_readback_seconds':baseline_seconds,'identical_scene_rays_and_seed':True if expected is not None else None,'timing_components_overlap':True,'production_pump':True,'full_sensor_bitwise_parity':True if expected is not None else None,'triangles':len(geom),'regions':region_count,**profile}
            (out/('full-image-pump-receipt.json' if full_image else 'production-pump-receipt.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));return
        a=make();before=read(a);loan=FirstHitLoan(a,source,r._prog,'fixture-v1',query_variant=variant)
        assert (a._row_off,a._pass,a._frame)==(0,0,0)
        for x,y in zip(before,read(a)):np.testing.assert_array_equal(x,y)
        try:a._dispatch_once();raise AssertionError('pending strip dispatched')
        except RuntimeError:pass
        request=loan.issue(seconds=60 if stage else 10)
        try:loan.issue();raise AssertionError('duplicate lease')
        except RuntimeError:pass
        # Authoritative GPU reference uses unchanged original hit_tri/traversal.
        src=source.replace('void main() {','void unused_sensor_main() {',1)
        src+='''\nlayout(std430,binding=16) readonly buffer ReferenceRays {vec4 studio_loan_rays[];};\nlayout(std430,binding=17) buffer ReferenceResult {vec4 answer[];};
void main(){uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*16u;if(i>=256u)return;
float t;int h=nearest_filtered(studio_loan_rays[2*i].xyz,studio_loan_rays[2*i+1].xyz,t,false);
answer[i]=vec4(h>=0?t:0.0,float(h),h>=0?1.0:0.0,float(h));}'''
        p=r._prog((src,g.GL_COMPUTE_SHADER));assert g.glGetProgramiv(p,g.GL_LINK_STATUS)
        result=buffer(np.zeros((256,4),np.float32));g.glBindBufferBase(g.GL_SHADER_STORAGE_BUFFER,17,result)
        for slot,b in [(0,gb),(1,nb),(2,ib),(9,sb),(16,loan.buffers[0])]:g.glBindBufferBase(g.GL_SHADER_STORAGE_BUFFER,slot,b)
        g.glUseProgram(p)
        for key,val in [('uNodeCount',len(nodes)),('uTriCount',len(geom)),('uStudioSphereCount',0)]:g.glUniform1i(g.glGetUniformLocation(p,key),val)
        g.glDispatchCompute(1,1,1);g.glMemoryBarrier(g.GL_SHADER_STORAGE_BARRIER_BIT);g.glFinish()
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,result);hits=np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,0,4096),np.float32).copy().reshape(-1,4)
        out=ROOT/'build/dot-first-hit-loan-20261109';out.mkdir(parents=True,exist_ok=True)
        (out/('stage-region-request.json' if stage else ('multi-request.json' if multi else 'request.json'))).write_text(json.dumps({'query_variant':variant,'revision':request['revision'],'rays':request['rays'].ravel().tolist(),'triangles':request['triangles'],'bvh_nodes':request['bvh_nodes'],'bvh_ids':request['bvh_ids'],'bvh_epsilon':request['bvh_epsilon'],'expected':hits.ravel().tolist()}))
        if multi and not stage:
            assert set(hits[:,1])=={-1,0,1}, 'Fixture must cover miss, near/far hit and exact tie'
        # A material change through the same GL handle invalidates the loan.
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,mb);g.glBufferSubData(g.GL_SHADER_STORAGE_BUFFER,0,4,np.array([1],np.float32))
        try:loan.complete(request['token'],request['revision'],hits,lambda *args:True);raise AssertionError('same-handle mutation accepted')
        except ValueError:pass
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,mb);g.glBufferSubData(g.GL_SHADER_STORAGE_BUFFER,0,4,np.asarray(mat,dtype=np.float32).ravel()[:1])
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,pb);g.glBufferSubData(g.GL_SHADER_STORAGE_BUFFER,12,4,np.array([.4],np.float32))
        try:loan.complete(request['token'],request['revision'],hits,lambda *args:True);raise AssertionError('native roughness mutation accepted')
        except ValueError:pass
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,pb);g.glBufferSubData(g.GL_SHADER_STORAGE_BUFFER,12,4,np.array([0],np.float32))
        saved_eye=a._eye.copy()
        def mutate_during_validation(*args):
            a._eye[0]+=.25
            return True
        try:loan.complete(request['token'],request['revision'],hits,mutate_during_validation);raise AssertionError('validator mutation accepted')
        except ValueError:pass
        a._eye[:]=saved_eye
        submitted=hits;relay=None;native_receipt=None
        from camera_designer.studio_completed_hit_verifier import CompletedFirstHitVerifier
        verifier=CompletedFirstHitVerifier(loan,source,r._prog)
        def verify_returned(rays,value):return verifier(rays,value)
        def commit(candidate):
            loan.complete(request['token'],request['revision'],candidate,verify_returned)
        if len(sys.argv)>1 and sys.argv[1] in ('--nodus','--nodus-recovery','--nodus-malformed'):
            from first_hit_nodus_gate import run
            loan.deadline=0;request=loan.issue(seconds=60)
            native_receipt=run(request,commit,reject_first=('malformed' if sys.argv[1]=='--nodus-malformed' else sys.argv[1]=='--nodus-recovery'))
        elif len(sys.argv)>1 and sys.argv[1]=='--relay':
            import threading,subprocess
            from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
            turing=ROOT.parent/'turing';sys.path.insert(0,str(turing/'examples'))
            from meowdoku.client_ray_relay import ClientRayRelay,handle_relay_post
            manifest=json.loads((turing/'build/dot-client-ray-20261109/studio-compute/studio_hit_tri.json').read_text())
            relay=ClientRayRelay();credential=relay.register('fixture','fixture-session')
            # Renew the uncommitted lease after reference preparation.
            loan.deadline=0;request=loan.issue(seconds=60)
            relay.publish('fixture',request,manifest['source_sha256'],seconds=60)
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def _json(self,value,status=200):
                    data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                def do_POST(self):
                    if not handle_relay_post(self,relay,'fixture-session'):self._json({},404)
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                run=subprocess.run(['node',str(ROOT/'tests/first_hit_relay_client.mjs')],input=json.dumps({'port':server.server_port,'credential':credential,'job':'fixture','kernel':manifest['source_sha256'],'wasm':str(turing/'build/dot-client-ray-20261109/studio-compute/studio_hit_tri.wasm')}),text=True,capture_output=True,timeout=30)
                assert run.returncode==0,run.stderr
                print(run.stdout)
                received=relay.take_result('fixture');assert received is not None
                submitted=np.asarray(received['hits'],np.float32)
                assert relay.take_result('fixture') is None
            finally:server.shutdown();server.server_close();thread.join()
        elif len(sys.argv)>1:
            payload=json.loads(Path(sys.argv[1]).read_text())
            submitted=np.asarray(payload['hits'],np.float32).reshape(-1,4)
            np.testing.assert_array_equal(submitted[:,1:3],hits[:,1:3])
            np.testing.assert_allclose(submitted[:,0],hits[:,0],rtol=1e-6,atol=1e-7)
        if native_receipt is None:commit(submitted)
        if relay is not None:
            relay.finish('fixture','committed');assert relay.poll('fixture','fixture-session',credential)['state']=='committed';relay.close('fixture')
        for x,y in zip(expected,read(a)):np.testing.assert_array_equal(x,y)
        assert (a._row_off,a._pass,a._frame)==(0,1,1)
        try:loan.complete(request['token'],request['revision'],hits,lambda *args:True);raise AssertionError('duplicate accepted')
        except ValueError:pass
        retry=FirstHitLoan(a,source,r._prog,'fixture-v1',query_variant=variant);old=retry.issue();retry.deadline=0
        try:retry.complete(old['token'],old['revision'],hits,lambda *args:True);raise AssertionError('expired accepted')
        except ValueError:pass
        new=retry.issue(include_scene=False);assert 'triangles' not in new and 'bvh_nodes' not in new;assert retry.scene_descriptor()['geometry_hash']==new['geometry_hash'];assert new['token']!=old['token'];np.testing.assert_array_equal(old['rays'],new['rays'])
        retry.close();assert a._pass==1;a._dispatch_once();assert a._pass==2
        receipt={'rays':256,'triangles':len(geom),'hits':int(hits[:,2].sum()),'region_count':region_count,'server_sphere_competition':sphere_test,'full_sensor_bitwise_parity':True,'capture_no_deposit':True,'pending_dispatch_blocked':True,'duplicate_rejected':True,'expiry_retained':True,'reissue_same_rays':True,'cancel_server_fallback':True,'same_handle_mutation_rejected':True,'validator_mutation_rejected':True,'external_result_used':len(sys.argv)>1,'external_result_path':sys.argv[1] if len(sys.argv)>1 else None}
        receipt.update(returned_batches=verifier.checked_batches,verified_rays=verifier.checked_rays,gpu_validation_seconds=verifier.seconds)
        verifier.close()
        if native_receipt:receipt.update(native_receipt)
        if stage and native_receipt:(out/'stage-region-nodus-receipt.json').write_text(json.dumps(receipt,indent=2))
        (out/('stage-region-gpu-receipt.json' if stage else ('sphere-gpu-receipt.json' if sphere_test else ('multi-gpu-receipt.json' if multi else 'gpu-receipt.json')))).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
        g.glDeleteProgram(p)
    finally:
        for a in accs:a.destroy()
        g.glDeleteBuffers(len(buffers),buffers);g.glDeleteTextures(textures)

if __name__=='__main__':main()
