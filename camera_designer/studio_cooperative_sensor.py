"""Opt-in still sensor transport through native Nodus and owner-scoped mailbox.

The socket contract/FIFO owns admission. HTTP transports only scene/payloads.
All validation and sensor mutation stays on the calling GL owner thread.
"""
import json,os,socket,ssl,sys,time,threading,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from .studio_first_hit_loan import FirstHitLoan
from .studio_completed_hit_verifier import CompletedFirstHitVerifier,AuthorizedParticipantAcceptance

class MailboxHTTPError(RuntimeError):
    def __init__(self,action,status,detail):
        self.action,self.status,self.detail=action,status,detail
        super().__init__(f'Client-ray mailbox {action}: HTTP {status}: {detail}')


def publish_retained_when_ready(mail,work,query_kernel,deadline,stop,clock=time.monotonic):
    """Retry only an explicit non-admission; never retry an uncertain publication.

    Job f4b838 failed at strip 39 because browser readiness expired during
    lease preparation. The same unconsumed lease stays authoritative until
    its original deadline; only the GL owner may subsequently reissue it.
    """
    while not stop.is_set() and clock()<deadline:
        state=mail('ready',{})
        if state.get('state') in ('canceled','uncertain'):
            raise RuntimeError('Client unavailable: '+state['state'])
        if state.get('ready') is True:
            remaining=deadline-clock()
            if remaining<=0:return False
            try:
                mail('publish',{'work':work,'query_kernel':query_kernel,'seconds':min(60,remaining)})
                return True
            except MailboxHTTPError as error:
                if not (error.action=='publish' and error.status==409 and error.detail=='Browser has not requested another batch'):
                    raise
        stop.wait(.1)
    return False


def client_time_ms(profile):
    """Ignore malformed advisory telemetry; it never authorizes a result."""
    value=profile.get('client_compute_ms') if isinstance(profile,dict) else None
    if isinstance(value,(int,float)) and not isinstance(value,bool) and 0<=value<=3600000 and math.isfinite(value):return value
    return None

class CooperativeSensorPump:
    def __init__(self,renderer,revision,*,mail=None,query_manifest=None,on_state=None,on_reject=None,on_verified=None,native_library=None,verification_mode="participant"):
        if verification_mode not in ("participant","gpu_reference"):raise ValueError("Unsupported verification mode")
        self.verification_mode=verification_mode
        self.renderer=renderer;self.revision=revision;self.sequence=0;self.scene_hash=None;self.on_reject=on_reject;self.on_verified=on_verified;self.on_state=on_state or (lambda state,profile:None)
        self.native_library=native_library;self.timeline=[];self.active_timeline=None;self.step_origin=0.;self.inner_end=0.
        self.profile=dict(verification_mode=verification_mode,committed_rays=0,completed_batches=0,verified_rays=0,verification_seconds=0.,mailbox_seconds=0.,native_step_seconds=0.,total_seconds=0.,client_reported_compute_ms=0.,client_reported_batches=0,verified_near_ties=0,reoffers=0)
        root=Path(__file__).resolve().parents[2];self.nodus=root/'nodus';self.turing=root/'turing'
        sys.path.insert(0,str(self.nodus/'tools'))
        self.dirs=[os.add_dll_directory(str(self.nodus/'build/Release')),os.add_dll_directory(str(Path(sys.executable).parent/'Lib/site-packages/torch/lib'))]
        self.base=os.environ.get('MEOWDOKU_RAY_BRIDGE_URL','');self.job=os.environ.get('MEOWDOKU_RAY_BRIDGE_JOB','');self.secret=os.environ.get('MEOWDOKU_RAY_BRIDGE_SECRET','')
        self.context=ssl.create_default_context(cafile=os.environ.get('MEOWDOKU_RAY_BRIDGE_CA')) if mail is None else None
        if mail is None and (not self.base.startswith('https://127.0.0.1:') or not self.job or not self.secret):raise RuntimeError('Cooperative job needs registered loopback mailbox credentials')
        self.mail=mail or self._http;self.query_manifest=query_manifest
    def _measure(self,name,operation):
        start=time.perf_counter()
        try:return operation()
        finally:
            elapsed=time.perf_counter()-start
            self.profile[name]=self.profile.get(name,0.)+elapsed
            if self.active_timeline is not None:self.active_timeline['phases'][name]=self.active_timeline['phases'].get(name,0.)+elapsed
    def _http(self,action,body):
        encoded=self._measure('json_encode_seconds',lambda:json.dumps(body,allow_nan=False).encode())
        self.profile['http_request_bytes']=self.profile.get('http_request_bytes',0)+len(encoded)
        request=Request(self.base+'/'+action,data=encoded,headers={'Content-Type':'application/json','Authorization':'Bearer '+self.secret,'X-Meowdoku-Engine-Job':self.job},method='POST')
        start=time.perf_counter()
        try:
            with urlopen(request,context=self.context,timeout=15) as response:
                raw=response.read()
                self.profile['http_response_bytes']=self.profile.get('http_response_bytes',0)+len(raw)
                return self._measure('json_decode_seconds',lambda:json.loads(raw))
        except HTTPError as error:
            try:detail=json.loads(error.read()).get('error',str(error))
            except (ValueError,TypeError):detail=str(error)
            raise MailboxHTTPError(action,error.code,detail) from error
        finally:self.profile['mailbox_seconds']+=time.perf_counter()-start
    def _await_ready(self):
        self.on_state('waiting_client',dict(self.profile))
        while True:
            state=self.mail('ready',{})
            if state.get('ready') is True:return
            if state.get('state') in ('canceled','uncertain'):raise RuntimeError('Client unavailable: '+state['state'])
            time.sleep(.1)
    def step(self,a):
        start=time.perf_counter();self.inner_end=start
        self.active_timeline=None
        if len(self.timeline)<128:
            self.active_timeline={'sequence':self.sequence+1,'phases':{}};self.timeline.append(self.active_timeline)
        else:self.profile['timeline_truncated']=True
        try:
            with ExitStack() as cleanup:return self._step(a,cleanup)
        finally:
            elapsed=time.perf_counter()-start
            self.profile['total_seconds']+=elapsed
            self.profile['cleanup_seconds']=self.profile.get('cleanup_seconds',0.)+time.perf_counter()-self.inner_end
            if self.active_timeline is not None:self.active_timeline['wall_seconds']=elapsed
            self.profile['timeline']=self.timeline.copy()
    def _step(self,a,cleanup):
        from nodus_mail_host import NativeMailHost
        from contract_service_peer import ServicePeer
        start=time.perf_counter();self.sequence+=1;variant='region-bvh' if a._n_regions else 'plain-bvh'
        source=self.renderer._GPU_SENSOR_REGION_CS if a._n_regions else self.renderer._GPU_SENSOR_CS
        # Resume the same specialized transport, including any declared optical field.
        source=getattr(a,'_sensor_shader_source',source)
        base=self.turing/'build/dot-client-ray-20261109'/('studio-region-compute' if a._n_regions else 'studio-compute')
        manifest=self.query_manifest or self._measure('manifest_read_seconds',lambda:json.loads((base/'studio_bvh_query.json').read_text()))
        query_kernel=manifest['query_kernel_sha256']
        loan=self._measure('capture_setup_seconds',lambda:FirstHitLoan(a,source,self.renderer._prog,self.revision,query_variant=variant))
        if self.active_timeline is not None:self.active_timeline['capture_detail']=dict(loan.preparation_profile)
        cleanup.callback(lambda:loan.close() if loan.state not in ('completed','uncertain') else None)
        verifier=self._measure('verifier_setup_seconds',lambda:CompletedFirstHitVerifier(loan,source,self.renderer._prog) if self.verification_mode=='gpu_reference' else AuthorizedParticipantAcceptance(loan))
        cleanup.callback(verifier.close)
        stop=threading.Event();outputs={};mailbox=[];admitted=[];rejected=set();committed=False;finish_sent=False;failed_checks=0
        query_scene_key=(loan.query_identity['query_geometry_hash'],query_kernel)
        if self.scene_hash!=query_scene_key:
            self._measure('scene_registration_seconds',lambda:self.mail('scene',{'scene':loan.scene_descriptor(),'query_kernel':query_kernel}));self.scene_hash=query_scene_key
        self._measure('readiness_seconds',self._await_ready)
        issued=self._measure('lease_issue_seconds',lambda:loan.issue(seconds=60,include_scene=False))
        self.on_state('waiting_client',dict(self.profile))
        contract=(self.sequence,1,1,self.sequence);left,right=socket.socketpair()
        cleanup.callback(left.close);cleanup.callback(right.close)
        def request_work():return None if admitted else contract
        def route(result):
            w=result.loan.work
            if (w.id,w.engine,w.revision,w.payload)!=contract or result.payload not in outputs:return -1
            if result.loan.generation in rejected:return -1
            if not mailbox:mailbox.append(result)
            return 1 if committed else 0
        def compute(offer):
            if (offer.contract,offer.engine,offer.revision,offer.payload)!=contract:raise RuntimeError('Native offer identity mismatch')
            work={**issued,'rays':issued['rays'].tolist()}
            if not publish_retained_when_ready(self.mail,work,query_kernel,loan.deadline,stop):
                handle=offer.generation;outputs[handle]={'state':'expired','result':None};return handle
            while not stop.is_set():
                response=self.mail('take',{})
                if response.get('result') is not None or response.get('state') in ('expired','canceled','uncertain'):
                    handle=offer.generation;outputs[handle]=response;return handle
                stop.wait(.1)
            raise RuntimeError('Cooperative worker stopped; sample not committed')
        host=self._measure('native_setup_seconds',lambda:NativeMailHost(self.native_library or self.nodus/'build/Release/nodus_contract_service.dll',left,1,request_work,route,admitted.append,capacity=8,ticks=1000000))
        cleanup.callback(host.close)
        executor=ThreadPoolExecutor(max_workers=1)
        cleanup.callback(lambda:executor.shutdown(wait=True,cancel_futures=True));cleanup.callback(stop.set)
        peer=ServicePeer(right,executor,compute,lambda *args:None);cleanup.callback(peer.close)
        exchange_start=time.perf_counter();native_calls=0
        try:
            while True:
                before=time.perf_counter();cpu_before=time.thread_time();host.step();native_elapsed=time.perf_counter()-before
                self.profile['native_step_seconds']+=native_elapsed
                key='native_first_steps_seconds' if native_calls==0 else 'native_later_steps_seconds'
                self.profile[key]=self.profile.get(key,0.)+native_elapsed
                self.profile['native_calling_thread_cpu_seconds']=self.profile.get('native_calling_thread_cpu_seconds',0.)+time.thread_time()-cpu_before
                self.profile['native_step_calls']=self.profile.get('native_step_calls',0)+1;native_calls+=1
                peer.poll()
                if host.errors:raise host.errors[0]
                if peer.error:raise peer.error
                if mailbox and not committed:
                    current=mailbox[0];response=outputs[current.payload];result=response.get('result')
                    if response.get('state') in ('canceled','uncertain'):raise RuntimeError('Client participation '+response['state']+'; no server fallback')
                    try:
                        if result is None:raise ValueError('Lease expired')
                        for key in ('token','revision','sample_prefix','geometry_hash','query_geometry_hash','kernel'):
                            if result.get(key)!=issued[key]:raise ValueError('Returned work identity changed')
                        if result.get('query_kernel')!=query_kernel:raise ValueError('Returned query law changed')
                        self.on_state('validating',dict(self.profile))
                        self.profile['completed_batches']+=1
                        duration=client_time_ms(result.get('profile'))
                        if duration is not None:
                            self.profile['client_reported_compute_ms']+=duration;self.profile['client_reported_batches']+=1
                        self._measure('completion_seconds',lambda:loan.complete(issued['token'],issued['revision'],np.asarray(result['hits'],np.float32),verifier))
                        committed=True;self.profile['committed_rays']+=loan.count
                        self.profile['verified_near_ties']+=getattr(verifier,'verified_near_ties',0)
                        if self.on_verified:self.on_verified(issued,result,verifier)
                    except ValueError as error:
                        if result is not None:
                            failed_checks+=1
                            self.profile['last_rejection']=str(error)
                            self.profile['last_validation']=getattr(verifier,'last_mismatch',{})
                            if self.on_reject:self.on_reject(issued,result,verifier)
                        if result is not None:self.mail('finish',{'outcome':'rejected'})
                        if failed_checks>=3:raise RuntimeError('Client validation repeatedly failed; sample not committed: '+str(error))
                        rejected.add(current.loan.generation);mailbox.clear();outputs.pop(current.payload,None)
                        loan.deadline=0;self._await_ready();issued=loan.issue(seconds=60,include_scene=False);self.profile['reoffers']+=1
                        self.on_state('waiting_client',dict(self.profile))
                if committed and host.retire(contract[0]):
                    self.mail('finish',{'outcome':'committed'});finish_sent=True;break
                time.sleep(.001)
            return self.profile
        except Exception:
            if loan.state=='uncertain' or committed:
                # Do not retry an uncertain or already deposited estimator batch.
                if not finish_sent:
                    try:self.mail('finish',{'outcome':'uncertain'})
                    except Exception:pass
            raise
        finally:
            stop.set()
            if hasattr(host.lib,'nodus_mail_timings'):
                import ctypes as C
                getter=host.lib.nodus_mail_timings;getter.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_size_t];getter.restype=C.c_int
                values=(C.c_double*6)()
                if getter(host.handle,values,6):
                    for name,value in zip(('step','request','route','admitted','socket','step_count'),values):
                        key='native_internal_'+name;self.profile[key]=self.profile.get(key,0)+value
            self.profile['freeze_check_seconds']=self.profile.get('freeze_check_seconds',0.)+loan.freeze_check_seconds
            self.profile['freeze_checks']=self.profile.get('freeze_checks',0)+loan.freeze_checks
            self.profile['structural_checked_rays']=self.profile.get('structural_checked_rays',0)+getattr(verifier,'structural_rays',0)
            self.profile['structural_check_seconds']=self.profile.get('structural_check_seconds',0.)+getattr(verifier,'structural_seconds',0.)
            self.profile['verified_rays']+=verifier.checked_rays;self.profile['verification_seconds']+=verifier.seconds
            self.profile['exchange_wall_seconds']=self.profile.get('exchange_wall_seconds',0.)+time.perf_counter()-exchange_start
            self.inner_end=time.perf_counter()
            self.on_state('committed' if committed else 'stopped',dict(self.profile))

    def close(self):
        for directory in self.dirs:directory.close()
        self.dirs=[]
