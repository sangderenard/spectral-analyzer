"""Actual Nodus graph/socket/FIFO round trip for the retained sensor fixture."""
import json,os,socket,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def run(request,commit,*,reject_first=False):
    root=Path(__file__).resolve().parents[2];nodus=root/'nodus';turing=root/'turing'
    sys.path.insert(0,str(nodus/'tools'))
    from nodus_mail_host import NativeMailHost
    from contract_service_peer import ServicePeer
    dirs=[os.add_dll_directory(str(nodus/'build/Release')),
          os.add_dll_directory(str(Path(sys.executable).parent/'Lib/site-packages/torch/lib'))]
    a,b=socket.socketpair();admitted=[];mailbox=[];receipts=[];committed=False;outputs={};calls=[];rejected=set()
    # Opaque native references resolve only inside this authorized fixture.
    contract=(101,7,1,11)
    def request_work():return None if admitted else contract
    def route(result):
        nonlocal committed
        w=result.loan.work
        if (w.id,w.engine,w.revision,w.payload)!=contract or result.payload not in outputs:return -1
        if result.loan.generation in rejected:return -1
        if not mailbox:mailbox.append(result)
        return 1 if committed else 0
    def compute(offer):
        if (offer.contract,offer.engine,offer.revision,offer.payload)!=contract:raise ValueError('Unauthorized payload handle')
        calls.append(offer.generation)
        serial={**request,'rays':request['rays'].tolist()}
        variant=request.get('query_variant','plain')
        if variant.endswith('-bvh'):
            base=turing/'build/dot-client-ray-20261109'/('studio-region-compute' if variant=='region-bvh' else 'studio-compute')
            payload={'request':serial,'base':str(base),'runtime':str(turing/'examples/meowdoku/web/studio_bvh_compute.js')}
            script='first_hit_bvh_client.mjs'
        else:
            payload={'request':serial,'wasm':str(turing/'build/dot-client-ray-20261109/studio-compute/studio_hit_tri.wasm')};script='first_hit_wasm_client.mjs'
        run=subprocess.run(['node',str(Path(__file__).with_name(script))],input=json.dumps(payload),capture_output=True,text=True,timeout=25)
        if run.returncode:raise RuntimeError(run.stderr)
        result=json.loads(run.stdout)['hits']
        if reject_first and len(calls)==1:
            if reject_first=='malformed':result.pop() # missing ray must never skip a sample
            else:result[0]=[1,0,1,0] # false but structurally valid
        handle=200+len(calls);outputs[handle]=result;return handle
    host=NativeMailHost(nodus/'build/Release/nodus_contract_service.dll',a,7,request_work,route,admitted.append,capacity=8,ticks=100000)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            peer=ServicePeer(b,executor,compute,lambda loan,accepted:receipts.append(accepted))
            try:
                deadline=time.monotonic()+30
                while time.monotonic()<deadline:
                    host.step();peer.poll()
                    if host.errors:raise host.errors[0]
                    if peer.error:raise peer.error
                    if mailbox and not committed:
                        # Outside Nodus callbacks: only this GL-owner thread commits.
                        try:commit(outputs[mailbox[0].payload]);committed=True
                        except ValueError:
                            if not reject_first or rejected:raise
                            rejected.add(mailbox[0].loan.generation);mailbox.clear()
                    if committed and host.retire(contract[0]):break
                    time.sleep(.001)
                else:raise TimeoutError('Native contract handoff did not commit')
                assert committed and len(calls)==(2 if reject_first else 1)
                assert len(set(calls))==len(calls)
                return {'native_graph':True,'native_result_fifo':True,'service_calls':len(calls),'owner_thread_commit':True,'authoritative_reject_reoffer':bool(rejected)}
            finally:peer.close()
    finally:
        host.close()
        for directory in dirs:directory.close()
