// Isolated native Nodus participant, using the shared compiled BVH runtime.
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
let text='';for await(const chunk of process.stdin)text+=chunk;
const {request,base,runtime}=JSON.parse(text);
globalThis.fetch=async u=>{const b=await readFile(new URL(u));return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
const {StudioBvhCompute}=await import(pathToFileURL(runtime));
const q=await StudioBvhCompute.create(pathToFileURL(base+'/'),{backend:'wasm'});
try{q.setScene(request);const result=await q.query(new Float32Array(request.rays.flat()),request.revision);const hits=[];for(let i=0;i<result.hits.length;i+=4)hits.push(Array.from(result.hits.slice(i,i+4)));process.stdout.write(JSON.stringify({hits,query_kernel:q.manifest.query_kernel_sha256,elapsedMs:result.elapsedMs}));}finally{q.close();}
