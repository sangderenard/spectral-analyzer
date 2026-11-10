// Isolated real HTTP/WASM client. Capability arrives on stdin, never argv/logs.
import {readFileSync} from 'node:fs';
let input='';for await(const chunk of process.stdin)input+=chunk;
const {port,credential,job,wasm,kernel}=JSON.parse(input);
async function call(action,value={}) {
 const r=await fetch(`http://127.0.0.1:${port}/api/client_rays/${action}`,{
  method:'POST',headers:{'Content-Type':'application/json','X-Meowdoku-Rays':'1',Authorization:'Bearer '+credential},
  body:JSON.stringify({job,...value})});
 const result=await r.json();if(!r.ok)throw Error(result.error);return result;
}
const {work}=await call('poll');if(work.query_kernel!==kernel)throw Error('Kernel identity mismatch');
const {instance}=await WebAssembly.instantiate(readFileSync(wasm));
const hits=[];
for(const ray of work.rays) {
 let best=ray[7],owner=-1,index=-1;
 for(const [i,t] of work.triangles.entries()) {
  instance.exports.hit_tri(...ray.slice(0,3),...ray.slice(4,7),...t.v0,...t.e1,...t.e2,16,24);
  const valid=Number(new BigInt64Array(instance.exports.memory.buffer,16,1)[0]);
  const distance=new Float64Array(instance.exports.memory.buffer,24,1)[0];
  if(valid&&distance>ray[3]&&distance<best){best=distance;owner=t.owner;index=i;}
 }
 hits.push([index<0?0:best,owner,index<0?0:1,index]);
}
const result=Object.fromEntries(['token','revision','sample_prefix','geometry_hash','kernel','query_kernel'].map(k=>[k,work[k]]));
result.hits=hits;
const response=await call('result',{result});
if(response.state!=='submitted'||response.committed!==false)throw Error('HTTP claimed premature commit');
console.log(JSON.stringify({backend:'wasm-node-http',rays:hits.length,submitted:true,committed:false}));
