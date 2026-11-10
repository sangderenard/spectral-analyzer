// Pure compiled-law fixture participant. No engine mutation or queue policy.
import {readFileSync} from 'node:fs';
let input='';for await(const chunk of process.stdin)input+=chunk;
const {request,wasm}=JSON.parse(input);
const {instance}=await WebAssembly.instantiate(readFileSync(wasm));
const hits=[];
for(const ray of request.rays){
 let best=ray[7],owner=-1,index=-1;
 for(const [i,t] of request.triangles.entries()){
  instance.exports.hit_tri(...ray.slice(0,3),...ray.slice(4,7),...t.v0,...t.e1,...t.e2,16,24);
  const valid=Number(new BigInt64Array(instance.exports.memory.buffer,16,1)[0]);
  const d=new Float64Array(instance.exports.memory.buffer,24,1)[0];
  if(valid&&d>ray[3]&&d<best){best=d;owner=t.owner;index=i;}
 }
 hits.push([index<0?0:best,owner,index<0?0:1,index]);
}
console.log(JSON.stringify({hits}));
