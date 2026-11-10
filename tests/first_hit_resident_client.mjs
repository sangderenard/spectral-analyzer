// Isolated acceptance transport: resident original compiled traversal, one scene upload.
import {readFile} from 'node:fs/promises';
import {resolve,sep} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createInterface} from 'node:readline';
const root=resolve(process.argv[2]);
const {StudioTraversalCompute}=await import(pathToFileURL(resolve(process.argv[3])));
globalThis.fetch=async url=>{const path=resolve(fileURLToPath(new URL(url)));if(!path.startsWith(root+sep))throw Error('Artifact outside selected directory');const b=await readFile(path);return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
const backend=await StudioTraversalCompute.create(pathToFileURL(root+sep));let sceneKey;
for await(const line of createInterface({input:process.stdin,crlfDelay:Infinity})){
 try{const request=JSON.parse(line);if(request.query_kernel!==backend.manifest.query_kernel_sha256)throw Error('Query identity mismatch');const key=request.query_geometry_hash+':'+request.query_kernel;
 if(sceneKey!==key){if(!request.triangles)throw Error('Missing resident geometry');backend.setScene({...request,revision:key});sceneKey=key;}
 const rays=new Float32Array(request.rays.flat()),hits=new Float32Array(request.rays.length*4);let elapsedMs=0;for(let start=0;start<request.rays.length;start+=4096){const result=await backend.query(rays.subarray(start*8,Math.min(start+4096,request.rays.length)*8),key);hits.set(result.hits,start*4);elapsedMs+=result.elapsedMs;}process.stdout.write(JSON.stringify({hits:Array.from(hits),elapsedMs,backend:'wasm'})+'\n');}
 catch(error){process.stdout.write(JSON.stringify({error:error.message})+'\n');}
}
backend.close();
