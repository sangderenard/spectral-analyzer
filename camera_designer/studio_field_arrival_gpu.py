"""Opt-in bounded full-grid arrival table GLSL, no renderer or GL execution."""
DOMAIN_CAPACITY=4096
EXTRA_DOMAIN_SLOTS=262144
CHUNK_ROWS=1048576
ROW_WORDS=32
DOMAIN_WORDS=36

def resource_plan(grid,bins):
    n=grid**3*bins;rows=n+EXTRA_DOMAIN_SLOTS
    sizes=[min(CHUNK_ROWS,rows-i)*ROW_WORDS*4 for i in range(0,rows,CHUNK_ROWS)]
    return dict(primary_records=n,extra_domain_records=EXTRA_DOMAIN_SLOTS,domain_capacity=DOMAIN_CAPACITY,
        record_capacity=rows,table_chunk_bytes=sizes,domain_bytes=DOMAIN_CAPACITY*DOMAIN_WORDS*4,
        gpu_bytes=sum(sizes)+DOMAIN_CAPACITY*DOMAIN_WORDS*4,
        cpu_admission_bytes=8*1024**3,maximum_field_bytes=2*1024**3)


def table_glsl(grid,bins,divisor):
    plan=resource_plan(grid,bins);count=len(plan['table_chunk_bytes']);wordchunk=CHUNK_ROWS*ROW_WORDS
    out=[]
    for i in range(count):out.append(f'layout(std430,binding={21+i}) coherent buffer ArrivalChunk{i} {{ uint arrival_chunk{i}[]; }};')
    out.append('layout(std430,binding=25) coherent buffer ArrivalDomains { uint arrival_domains[]; };')
    for name,params,expr in [('load','int i','DATA'),('cas','int i,uint a,uint b','atomicCompSwap(DATA,a,b)'),('add','int i,uint a','atomicAdd(DATA,a)'),('bor','int i,uint a','atomicOr(DATA,a)'),('bmin','int i,uint a','atomicMin(DATA,a)'),('bmax','int i,uint a','atomicMax(DATA,a)'),('exchange','int i,uint a','atomicExchange(DATA,a)')]:
        body=[]
        for j in range(count):
            data=f'arrival_chunk{j}[i-{j*wordchunk}]'
            body.append(f'if(i<{(j+1)*wordchunk})return '+expr.replace('DATA',data)+';')
        out.append(f'uint arrival_{name}({params}){{'+''.join(body)+'return 0u;}')
    out.append(r'''
int arrival_medium=-2,arrival_domain_id=-1;
uint arrival_bounce=0u,arrival_root=0xffffffffu,arrival_endpoint=0xffffffffu,arrival_kind=0u;
uint arrival_domain[32];
void arrival_bad(){atomicOr(studio_directional[WORDS],4);}
int arrival_domain_lookup(){
 uint medium=uint(max(arrival_medium+2,0)),h=medium*16777619u;
 for(int i=0;i<32;++i)h=(h^arrival_domain[i])*16777619u;
 for(uint probe=0u;probe<256u;++probe){
  int id=int((h+probe)%4096u),base=id*36;
  uint state=atomicCompSwap(arrival_domains[base],0u,1u);
  if(state==0u){
   arrival_domains[base+1]=medium;
   for(int i=0;i<32;++i)arrival_domains[base+2+i]=arrival_domain[i];
   memoryBarrierBuffer();atomicExchange(arrival_domains[base],2u);return id;
  }
  for(int spin=0;spin<64&&state==1u;++spin){memoryBarrierBuffer();state=atomicAdd(arrival_domains[base],0u);}
  if(state!=2u){arrival_bad();return -1;}
  bool same=arrival_domains[base+1]==medium;
  for(int i=0;i<32&&same;++i)same=arrival_domains[base+2+i]==arrival_domain[i];
  if(same)return id;
 }
 arrival_bad();return -1;
}
int arrival_claim(int slot,uint index,uint domain){
 int base=slot*32;uint state=arrival_cas(base,0u,1u);
 if(state==0u){
  arrival_exchange(base+1,index);arrival_exchange(base+2,domain);
  arrival_exchange(base+25,0xffffffffu);arrival_exchange(base+27,0xffffffffu);
  memoryBarrierBuffer();arrival_exchange(base,2u);return base;
 }
 for(int spin=0;spin<64&&state==1u;++spin){memoryBarrierBuffer();state=arrival_add(base,0u);}
 if(state!=2u){arrival_bad();return -2;}
 return arrival_load(base+1)==index&&arrival_load(base+2)==domain?base:-1;
}
int arrival_slot(uint index,uint domain){
 int found=arrival_claim(int(index),index,domain);if(found>=0||found==-2)return found;
 uint h=(index^domain)*16777619u;
 for(uint probe=0u;probe<256u;++probe){
  found=arrival_claim(PRIMARY+int((h+probe)%262144u),index,domain);
  if(found>=0||found==-2)return found;
 }
 arrival_bad();return -1;
}
void arrival_inc(int i){uint old=arrival_add(i,1u);if(old==0xffffffffu)arrival_bad();}
void arrival_signed(int i,int value){int old=int(arrival_add(i,uint(value)));if((value>0&&old>2147483647-value)||(value<0&&old<(-2147483647-1)-value))arrival_bad();}
void arrival_observe(ivec3 q,vec3 d,vec4 packet,float meters){
 uint index=uint(((q.z*GRID+q.y)*GRID+q.x)*STUDIO_DIRECTION_BIN_COUNT+studio_record_bin);
 if(arrival_domain_id<0)arrival_domain_id=arrival_domain_lookup();if(arrival_domain_id<0)return;
 int base=arrival_slot(index,uint(arrival_domain_id));if(base<0)return;
 int band=studio_band_index(packet.x);
 arrival_bor(base+24,arrival_bounce<32u?(1u<<arrival_bounce):0x80000000u);
 arrival_bmin(base+25,arrival_root);arrival_bmax(base+26,arrival_root);
 arrival_bmin(base+27,arrival_endpoint);arrival_bmax(base+28,arrival_endpoint);arrival_bor(base+29,arrival_kind);
 float scaled=packet.z*studio_record_root_weight*meters*DIVISOR;
 if(arrival_medium < -1||isnan(scaled)||isinf(scaled)||scaled<0.0){arrival_inc(base+21+band);return;}
 arrival_inc(base+15+band);
 if(scaled>=2147483000.0){arrival_bad();return;}
 int w=int(round(scaled));if(w==0)arrival_inc(base+18+band);
 arrival_signed(base+3+band,w);
 arrival_signed(base+6+band*3,int(round(float(w)*d.x)));
 arrival_signed(base+7+band*3,int(round(float(w)*d.y)));
 arrival_signed(base+8+band*3,int(round(float(w)*d.z)));
}
'''.replace('WORDS',str(grid**3*bins*6)).replace('PRIMARY',str(grid**3*bins)).replace('GRID',str(grid)).replace('DIVISOR',str(divisor)+'.0'))
    return '\n'.join(out)
