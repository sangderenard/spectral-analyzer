"""Source-only candidate: one shared terminal-copy epilogue per native bounce.

Not installed by production. No physics, sampling or packet arithmetic is added.
"""
import re


def _mask(source):
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/',lambda m:''.join('\n' if c=='\n' else ' ' for c in m[0]),source)


def _close(text, start, opening, closing):
    depth=0
    for i in range(start,len(text)):
        if text[i]==opening:depth+=1
        elif text[i]==closing:
            depth-=1
            if depth==0:return i
    raise ValueError('unbalanced native source')


_LATCH=r'''
bool studio_terminal_present=false;
uint studio_terminal_kind,studio_terminal_path,studio_terminal_ray,studio_terminal_root;
uint studio_terminal_validity,studio_terminal_termination;
int studio_terminal_bounce,studio_terminal_hit,studio_terminal_medium;
vec4 studio_terminal_packet;
vec3 studio_terminal_ro,studio_terminal_hp,studio_terminal_direction;
float studio_terminal_factor;
void studio_terminal_latch(uint kind,uint path,uint ray,uint root,int bounce,int hit,vec4 packet,vec3 ro,vec3 hp,vec3 direction,int medium,float factor,uint validity,uint termination){
 studio_terminal_present=true;
 studio_terminal_kind=kind;studio_terminal_path=path;studio_terminal_ray=ray;studio_terminal_root=root;
 studio_terminal_bounce=bounce;studio_terminal_hit=hit;studio_terminal_packet=packet;
 studio_terminal_ro=ro;studio_terminal_hp=hp;studio_terminal_direction=direction;
 studio_terminal_medium=medium;studio_terminal_factor=factor;
 studio_terminal_validity=validity;studio_terminal_termination=termination;
}
'''
_EMIT='''if(studio_terminal_present)studio_response_copy(studio_terminal_kind,studio_terminal_path,studio_terminal_ray,studio_terminal_root,studio_terminal_bounce,studio_terminal_hit,studio_terminal_packet,studio_terminal_ro,studio_terminal_hp,studio_terminal_direction,studio_terminal_medium,studio_terminal_factor,studio_terminal_validity,studio_terminal_termination);'''


def single_forward_epilogue(source):
    """Require specialized forward entry, retain every native expression verbatim.

    Source-site global evidence (before packet, active regions, primitive and
    stages) must remain unchanged from latch to epilogue. This candidate is only
    valid for the audited terminal-only control graph; fail on seam drift.
    """
    if 'if (uMode != PASS_FORWARD) return;' not in source:
        raise ValueError('requires explicit prebuilt forward specialization')
    if 'studio_terminal_latch' in source:raise ValueError('epilogue already installed')
    masked=_mask(source)
    main=masked.index('void main() {')
    loop=re.search(r'for \(int bounce = 0; bounce < uMaxBounces; \+\+bounce\) \{',masked[main:])
    if loop is None:raise ValueError('native forward bounce seam changed')
    begin=main+loop.end()-1;end=_close(masked,begin,'{','}')
    body=source[begin+1:end];code=_mask(body)
    # Only loop-owned control targets change. Nested manifold loops stay exact.
    nested=[]
    for m in re.finditer(r'\b(for|while)\s*\(',code):
        op=code.index('(',m.start());cl=_close(code,op,'(',')')
        block=cl+1
        while code[block].isspace():block+=1
        if code[block]!='{':raise ValueError('unaudited unbraced nested loop')
        nested.append((block,_close(code,block,'{','}')))
    if re.search(r'\b(do|switch|return)\b',code):raise ValueError('unaudited native control flow')
    replacements=[];control={'break':0,'continue':0}
    for m in re.finditer(r'\b(break|continue)\s*;',code):
        if any(a<m.start()<b for a,b in nested):continue
        kind=m[1];control[kind]+=1
        replacements.append((m.start(),m.end(),'{studio_bounce_exit='+('1' if kind=='break' else '2')+';break;}'))
    calls=list(re.finditer(r'\bstudio_response_copy\s*\(',code))
    if len(calls)!=9 or control!={'break':9,'continue':3}:
        raise ValueError('audited terminal graph changed')
    for m in calls:
        replacements.append((m.start(),m.start()+len('studio_response_copy'),'studio_terminal_latch'))
    for a,b,value in sorted(replacements,reverse=True):body=body[:a]+value+body[b:]
    replacement='''{
        studio_terminal_present=false;
        int studio_bounce_exit=0;
        do {'''+body+'''\n        } while(false);
        '''+_EMIT+'''
        if(studio_bounce_exit==1)break;
        if(studio_bounce_exit==2)continue;
    }'''
    result=source[:begin]+replacement+source[end+1:]
    at=result.index('void main() {')
    result=result[:at]+_LATCH+'\n'+result[at:]
    return result,dict(version='single-forward-terminal-epilogue-candidate-v1',writer_sites_before=9,writer_sites_after=1,latched_sites=9,control=control,production_enabled=False)
