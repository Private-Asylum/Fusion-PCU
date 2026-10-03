import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
// Offline maintenance only. No compiler or Node runtime is a crate dependency.
const directory=path.dirname(fileURLToPath(import.meta.url));
const base=path.resolve(directory,'../../checked_binary/shader');
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'pcu-static-composed-'));
const names=['native_nop','native_load','native_store','native_constant','native_add','native_sub','native_mul','native_div','native_neg','native_relu'];
for(const format of ['f32','f64','low']){
 const original=fs.readFileSync(base+'/'+(format==='f32'?'checked_binary.comp':format==='f64'?'checked_binary_f64.comp':'checked_binary_low.comp'),'utf8');
 let arithmetic=original.slice(original.indexOf('struct Result'),original.indexOf(format==='low'?'// Each invocation owns':'void main()'));
 arithmetic=arithmetic.replaceAll('POLICY','policy');
 const wide=format==='f64',low=format==='low',type=wide?'uvec2':'uint';
 const header=`#version 450
// Cold rewriting replaces direct native function call targets and freezes constants.
// No operation interpreter, instruction buffer or native floating arithmetic.
layout(local_size_x=64) in;
layout(constant_id=0) const uint EXTENT=1;
layout(constant_id=1) const uint FORMAT=0;
${low?original.slice(original.indexOf('const uint WIDTH'),original.indexOf('layout(set=0,binding=0')):'const uint LANES=1;'}
layout(set=0,binding=0,std430) buffer Bank0 { uint words[]; } bank0;
layout(set=0,binding=1,std430) buffer Bank1 { uint words[]; } bank1;
layout(set=0,binding=2,std430) buffer Bank2 { uint words[]; } bank2;
layout(set=0,binding=3,std430) buffer Bank3 { uint words[]; } bank3;
layout(set=0,binding=4,std430) writeonly buffer Status { uint words[]; } status_data;
`;
 let specs='';
 for(let bank=0;bank<4;bank++)specs+=`layout(constant_id=${2+bank}) const uint READ_WORDS_${bank}=0;\nlayout(constant_id=${6+bank}) const uint WRITE_${bank}=0;\n`;
 for(let step=0;step<64;step++)for(let arg=0;arg<8;arg++)specs+=`layout(constant_id=${10+step*8+arg}) const uint S${step}_${arg}=${arg===7?step:0};\n`;
 const state=`uint policy;
${type} registers[64];
uvec2 private_words[4];
uint logical_index;
uint packed_lane;
uint notice;
uint notice_step;
bool fatal;
uint read_word(uint bank,uint index) {
 if(bank==0)return bank0.words[index]; if(bank==1)return bank1.words[index];
 if(bank==2)return bank2.words[index]; return bank3.words[index];
}
void write_word(uint bank,uint index,uint bits) {
 if(bank==0)bank0.words[index]=bits;else if(bank==1)bank1.words[index]=bits;
 else if(bank==2)bank2.words[index]=bits;else bank3.words[index]=bits;
}
void record_fault(uint code,uint range,uint ordinal) {
 if(code==0)return;
 bool recovered=range==1&&(code==2||code==3);
 if(!recovered){notice=code;notice_step=ordinal;fatal=true;}
 else if(notice==0){notice=code|256u;notice_step=ordinal;}
}
`;
 const finite=wide?'((bits.y>>20)&2047u)!=2047u':low?'(bits&(SIGN-1u))<=MAXIMUM':'(bits&0x7fffffffu)<0x7f800000u';
 const subnormal=wide?'(bits.x|(bits.y&0x7fffffffu))!=0&&(bits.y&0x7fffffffu)<0x100000u':low?'(bits&(SIGN-1u))!=0&&(bits&(SIGN-1u))<(1u<<FRACTION)':'(bits&0x7fffffffu)!=0&&(bits&0x7fffffffu)<0x800000u';
 const positive=wide?'(bits.y&0x80000000u)==0&&(bits.x|bits.y)!=0':low?'(bits&SIGN)==0&&(bits&(SIGN-1u))!=0':'(bits&0x80000000u)==0&&(bits&0x7fffffffu)!=0';
 const unary=`bool finite_bits(${type} bits){return ${finite};}
Result unary_bits(${type} bits,bool negate){
 if(!finite_bits(bits))return fault(1);
 if(negate)bits=${wide?'uvec2(bits.x,bits.y^0x80000000u)':low?'bits^SIGN':'bits^0x80000000u'};
 else if(!(${positive}))bits=${wide?'uvec2(0)':'0'};
 return policy==1&&(${subnormal})?Result(bits,2):ok(bits);
}
`;
 const signature='uint dst,uint a,uint b,uint uf,uint range,uint lo,uint hi,uint ordinal';
 let functions=`void finish_result(uint dst,Result result,uint range,uint ordinal){record_fault(result.status,range,ordinal);if(!fatal)registers[dst]=result.bits;}\nvoid native_nop(${signature}){}\n`;
 const load=wide?'b==0?private_words[a]:uvec2(read_word(a,0),read_word(a,1))':low?'b==0?(private_words[a].x>>(packed_lane*WIDTH))&((SIGN<<1)-1u):read_word(a,0)&((SIGN<<1)-1u)':'b==0?private_words[a].x:read_word(a,0)';
 functions+=`void native_load(${signature}){if(fatal)return;registers[dst]=${load};}\n`;
 functions+=`void native_store(${signature}){if(fatal)return;${wide?'private_words[a]=registers[b];':low?'uint shift=packed_lane*WIDTH;uint mask=((SIGN<<1)-1u)<<shift;private_words[a].x=(private_words[a].x&~mask)|(registers[b]<<shift);':'private_words[a].x=registers[b];'}}\n`;
 functions+=`void native_constant(${signature}){if(fatal)return;registers[dst]=${wide?'uvec2(lo,hi)':'lo'};}\n`;
 for(const [name,expression]of [['add','add(registers[a],registers[b],false)'],['sub','add(registers[a],registers[b],true)'],['mul','multiply(registers[a],registers[b])'],['div','divide(registers[a],registers[b])']])functions+=`void native_${name}(${signature}){if(fatal)return;policy=uf;Result result;if(!finite_bits(registers[a])||!finite_bits(registers[b]))result=fault(1);else result=${expression};finish_result(dst,result,range,ordinal);}\n`;
 for(const [name,negate]of [['neg','true'],['relu','false']])functions+=`void native_${name}(${signature}){if(fatal)return;policy=uf;finish_result(dst,unary_bits(registers[a],${negate}),range,ordinal);}\n`;
 let main=`void main(){
 uint word=gl_GlobalInvocationID.x;
 uint word_extent=EXTENT/LANES+uint(EXTENT%LANES!=0);
 if(word>=word_extent)return;
`;
 for(let bank=0;bank<4;bank++)main+=wide?`private_words[${bank}]=word*2u+1u<READ_WORDS_${bank}?uvec2(read_word(${bank},word*2u),read_word(${bank},word*2u+1u)):uvec2(0);\n`:`private_words[${bank}]=uvec2(word<READ_WORDS_${bank}?read_word(${bank},word):0,0);\n`;
 main+='for(packed_lane=0;packed_lane<LANES;packed_lane++){logical_index=word*LANES+packed_lane;if(logical_index>=EXTENT)break;notice=0;notice_step=0;fatal=false;\n';
 for(let step=0;step<64;step++)main+=`${names[step%names.length]}(${Array.from({length:8},(_,arg)=>`S${step}_${arg}`).join(',')});\n`;
 main+='status_data.words[2u*logical_index]=notice;status_data.words[2u*logical_index+1u]=notice_step;}\n';
 for(let bank=0;bank<4;bank++)main+=`if(WRITE_${bank}!=0){write_word(${bank},${wide?'word*2u':'word'},private_words[${bank}].x);${wide?`write_word(${bank},word*2u+1u,private_words[${bank}].y);`:''}}\n`;
 main+='}\n';
 const shader=path.join(directory,format+'.comp');
 fs.writeFileSync(shader,header+specs+state+arithmetic+unary+functions+main);
 const binary=path.join(scratch,format+'.spv');
 execFileSync('glslangValidator',['-V','--target-env','vulkan1.0','-Od','-o',binary,shader],{stdio:'inherit'});
 execFileSync('spirv-val',['--target-env','vulkan1.0',binary],{stdio:'inherit'});
 const bytes=fs.readFileSync(binary),words=Array.from({length:bytes.length/4},(_,i)=>bytes.readUInt32LE(i*4));
 const ops=[];for(let offset=5;offset<words.length;){const count=words[offset]>>>16;if(count===0||offset+count>words.length)throw Error('malformed offline module');ops.push({offset,count,op:words[offset]&65535});offset+=count;}
 const functionIds=new Map(),types=new Map(),specIds=new Map();
 for(const{offset:o,count,op}of ops){
  if(op===5){const text=Buffer.alloc((count-2)*4);for(let i=0;i<count-2;i++)text.writeUInt32LE(words[o+2+i],i*4);const name=text.toString('utf8').split('\0')[0].split('(')[0];if(names.includes(name))functionIds.set(name,words[o+1]);}
  if(op===54)types.set(words[o+2],words[o+4]);
  if(op===71&&words[o+2]===1)specIds.set(words[o+1],words[o+3]);
 }
 if(functionIds.size!==10||new Set([...functionIds.values()].map(id=>types.get(id))).size!==1)throw Error('native call signatures differ');
 const calls=ops.filter(({offset:o,op})=>op===57&&[...functionIds.values()].includes(words[o+3])).map(({offset})=>offset+3);
 if(calls.length!==64)throw Error('incorrect static native call slot count');
 const defaults=Array(522).fill(0);for(const{offset:o,count,op}of ops)if(op===50){const id=specIds.get(words[o+2]);if(count!==4||id===undefined||id>=522||defaults[id]!==0)throw Error('unknown specialization');defaults[id]=o+3;}
 if(defaults.some(offset=>offset===0))throw Error('missing specialization');
 const literal=word=>'0x'+word.toString(16).padStart(8,'0').replace(/(.{4})$/,'_$1');
 const array=(name,values,type)=>`#[rustfmt::skip]\npub(super) const ${name}: &[${type}] = &[\n${Array.from({length:Math.ceil(values.length/8)},(_,row)=>'    '+values.slice(row*8,row*8+8).map(type==='u32'?literal:String).join(', ')+',').join('\n')}\n];\n`;
 const argumentIds=calls.flatMap(offset=>words.slice(offset+1,offset+9));
 fs.writeFileSync(path.resolve(directory,'../bytecode',format+'.rs'),'//! Generated offline from the adjacent auditable GLSL; see shader/regenerate.mjs.\n'+array('WORDS',words,'u32')+array('FUNCTION_IDS',names.map(name=>functionIds.get(name)),'u32')+array('CALL_TARGET_OFFSETS',calls,'usize')+array('CALL_ARGUMENT_IDS',argumentIds,'u32')+array('SPECIALIZATION_OFFSETS',defaults,'usize'));
}
fs.rmSync(scratch,{recursive:true});
