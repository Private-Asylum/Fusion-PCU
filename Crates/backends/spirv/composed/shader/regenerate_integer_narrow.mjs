import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
// Offline maintenance only; preserves existing float and wide integer assets.
const directory=path.dirname(fileURLToPath(import.meta.url));
const repo=path.resolve(directory,'../..');
const original=fs.readFileSync(repo+'/checked_integer/shader/checked_integer.comp','utf8');
const float=fs.readFileSync(repo+'/composed/shader/f32.comp','utf8');
const folder=fs.mkdtempSync(path.join(os.tmpdir(),'pcu-static-composed-integer-'));
const names=['native_nop','native_load','native_store','native_constant','native_add','native_sub','native_mul'];
let header=float.slice(0,float.indexOf('uint policy;')).replace('const uint LANES=1;','const uint ELEMENT_BYTES=1u<<(FORMAT/2u);\nconst uint SIGNED=FORMAT&1u;\nconst uint LANES=ELEMENT_BYTES<4u?4u/ELEMENT_BYTES:1u;');
let arithmetic=original.slice(original.indexOf('uint width()'),original.indexOf('uint load_word'))+original.slice(original.indexOf('void negate'),original.indexOf('uint evaluate'));
let body=original.slice(original.indexOf('    bool left_negative='),original.indexOf('\nvoid main()'));
body=body.replaceAll('OP','operation').replaceAll('CLAMP','range');
arithmetic+='uint evaluate(in uint input_a[16],in uint input_b[16],uint operation,uint range,out uint result[16]) {\nuint left[16],right[16];uint n=width();\nfor(uint i=0u;i<16u;i++){left[i]=input_a[i];right[i]=input_b[i];result[i]=0u;}\n'+body;
const state=`uint registers[64][16];uint private_words[4][16];uint logical_index;uint notice;uint notice_step;bool fatal;
uint read_word(uint bank,uint index){if(bank==0)return bank0.words[index];if(bank==1)return bank1.words[index];if(bank==2)return bank2.words[index];return bank3.words[index];}
void write_word(uint bank,uint index,uint bits){if(bank==0)bank0.words[index]=bits;else if(bank==1)bank1.words[index]=bits;else if(bank==2)bank2.words[index]=bits;else bank3.words[index]=bits;}
uint load_value(uint bank,uint index,uint limb){uint byte=index*ELEMENT_BYTES;uint word=read_word(bank,byte/4u+limb);return ELEMENT_BYTES<4u?(word>>((byte%4u)*8u))&(ELEMENT_BYTES==1u?255u:65535u):word;}
void record_fault(uint code,uint range,uint ordinal){if(code==0)return;bool recovered=range==1&&(code==2||code==3);if(!recovered){notice=code;notice_step=ordinal;fatal=true;}else if(notice==0){notice=code|256u;notice_step=ordinal;}}
`;
const signature='uint dst,uint a,uint b,uint uf,uint range,uint lo,uint hi,uint ordinal';
let functions=`void native_nop(${signature}){}\nvoid native_load(${signature}){if(fatal)return;for(uint limb=0;limb<16u;limb++)registers[dst][limb]=b==0?private_words[a][limb]:(limb<width()?load_value(a,0u,limb):0u);}\nvoid native_store(${signature}){if(fatal)return;for(uint limb=0;limb<16u;limb++)private_words[a][limb]=registers[b][limb];}\nvoid native_constant(${signature}){if(fatal)return;for(uint limb=0;limb<16u;limb++)registers[dst][limb]=limb==0?lo:limb==1?hi:0u;}\n`;
for(const [i,name]of ['native_add','native_sub','native_mul'].entries())functions+=`void ${name}(${signature}){if(fatal)return;uint result[16];uint code=evaluate(registers[a],registers[b],${i}u,range,result);record_fault(code,range,ordinal);if(!fatal)for(uint limb=0;limb<16u;limb++)registers[dst][limb]=result[limb];}\n`;
let main='void main(){uint word_index=gl_GlobalInvocationID.x;if(word_index>=EXTENT/LANES+uint(EXTENT%LANES!=0u))return;uint packed_words[4];for(uint bank=0u;bank<4u;bank++)packed_words[bank]=0u;\nfor(uint lane=0u;lane<LANES;lane++){logical_index=word_index*LANES+lane;if(logical_index>=EXTENT)break;notice=0;notice_step=0;fatal=false;\n';
for(let bank=0;bank<4;bank++)main+=`for(uint limb=0;limb<16u;limb++){uint word=logical_index*ELEMENT_BYTES/4u+limb;private_words[${bank}][limb]=limb<width()&&word<READ_WORDS_${bank}?load_value(${bank},logical_index,limb):0u;}\n`;
for(let step=0;step<64;step++)main+=`${names[step%names.length]}(${Array.from({length:8},(_,arg)=>`S${step}_${arg}`).join(',')});\n`;
main+='status_data.words[2u*logical_index]=notice;status_data.words[2u*logical_index+1u]=notice_step;\n';
for(let bank=0;bank<4;bank++)main+=`if(WRITE_${bank}!=0){if(ELEMENT_BYTES<4u)packed_words[${bank}]|=(private_words[${bank}][0]&mask())<<(lane*ELEMENT_BYTES*8u);else for(uint limb=0;limb<width();limb++)write_word(${bank},logical_index*width()+limb,private_words[${bank}][limb]);}\n`;
main+='}\n';
for(let bank=0;bank<4;bank++)main+=`if(ELEMENT_BYTES<4u&&WRITE_${bank}!=0)write_word(${bank},word_index,packed_words[${bank}]);\n`;
main+='}\n';const shader=path.join(directory,'integer_narrow.comp');fs.writeFileSync(shader,header+state+arithmetic+functions+main);
execFileSync('glslangValidator',['-V','--target-env','vulkan1.0','-Od','-o',folder+'/integer.spv',shader],{stdio:'inherit'});
execFileSync('spirv-val',['--target-env','vulkan1.0',folder+'/integer.spv'],{stdio:'inherit'});

const bytes=fs.readFileSync(folder+'/integer.spv'),words=Array.from({length:bytes.length/4},(_,i)=>bytes.readUInt32LE(i*4));
const ops=[];for(let offset=5;offset<words.length;){const count=words[offset]>>>16;if(count===0||offset+count>words.length)throw Error('malformed module');ops.push({offset,count,op:words[offset]&65535});offset+=count;}
const functionIds=new Map(),types=new Map(),specIds=new Map();
for(const{offset:o,count,op}of ops){
 if(op===5){const text=Buffer.alloc((count-2)*4);for(let i=0;i<count-2;i++)text.writeUInt32LE(words[o+2+i],i*4);const name=text.toString('utf8').split('\0')[0].split('(')[0];if(names.includes(name))functionIds.set(name,words[o+1]);}
 if(op===54)types.set(words[o+2],words[o+4]);
 if(op===71&&words[o+2]===1)specIds.set(words[o+1],words[o+3]);
}
if(functionIds.size!==7||new Set([...functionIds.values()].map(id=>types.get(id))).size!==1)throw Error('native signatures differ');
const calls=ops.filter(({offset:o,op})=>op===57&&[...functionIds.values()].includes(words[o+3])).map(({offset})=>offset+3);
if(calls.length!==64)throw Error('incorrect call count');
const defaults=Array(522).fill(0);for(const{offset:o,count,op}of ops)if(op===50){const id=specIds.get(words[o+2]);if(count!==4||id===undefined||id>=522||defaults[id]!==0)throw Error('unknown specialization');defaults[id]=o+3;}
if(defaults.some(offset=>offset===0))throw Error('missing specialization');
const literal=word=>'0x'+word.toString(16).padStart(8,'0').replace(/(.{4})$/,'_$1');
const array=(name,values,type)=>`#[rustfmt::skip]\npub(super) const ${name}: &[${type}] = &[\n${Array.from({length:Math.ceil(values.length/8)},(_,row)=>'    '+values.slice(row*8,row*8+8).map(type==='u32'?literal:String).join(', ')+',').join('\n')}\n];\n`;
const argumentsIds=calls.flatMap(offset=>words.slice(offset+1,offset+9));
fs.writeFileSync(path.resolve(directory,'../bytecode/integer_narrow.rs'),'//! Generated offline from `integer_narrow.comp`; see `shader/regenerate_integer_narrow.mjs`.\n'+array('WORDS',words,'u32')+array('FUNCTION_IDS',names.map(name=>functionIds.get(name)),'u32')+array('CALL_TARGET_OFFSETS',calls,'usize')+array('CALL_ARGUMENT_IDS',argumentsIds,'u32')+array('SPECIALIZATION_OFFSETS',defaults,'usize'));
console.log('integer-narrow raw SPIRV bytes',bytes.length);fs.rmSync(folder,{recursive:true});
