import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
// Offline compiler maintenance only; no shader compiler or JS runtime is a crate dependency.
const dir=path.dirname(fileURLToPath(import.meta.url));
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'pcu-ordered-carrier-'));
const names=['load_indexed','load_readonly_zero','load_mutable_zero','store_indexed'];
let shader=`#version 450
// Frozen direct native load/store calls preserve scalar SSA representations. U32 only.
// Each invocation owns one complete private output word; no packed-tail racing writes.
layout(local_size_x=64) in;
layout(constant_id=0) const uint ELEMENT_BYTES=4u;
layout(constant_id=1) const uint EXTENT=1u;
layout(set=0,binding=0,std430) buffer Bank0 { uint data[]; } bank0;
layout(set=0,binding=1,std430) buffer Bank1 { uint data[]; } bank1;
layout(set=0,binding=2,std430) buffer Bank2 { uint data[]; } bank2;
layout(set=0,binding=3,std430) buffer Bank3 { uint data[]; } bank3;
layout(set=0,binding=4,std430) writeonly buffer Status { uint data[]; } status_data;
`;
for(let bank=0;bank<4;bank++)shader+=`layout(constant_id=${2+bank}) const uint READ_WORDS_${bank}=0u;\nlayout(constant_id=${6+bank}) const uint WRITE_${bank}=0u;\n`;
for(let step=0;step<64;step++)for(let arg=0;arg<3;arg++)shader+=`layout(constant_id=${10+step*3+arg}) const uint S${step}_${arg}=0u;\n`;
shader+=`uint registers[64];
uint private_words[4];
uint word_index;
uint shift_bits;
uint scalar_mask;
uint read_word(uint bank,uint index) {
 if(bank==0u)return bank0.data[index];if(bank==1u)return bank1.data[index];
 if(bank==2u)return bank2.data[index];return bank3.data[index];
}
void write_word(uint bank,uint index,uint bits) {
 if(bank==0u)bank0.data[index]=bits;else if(bank==1u)bank1.data[index]=bits;
 else if(bank==2u)bank2.data[index]=bits;else bank3.data[index]=bits;
}
void load_indexed(uint dst,uint bank,uint source) {
 registers[dst]=(private_words[bank]>>shift_bits)&scalar_mask;
}
void load_readonly_zero(uint dst,uint bank,uint source) {
 uint limb=ELEMENT_BYTES>=4u?word_index%(ELEMENT_BYTES/4u):0u;
 registers[dst]=read_word(bank,limb)&scalar_mask;
}
void load_mutable_zero(uint dst,uint bank,uint source) {
 // Cold admission permits this only for logical extent one, so the current private word
 // is the current scalar limb. Never reload stale original GPU storage after a store.
 registers[dst]=private_words[bank]&scalar_mask;
}
void store_indexed(uint dst,uint bank,uint source) {
 uint mask=scalar_mask<<shift_bits;
 private_words[bank]=(private_words[bank]&~mask)|((registers[source]&scalar_mask)<<shift_bits);
}
void main() {
 word_index=gl_GlobalInvocationID.x;
 uint bytes=ELEMENT_BYTES*EXTENT;
 uint word_extent=bytes/4u+uint(bytes%4u!=0u);
 if(word_index>=word_extent)return;
`;
for(let bank=0;bank<4;bank++)shader+=`private_words[${bank}]=word_index<READ_WORDS_${bank}?read_word(${bank}u,word_index):0u;\n`;
shader+=`uint lanes=ELEMENT_BYTES>=4u?1u:4u/ELEMENT_BYTES;
 scalar_mask=ELEMENT_BYTES>=4u?0xffffffffu:(1u<<(ELEMENT_BYTES*8u))-1u;
 for(uint lane=0u;lane<lanes;lane++) {
  uint logical_index=ELEMENT_BYTES>=4u?word_index/(ELEMENT_BYTES/4u):word_index*lanes+lane;
  if(logical_index>=EXTENT)break;
  shift_bits=ELEMENT_BYTES>=4u?0u:lane*ELEMENT_BYTES*8u;
`;
for(let step=0;step<64;step++)shader+=`${names[step%4]}(S${step}_0,S${step}_1,S${step}_2);\n`;
shader+='}\n';
for(let bank=0;bank<4;bank++)shader+=`if(WRITE_${bank}!=0u)write_word(${bank}u,word_index,private_words[${bank}]);\n`;
shader+='status_data.data[word_index]=0u;\n}\n';
fs.writeFileSync(dir+'/ordered_transport.comp',shader);
const binary=scratch+'/module.spv';
execFileSync('glslangValidator',['-V','--target-env','vulkan1.0','-Od','-o',binary,dir+'/ordered_transport.comp'],{stdio:'inherit'});
execFileSync('spirv-val',['--target-env','vulkan1.0',binary],{stdio:'inherit'});
const b=fs.readFileSync(binary),words=Array.from({length:b.length/4},(_,i)=>b.readUInt32LE(i*4));
const ops=[];for(let o=5;o<words.length;){const count=words[o]>>>16;if(count===0||o+count>words.length)throw Error('malformed instruction');ops.push({o,count,op:words[o]&65535});o+=count;}
const ids=new Map(),specs=new Map(),types=new Map();
for(const{o,count,op}of ops){if(op===5){const bytes=Buffer.alloc((count-2)*4);for(let i=0;i<count-2;i++)bytes.writeUInt32LE(words[o+2+i],i*4);const name=bytes.toString().split('\0')[0].split('(')[0];if(names.includes(name))ids.set(name,words[o+1]);}if(op===54)types.set(words[o+2],words[o+4]);if(op===71&&words[o+2]===1)specs.set(words[o+1],words[o+3]);}
if(ids.size!==4||new Set([...ids.values()].map(id=>types.get(id))).size!==1)throw Error('native signatures');
const calls=ops.filter(({o,op})=>op===57&&[...ids.values()].includes(words[o+3])).map(({o})=>o+3);
if(calls.length!==64)throw Error('call slots');
const defaults=Array(202).fill(0);for(const{o,count,op}of ops)if(op===50){const id=specs.get(words[o+2]);if(count!==4||id===undefined||id>=202||defaults[id]!==0)throw Error('specialization');defaults[id]=o+3;}
if(defaults.some(x=>x===0))throw Error('missing default');
const literal=x=>'0x'+x.toString(16).padStart(8,'0').replace(/(.{4})$/,'_$1');
const array=(name,values,type)=>`#[rustfmt::skip]\npub(super) const ${name}: &[${type}] = &[\n${Array.from({length:Math.ceil(values.length/8)},(_,row)=>'    '+values.slice(row*8,row*8+8).map(type==='u32'?literal:String).join(', ')+',').join('\n')}\n];\n`;
fs.writeFileSync(path.resolve(dir,'../bytecode/bytecode.rs'),'//! Generated offline from adjacent GLSL; see shader/regenerate.mjs.\n'+array('WORDS',words,'u32')+array('FUNCTION_IDS',names.map(n=>ids.get(n)),'u32')+array('CALL_TARGET_OFFSETS',calls,'usize')+array('CALL_ARGUMENT_IDS',calls.flatMap(o=>words.slice(o+1,o+4)),'u32')+array('SPECIALIZATION_OFFSETS',defaults,'usize'));
fs.rmSync(scratch,{recursive:true});
