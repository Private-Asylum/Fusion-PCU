//! Actual generic checked integer source; recovery remains an observable error.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedInteger};
macro_rules! source {($name:ident,$op:tt,$($flag:ident),*)=>{
    #[pcu(invocations=N,$(flag($flag),)*crate_path=::pcu_facade)]
    pub fn $name<T:PcuCheckedInteger,const N:usize>(left:&[T],right:&[T],output:&mut[T]){
        let id=pcu::context::global_invocation_id();output[id]=left[id] $op right[id];
    }
};}
source!(add,+,);
source!(sub,-,strict);
source!(mul,*,);
source!(add_clamp,+,clamp_range);
source!(sub_clamp,-,strict,clamp_range);
source!(mul_clamp,*,clamp_range);
#[pcu(invocations=1,flag(clamp_range),crate_path=::pcu_facade)]
pub fn grid_add<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations=N,flag(clamp_range),crate_path=::pcu_facade)]
pub fn scalar_mul<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * *right;
}
