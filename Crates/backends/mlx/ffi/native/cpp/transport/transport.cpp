// Ordered scalar representation transport; actual execution and storage belong to MLX.
#include "pcu_mlx/transport.h"
#include "reporting/reporting.hpp"
#include "mlx/c/private/mlx.h"
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/version.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
static_assert(MLX_VERSION_NUMERIC == 32003, "transport requires exact MLX0.32.3");
namespace {
namespace mx = mlx::core;
struct Transport {
  std::shared_ptr<mx::Primitive> primitive;
  std::vector<mx::Shape> inputs, outputs;
  mx::Dtype dtype;
  mx::Stream stream;
  std::thread::id thread;
};
template<class F> int checked(F&& function) noexcept {
  try { function(); return 0; }
  catch (const std::exception& error) { pcu::mlx_c::detail::report(error.what()); }
  catch (...) { pcu::mlx_c::detail::report("unknown exception in MLX-owned scalar transport"); }
  return 1;
}
}
extern "C" int pcu_mlx_c_transport_new(pcu_mlx_c_transport* owner, mlx_stream stream,
    uint32_t dtype, const uint32_t* input_counts, uint32_t input_count,
    uint32_t output_lanes, uint32_t output_count, const char* source) {
  return checked([&] {
    if (!owner || owner->ctx || !source || !input_counts || dtype<1 || dtype>3
        || !input_count || input_count>4 || !output_count || output_count>2
        || !output_lanes || output_lanes>uint32_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("invalid bounded scalar transport profile");
    const auto s=mlx_stream_get_(stream);
    if(s.device.type!=mx::Device::gpu)
      throw std::invalid_argument("transport requires explicit MLX GPU stream");
    const auto carrier=dtype==1?mx::uint8:dtype==2?mx::uint16:mx::uint32;
    std::vector<std::string> input_names,output_names;
    std::vector<mx::Shape> input_shapes,output_shapes;
    std::vector<mx::array> inputs;
    for(uint32_t i=0;i<input_count;++i) {
      if(!input_counts[i] || input_counts[i]>uint32_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid exact transport input shape");
      input_names.push_back("input"+std::to_string(i));
      input_shapes.push_back({int(input_counts[i])});
      inputs.push_back(mx::broadcast_to(mx::array(uint32_t(0),carrier),input_shapes.back(),s));
    }
    for(uint32_t i=0;i<output_count;++i) {
      output_names.push_back("output"+std::to_string(i));
      output_shapes.push_back({int(output_lanes)});
    }
    auto kernel=mx::fast::metal_kernel("pcu_mlx_scalar_transport",input_names,output_names,source);
    auto outputs=kernel(inputs,output_shapes,std::vector<mx::Dtype>(output_count,carrier),
        {int(output_lanes),1,1},{1,1,1},{},std::nullopt,false,s);
    if(outputs.size()!=output_count) throw std::runtime_error("transport output arity mismatch");
    for(const auto& output:outputs) {
      if(!output.has_primitive() || output.primitive().stream()!=s || output.dtype()!=carrier
          || output.primitive_ptr()!=outputs[0].primitive_ptr())
        throw std::runtime_error("transport primitive/stream/type mismatch");
    }
    owner->ctx=new Transport{outputs[0].primitive_ptr(),input_shapes,output_shapes,
        carrier,s,std::this_thread::get_id()};
  });
}
extern "C" int pcu_mlx_c_transport_apply(mlx_array* outputs,
    pcu_mlx_c_transport owner,const mlx_array* inputs,uint32_t input_count) {
  return checked([&] {
    if(!owner.ctx || !outputs || !inputs) throw std::invalid_argument("nil transport argument");
    const auto& state=*static_cast<Transport*>(owner.ctx);
    if(state.thread!=std::this_thread::get_id() || input_count!=state.inputs.size())
      throw std::invalid_argument("transport thread/actual arity mismatch");
    std::vector<mx::array> values;
    for(uint32_t i=0;i<input_count;++i) {
      const auto& value=mlx_array_get_(inputs[i]);
      if(value.dtype()!=state.dtype || value.shape()!=state.inputs[i])
        throw std::invalid_argument("transport exact physical dtype/shape mismatch");
      values.push_back(value);
    }
    // Retained primitive replay only. Original immutable inputs retain separate holders;
    // no shape/source construction, fast kernel preparation or original data donation.
    auto results=mx::array::make_arrays(state.outputs,
        std::vector<mx::Dtype>(state.outputs.size(),state.dtype),state.primitive,values);
    for(size_t i=0;i<results.size();++i) mlx_array_set_(outputs[i],results[i]);
  });
}
extern "C" int pcu_mlx_c_transport_free(pcu_mlx_c_transport owner) {
  return checked([&] {
    auto* state=static_cast<Transport*>(owner.ctx);
    if(state && state->thread!=std::this_thread::get_id())
      throw std::invalid_argument("transport release on another thread");
    delete state;
  });
}
