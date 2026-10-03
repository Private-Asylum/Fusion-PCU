// Actual immutable MLX primitive replay; heterogeneous status is a real sibling.
#include "pcu_mlx/composed.h"
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
static_assert(MLX_VERSION_NUMERIC == 32003, "composition requires exact MLX0.32.3");
namespace {
namespace mx = mlx::core;
struct Composed {
  std::shared_ptr<mx::Primitive> primitive;
  std::vector<mx::Shape> inputs, outputs;
  std::vector<mx::Dtype> output_types;
  mx::Dtype input_type;
  mx::Stream stream;
  std::thread::id thread;
};
template<class F> int contained(F&& function) noexcept {
  try { function(); return 0; }
  catch (const std::exception& error) { pcu::mlx_c::detail::report(error.what()); }
  catch (...) { pcu::mlx_c::detail::report("unknown exception in MLX checked composition"); }
  return 1;
}
bool bounded(uint32_t count) {
  return count && count <= uint32_t(std::numeric_limits<int>::max());
}
}
extern "C" int pcu_mlx_c_composed_new(pcu_mlx_c_composed* owner, mlx_stream stream,
    uint32_t dtype, const uint32_t* input_counts, uint32_t input_count,
    uint32_t logical_count, uint32_t payload_lanes, uint32_t payload_count,
    uint32_t status_lanes, const char* header, const char* body) {
  return contained([&] {
    if (!owner || owner->ctx || !header || !body || !input_counts || dtype<1 || dtype>3
        || !input_count || input_count>4 || !payload_count || payload_count>2
        || !bounded(logical_count) || !bounded(payload_lanes) || !bounded(status_lanes)
        || status_lanes%logical_count || status_lanes/logical_count>32)
      throw std::invalid_argument("invalid bounded checked composition profile");
    const auto s=mlx_stream_get_(stream);
    if(s.device.type!=mx::Device::gpu)
      throw std::invalid_argument("composition requires explicit actual MLX GPU stream");
    const auto carrier=dtype==1?mx::uint8:dtype==2?mx::uint16:mx::uint32;
    std::vector<std::string> input_names,output_names;
    std::vector<mx::Shape> input_shapes,output_shapes;
    std::vector<mx::Dtype> output_types;
    std::vector<mx::array> inputs;
    for(uint32_t i=0;i<input_count;++i) {
      if(!bounded(input_counts[i])) throw std::invalid_argument("invalid exact composition input shape");
      input_names.push_back("input"+std::to_string(i));
      input_shapes.push_back({int(input_counts[i])});
      inputs.push_back(mx::broadcast_to(mx::array(uint32_t(0),carrier),input_shapes.back(),s));
    }
    for(uint32_t i=0;i<payload_count;++i) {
      output_names.push_back("output"+std::to_string(i));
      output_shapes.push_back({int(payload_lanes)});
      output_types.push_back(carrier);
    }
    output_names.push_back("records");
    output_shapes.push_back({int(status_lanes)});
    output_types.push_back(mx::uint32);
    auto kernel=mx::fast::metal_kernel("pcu_mlx_checked_composed",input_names,output_names,body,header);
    auto outputs=kernel(inputs,output_shapes,output_types,
        {int(logical_count),1,1},{1,1,1},{},std::nullopt,false,s);
    if(outputs.size()!=payload_count+1) throw std::runtime_error("composition sibling arity mismatch");
    for(size_t i=0;i<outputs.size();++i) {
      if(!outputs[i].has_primitive() || outputs[i].primitive().stream()!=s
          || outputs[i].dtype()!=output_types[i] || outputs[i].shape()!=output_shapes[i]
          || outputs[i].primitive_ptr()!=outputs[0].primitive_ptr())
        throw std::runtime_error("composition primitive/stream/type/shape mismatch");
    }
    owner->ctx=new Composed{outputs[0].primitive_ptr(),input_shapes,output_shapes,
        output_types,carrier,s,std::this_thread::get_id()};
  });
}
extern "C" int pcu_mlx_c_composed_apply(mlx_array* outputs,
    pcu_mlx_c_composed owner,const mlx_array* inputs,uint32_t input_count) {
  return contained([&] {
    if(!owner.ctx || !outputs || !inputs) throw std::invalid_argument("nil composition argument");
    const auto& state=*static_cast<Composed*>(owner.ctx);
    if(state.thread!=std::this_thread::get_id() || input_count!=state.inputs.size())
      throw std::invalid_argument("composition thread/actual input arity mismatch");
    std::vector<mx::array> values;
    for(uint32_t i=0;i<input_count;++i) {
      const auto& value=mlx_array_get_(inputs[i]);
      if(value.dtype()!=state.input_type || value.shape()!=state.inputs[i])
        throw std::invalid_argument("composition exact physical type/shape mismatch");
      values.push_back(value);
    }
    // No descriptor/source/shape reconstruction, original-owner donation or host materialization.
    auto results=mx::array::make_arrays(state.outputs,state.output_types,state.primitive,values);
    if(results.size()!=state.outputs.size()) throw std::runtime_error("composition replay arity mismatch");
    for(size_t i=0;i<results.size();++i) mlx_array_set_(outputs[i],results[i]);
  });
}
extern "C" int pcu_mlx_c_composed_free(pcu_mlx_c_composed owner) {
  return contained([&] {
    auto* state=static_cast<Composed*>(owner.ctx);
    if(state && state->thread!=std::this_thread::get_id())
      throw std::invalid_argument("composition release on another thread");
    delete state;
  });
}
