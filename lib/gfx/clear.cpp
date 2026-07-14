#include "clear.hpp"

#include "../webgpu/gpu.hpp"
#include "tracy/Tracy.hpp"

namespace aurora::gfx::clear {

using webgpu::g_graphicsConfig;

namespace {
gl::ColorWriteMask clear_write_mask(bool clearColor, bool clearAlpha) {
  auto writeMask = gl::ColorWriteMask::None;
  if (clearColor) {
    writeMask = writeMask | gl::ColorWriteMask::Red | gl::ColorWriteMask::Green | gl::ColorWriteMask::Blue;
  }
  if (clearAlpha) {
    writeMask = writeMask | gl::ColorWriteMask::Alpha;
  }
  return writeMask;
}
} // namespace

// The clear "pipeline" is a fullscreen triangle that writes the blend-constant
// color (Phase 2 ports the GLSL). Here we bake only the fixed-function state;
// `program` stays 0 until Phase 2 compiles the shader on the render worker.
gl::Pipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  gl::BakedState state{};
  // color = blendConstant * 1 + dst * 0 => write the constant, masked by writeMask
  state.blendEnabled = true;
  state.colorOp = gl::BlendOperation::Add;
  state.colorSrc = gl::BlendFactor::Constant;
  state.colorDst = gl::BlendFactor::Zero;
  state.alphaOp = gl::BlendOperation::Add;
  state.alphaSrc = gl::BlendFactor::Constant;
  state.alphaDst = gl::BlendFactor::Zero;
  state.writeMask = clear_write_mask(config.clearColor, config.clearAlpha);
  state.depthTest = true;
  state.depthWrite = config.clearDepth;
  state.depthCompare = gl::CompareFunction::Always;
  state.cull = gl::CullMode::None;
  state.topology = gl::PrimitiveTopology::TriangleList;
  return gl::Pipeline{
      .program = 0,
      .state = state,
      .vertexLayout = 0,
  };
}

void render(const DrawData& data, gl::PassEncoder& pass, const gl::Extent3D& targetSize) {
  if (!bind_pipeline(data.pipeline, pass)) {
    return;
  }

  pass.SetBlendConstant(&data.color);
  pass.SetViewport(0.f, 0.f, static_cast<float>(targetSize.width), static_cast<float>(targetSize.height), data.depth,
                   data.depth);
  pass.SetScissorRect(0, 0, targetSize.width, targetSize.height);
  pass.Draw(3);
}
} // namespace aurora::gfx::clear
