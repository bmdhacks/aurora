#include "pipeline.hpp"

#include "../webgpu/gpu.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora::gx {
static Module Log("aurora::gx");

// Build the hardware vertex-attribute layout for a native-fetch pipeline.
//
// The attribute ORDER here must stay identical to the shader's native input
// emission (shader.cpp addNativeInput) and the CPU expansion order
// (command_processor build_native_layout): canonical GX attr order, one
// @location per present attr, offset taken from the CPU-expanded stride. Any
// divergence silently scrambles attributes.
static wgpu::VertexBufferLayout native_vertex_layout(const ShaderConfig& sc,
                                                     std::array<wgpu::VertexAttribute, MaxVtxAttr>& attrs) {
  uint32_t count = 0;
  const auto add = [&](GXAttr attr, wgpu::VertexFormat format) {
    if (sc.attrs[attr].attrType == GX_NONE) {
      return;
    }
    attrs[count] = wgpu::VertexAttribute{
        .format = format,
        .offset = sc.attrs[attr].offset,
        .shaderLocation = count,
    };
    ++count;
  };
  add(GX_VA_PNMTXIDX, wgpu::VertexFormat::Uint32);
  for (int i = GX_VA_TEX0MTXIDX; i <= GX_VA_TEX7MTXIDX; ++i) {
    add(static_cast<GXAttr>(i), wgpu::VertexFormat::Uint32);
  }
  add(GX_VA_POS, wgpu::VertexFormat::Float32x3);
  add(GX_VA_NRM, wgpu::VertexFormat::Float32x3);
  add(GX_VA_CLR0, wgpu::VertexFormat::Unorm8x4);
  add(GX_VA_CLR1, wgpu::VertexFormat::Unorm8x4);
  for (int i = GX_VA_TEX0; i <= GX_VA_TEX7; ++i) {
    add(static_cast<GXAttr>(i), wgpu::VertexFormat::Float32x2);
  }
  return wgpu::VertexBufferLayout{
      .stepMode = wgpu::VertexStepMode::Vertex,
      .arrayStride = sc.vtxStride,
      .attributeCount = count,
      .attributes = attrs.data(),
  };
}

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto shader = build_shader(config.shaderConfig);
  const auto label = fmt::format("GX Pipeline {:x} shader {:x}",
                                 xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)),
                                 xxh3_hash(config.shaderConfig));
  if (config.shaderConfig.nativeVertexFetch) {
    std::array<wgpu::VertexAttribute, MaxVtxAttr> attrs{};
    const std::array vtxBuffers{native_vertex_layout(config.shaderConfig, attrs)};
    return build_pipeline(config, vtxBuffers, shader, label.c_str());
  }
  return build_pipeline(config, {}, shader, label.c_str());
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }

  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, gfx::g_uniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
  }
  if (data.nativeVertexFetch) {
    // Native draws read from a real vertex buffer instead of the group-0 SSBOs. Cached
    // geometry lives in the persistent cross-frame cache buffer; everything else in the
    // shared per-frame vertex ring.
    const wgpu::Buffer& vtxBuffer = data.cachedGeometry ? gfx::g_nativeVertexCacheBuffer : gfx::g_vertexBuffer;
    pass.SetVertexBuffer(0, vtxBuffer, data.vertRange.offset, data.vertRange.size);
  }
  const wgpu::Buffer& idxBuffer = data.cachedGeometry ? gfx::g_nativeIndexCacheBuffer : gfx::g_indexBuffer;
  pass.SetIndexBuffer(idxBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  if (data.indexCount == 0) {
    pass.Draw(data.vtxCount, data.instanceCount);
  } else {
    pass.DrawIndexed(data.indexCount, data.instanceCount);
  }
}
} // namespace aurora::gx
