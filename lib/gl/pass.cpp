#include "pass.hpp"

#include "gl_core.hpp"
#include "state.hpp"

// The PassEncoder issues GL directly against the render worker's current context
// (aurora's own command list already provides the WebGPU CommandEncoder ordering).
//
// Phase 2 wires the fixed-function surface that the clear/present paths need:
// pipeline+state application, viewport/scissor (with the S1b top-left -> bottom-left
// Y-flip), blend constant, and attribute-less Draw. The bind-group + vertex/index
// buffer setup and DrawIndexed stay Phase 3 stubs (GX geometry submission); the
// clear draw is attribute-less so it does not need them.

namespace aurora::gl {
namespace {
GLenum topology_gl(PrimitiveTopology topology) {
  switch (topology) {
  case PrimitiveTopology::PointList:
    return GL_POINTS;
  case PrimitiveTopology::LineList:
    return GL_LINES;
  case PrimitiveTopology::LineStrip:
    return GL_LINE_STRIP;
  case PrimitiveTopology::TriangleList:
    return GL_TRIANGLES;
  case PrimitiveTopology::TriangleStrip:
    return GL_TRIANGLE_STRIP;
  }
  return GL_TRIANGLES;
}
} // namespace

void PassEncoder::SetPipeline(const Pipeline& pipeline) {
  m_pipeline = pipeline;
  m_hasPipeline = true;
  use_program(pipeline.program);
  apply_baked_state(pipeline.state);
}

void PassEncoder::SetBindGroup(uint32_t index, const BindingSet& set, size_t dynamicOffsetCount,
                               const uint32_t* dynamicOffsets) {
  // Phase 3: apply texture units + glBindBufferRange with the dynamic offset.
  (void)index;
  (void)set;
  (void)dynamicOffsetCount;
  (void)dynamicOffsets;
}

void PassEncoder::SetVertexBuffer(uint32_t slot, const Buffer& buffer, uint64_t offset, uint64_t size) {
  (void)slot;
  (void)size;
  m_vertexBuffer = buffer;
  m_vertexOffset = offset;
}

void PassEncoder::SetIndexBuffer(const Buffer& buffer, IndexFormat format, uint64_t offset, uint64_t size) {
  (void)size;
  m_indexBuffer = buffer;
  m_indexFormat = format;
  m_indexOffset = offset;
}

void PassEncoder::SetViewport(float x, float y, float width, float height, float minDepth, float maxDepth) {
  // WebGPU/GX viewports are top-left origin; GL is bottom-left. Flip once, here (S1b).
  const auto gx = static_cast<GLint>(x);
  const auto gw = static_cast<GLsizei>(width);
  const auto gh = static_cast<GLsizei>(height);
  const auto gy = static_cast<GLint>(static_cast<float>(m_target.height) - (y + height));
  set_viewport_gl(gx, gy, gw, gh, minDepth, maxDepth);
}

void PassEncoder::SetScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
  const auto gy = static_cast<GLint>(m_target.height - (y + height));
  set_scissor_gl(static_cast<GLint>(x), gy, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
}

void PassEncoder::SetBlendConstant(const Color* color) {
  if (color == nullptr) {
    return;
  }
  set_blend_color(static_cast<float>(color->r), static_cast<float>(color->g), static_cast<float>(color->b),
                  static_cast<float>(color->a));
}

void PassEncoder::SetStencilReference(uint32_t reference) {
  m_stencilRef = reference;
  set_stencil_reference(reference);
}

void PassEncoder::Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
  (void)firstInstance;
  if (!m_hasPipeline || m_pipeline.program == 0 || vertexCount == 0) {
    return;
  }
  const GLenum mode = topology_gl(m_pipeline.state.topology);
  if (instanceCount > 1) {
    gl.DrawArraysInstanced(mode, static_cast<GLint>(firstVertex), static_cast<GLsizei>(vertexCount),
                           static_cast<GLsizei>(instanceCount));
  } else {
    gl.DrawArrays(mode, static_cast<GLint>(firstVertex), static_cast<GLsizei>(vertexCount));
  }
}

void PassEncoder::DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex,
                              uint32_t firstInstance) {
  // Phase 3: glDrawElements(Instanced) over the GX vertex/index buffers + VAO.
  (void)indexCount;
  (void)instanceCount;
  (void)firstIndex;
  (void)baseVertex;
  (void)firstInstance;
}

void PassEncoder::End() {
  // Store-op handling (StoreOp::Discard -> glInvalidateFramebuffer, S7) is applied in
  // gfx::common render() where the pass's per-attachment store ops are known.
}

// KHR_debug markers (glPushDebugGroup source enum + gating) land with the debug-
// output plumbing in a later phase; these are gated behind AURORA_GFX_DEBUG_GROUPS
// at the call sites and stay no-ops for now.
void PassEncoder::PushDebugGroup(const char* label) { (void)label; }
void PassEncoder::PopDebugGroup() {}
void PassEncoder::InsertDebugMarker(const char* label) { (void)label; }

} // namespace aurora::gl
