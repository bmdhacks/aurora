#include "pass.hpp"

// Phase 1 stubs. Each method records the state WebGPU carried implicitly; the GL
// emission (state-cache application, glBindBufferRange, VAO setup, glDraw*) lands
// in Phases 2-3. Keeping them as tracking stubs lets the Phase 1 cutover build and
// boot to a black frame with the render path wired but inert.

namespace aurora::gl {

void PassEncoder::SetPipeline(const Pipeline& pipeline) { m_pipeline = &pipeline; }

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
  // Phase 2: glViewport with y flipped to bottom-left origin (S1b) + glDepthRangef.
  (void)x;
  (void)y;
  (void)width;
  (void)height;
  (void)minDepth;
  (void)maxDepth;
}

void PassEncoder::SetScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
  // Phase 2: glScissor with y flipped to bottom-left origin (S1b).
  (void)x;
  (void)y;
  (void)width;
  (void)height;
}

void PassEncoder::SetBlendConstant(const Color* color) {
  // Phase 3: glBlendColor.
  (void)color;
}

void PassEncoder::SetStencilReference(uint32_t reference) { m_stencilRef = reference; }

void PassEncoder::Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
  // Phase 3: glDrawArrays / glDrawArraysInstanced.
  (void)vertexCount;
  (void)instanceCount;
  (void)firstVertex;
  (void)firstInstance;
}

void PassEncoder::DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex,
                              uint32_t firstInstance) {
  // Phase 3: glDrawElements(Instanced) with primitive restart per pipeline state.
  (void)indexCount;
  (void)instanceCount;
  (void)firstIndex;
  (void)baseVertex;
  (void)firstInstance;
}

void PassEncoder::End() {
  // Phase 2: StoreOp::Discard -> glInvalidateFramebuffer (S7).
}

void PassEncoder::PushDebugGroup(const char* label) { (void)label; }
void PassEncoder::PopDebugGroup() {}
void PassEncoder::InsertDebugMarker(const char* label) { (void)label; }

} // namespace aurora::gl
