#pragma once

// GL buffer primitives for the hand-rolled backend.
//
// Thin, corpus-normal wrappers: immutable-size buffers created with glBufferData
// and updated in place with glBufferSubData (Normalcy Doctrine rule 4 -- plain
// uploads first; the persistent-map / UNSYNCHRONIZED path is a later, measured
// escalation that will grow here). The per-frame ring / frame-slot bookkeeping
// stays in the device layer (common.cpp); this module owns only the GL object.
//
// Scaffolded beside Dawn: nothing calls this until the Phase 1 cutover replaces
// common.cpp's g_device.CreateBuffer / queue.WriteBuffer.

#include "gl_core.hpp"
#include "handles.hpp"

namespace aurora::gl {

// Create a GL buffer of `size` bytes bound to `target` (GL_ARRAY_BUFFER,
// GL_UNIFORM_BUFFER, ...). `dynamic` picks GL_DYNAMIC_DRAW vs GL_STATIC_DRAW as
// the usage hint. Contents are left undefined. Requires the owning context current.
Buffer create_buffer(GLenum target, uint64_t size, bool dynamic);

// Update a sub-range in place (glBufferSubData). `offset + size` must be within
// the buffer. No implicit-sync avoidance here -- that is the escalation path.
void upload_buffer(const Buffer& buffer, uint64_t offset, const void* data, uint64_t size);

void destroy_buffer(Buffer& buffer) noexcept;

} // namespace aurora::gl
