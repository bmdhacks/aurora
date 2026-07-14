#pragma once

// Redundant-state-elimination cache for the GL backend.
//
// Shadows the current GL fixed-function + binding state and issues a GL call only
// when a value actually changes. This is both a large CPU-submission win (the
// bottleneck on Mali) and Normalcy-Doctrine hygiene: fewer, coalesced state
// changes are what well-behaved GLES engines emit. Single writer -- the render
// worker. Anything that touches GL outside this cache (present blit, imgui/RmlUi
// raw GL, context switch) must be followed by reset_state_cache().
//
// This is the GL re-implementation of the Dawn GL state cache (the crown-jewel
// perf lever from the Dawn-patch era), now in code we own end to end.
//
// Scaffolded beside Dawn: nothing calls this until the Phase 3 draw path.

#include "gl_core.hpp"
#include "handles.hpp"

namespace aurora::gl {

// Forget all shadowed state so the next apply_* re-issues every GL call. Call at
// frame start and after any out-of-band GL.
void reset_state_cache() noexcept;

// Apply a pipeline's baked fixed-function state (blend, depth, raster incl. the
// S1a front-face flip, color mask, polygon offset, primitive restart, stencil
// op/mask), skipping every value that already matches.
void apply_baked_state(const BakedState& state);

void use_program(GLuint program);
void bind_vertex_array(GLuint vao);
// Bind texture + sampler object to unit `unit` (GX slot i == unit i).
void bind_texture_unit(uint32_t unit, GLuint texture, GLuint sampler);
// glBindBufferRange(GL_UNIFORM_BUFFER, index, ...). `index` is the block binding point.
void bind_uniform_range(uint32_t index, GLuint buffer, uint32_t offset, uint32_t size);

// Dynamic per-draw / per-pass state. Coordinates are already final GL values
// (the pass encoder does the S1b top-left->bottom-left Y-flip before calling).
void set_viewport_gl(GLint x, GLint y, GLsizei width, GLsizei height, GLfloat nearDepth, GLfloat farDepth);
void set_scissor_gl(GLint x, GLint y, GLsizei width, GLsizei height);
void set_blend_color(float r, float g, float b, float a);
// Re-issue stencil func with a new reference, reusing the compare+read-mask that
// apply_baked_state last baked.
void set_stencil_reference(uint32_t reference);

} // namespace aurora::gl
