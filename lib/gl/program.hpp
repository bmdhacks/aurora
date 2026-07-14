#pragma once

// Minimal GLSL compile/link helper for the hand-rolled backend.
//
// This is the raw shader-object plumbing (glCreateShader/glCompileShader/
// glLinkProgram + error logging). Phase 2 uses it directly for the two backend-
// owned programs that are not driven by the GX shader emitter -- the fullscreen
// clear triangle and the present blit. Phase 3's program cache (the GX/RmlUi
// PipelineConfig -> program path) builds on the same primitives.
//
// Every entry point issues GL and so must be called with the owning context
// current (the render worker, or the compiler thread's share context).

#include "gl_core.hpp"

namespace aurora::gl {

// Compile a `#version 300 es` vertex+fragment pair and link them into a program.
// `label` is used only for the error log. Returns 0 on any compile/link failure
// (logged), else the linked program name. Attribute locations, if the caller needs
// fixed ones, should be bound via glBindAttribLocation before linking -- pass them
// through `bindAttribs` (name -> location), applied between attach and link.
GLuint compile_program(const char* vertexSource, const char* fragmentSource, const char* label);

} // namespace aurora::gl
