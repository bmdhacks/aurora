#include "context.hpp"

#include "../internal.hpp"

#include <SDL3/SDL_video.h>

namespace aurora::gl {
namespace {
Module Log("aurora::gl");

ContextMode g_mode = ContextMode::Desktop;

// Desktop (SDL_GL) state
SDL_Window* g_window = nullptr;
SDL_GLContext g_renderCtx = nullptr;
SDL_GLContext g_shareCtx = nullptr;

// Device (raw EGL) state is added in Phase 6 (device bring-up), where this path
// is validated on the Mali/PowerVR hardware.

// SDL_GL_GetProcAddress returns SDL_FunctionPointer (void(*)()); adapt to our
// void*(const char*) loader signature.
void* desktop_get_proc(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

bool create_desktop(const ContextConfig& cfg) {
  g_window = static_cast<SDL_Window*>(cfg.sdlWindow);
  if (g_window == nullptr) {
    Log.error("[gl] desktop context: no SDL window");
    return false;
  }
  // ES 3.0, offscreen rendering (we never draw to the window's own buffers --
  // present is an explicit blit), so no depth/stencil on the default framebuffer.
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  g_renderCtx = SDL_GL_CreateContext(g_window);
  if (g_renderCtx == nullptr) {
    Log.error("[gl] SDL_GL_CreateContext (render) failed: {}", SDL_GetError());
    return false;
  }
  SDL_GL_MakeCurrent(g_window, g_renderCtx);
  SDL_GL_SetSwapInterval(0);

  // A second context sharing objects with the render context, for the compiler
  // thread. SDL makes a freshly created context current, so restore render after.
  SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
  g_shareCtx = SDL_GL_CreateContext(g_window);
  if (g_shareCtx == nullptr) {
    // Non-fatal: the pipeline cache falls back to threadless (compile on worker).
    Log.warn("[gl] SDL_GL_CreateContext (share) failed: {}; compiler thread disabled", SDL_GetError());
  }
  SDL_GL_MakeCurrent(g_window, g_renderCtx);

  if (!load(desktop_get_proc)) {
    return false;
  }
  Log.info("[gl] desktop context up: {} / {} / {}", reinterpret_cast<const char*>(gl.GetString(GL_VENDOR)),
           reinterpret_cast<const char*>(gl.GetString(GL_RENDERER)),
           reinterpret_cast<const char*>(gl.GetString(GL_VERSION)));
  return true;
}

bool create_device(const ContextConfig& cfg) {
  // TODO(Phase 6): raw-EGL render + share contexts on cfg.eglDisplay, sharing
  // with cfg.shareEglContext, surfaceless with a 1x1 pbuffer fallback. The EGL
  // entry points are already in the proc table; this path is validated on the
  // Mali/PowerVR devices, not the dev box, so it lands with device bring-up.
  (void)cfg;
  Log.error("[gl] sdl2-shim (device) context path lands in Phase 6");
  return false;
}
} // namespace

bool create_contexts(const ContextConfig& cfg) {
  g_mode = cfg.mode;
  switch (cfg.mode) {
  case ContextMode::Desktop:
    return create_desktop(cfg);
  case ContextMode::Sdl2Shim:
    return create_device(cfg);
  }
  return false;
}

bool make_render_current() {
  if (g_mode == ContextMode::Desktop) {
    return SDL_GL_MakeCurrent(g_window, g_renderCtx);
  }
  return false; // Phase 6
}

bool make_share_current() {
  if (g_mode == ContextMode::Desktop) {
    return g_shareCtx != nullptr && SDL_GL_MakeCurrent(g_window, g_shareCtx);
  }
  return false; // Phase 6
}

bool make_none_current() {
  if (g_mode == ContextMode::Desktop) {
    return SDL_GL_MakeCurrent(g_window, nullptr);
  }
  return false; // Phase 6
}

void destroy_contexts() {
  if (g_mode == ContextMode::Desktop) {
    SDL_GL_MakeCurrent(g_window, nullptr);
    if (g_shareCtx != nullptr) {
      SDL_GL_DestroyContext(g_shareCtx);
      g_shareCtx = nullptr;
    }
    if (g_renderCtx != nullptr) {
      SDL_GL_DestroyContext(g_renderCtx);
      g_renderCtx = nullptr;
    }
  }
  // Phase 6: EGL teardown.
}

ContextMode current_mode() { return g_mode; }

} // namespace aurora::gl
