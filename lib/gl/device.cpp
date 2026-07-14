// GL backend device layer -- the replacement for lib/webgpu/gpu.cpp.
//
// Implements the historical aurora::webgpu interface (initialize/shutdown, caps,
// g_graphicsConfig, render-target/present helpers) against the hand-rolled GLES
// backend. The old file created a Dawn instance/adapter/device/surface; this one
// creates a GL context (lib/gl/context.cpp) and issues GL directly.
//
// Threading (Normalcy Doctrine rule 3): the context is created on the main thread
// (to load the proc table and query caps), then released. The render worker binds
// it once and owns it for its whole life -- so every GL call (buffer/texture
// creation, draws, present) runs on the worker. present_frame() is invoked from the
// end-frame callback, which the worker executes.

#include "../webgpu/gpu.hpp"

#include "../internal.hpp"
#include "../window.hpp"
#include "context.hpp"
#include "gl_core.hpp"

#include <aurora/gfx.h>
#include <SDL3/SDL_video.h>

#include <cstring>

namespace aurora::webgpu {
namespace {
Module Log("aurora::gl");

SDL_Window* g_sdlWindow = nullptr;
AuroraSampler g_Resampler = SAMPLER_BILINEAR;

bool has_gl_extension(const char* name) {
  const auto* exts = reinterpret_cast<const char*>(gl::gl.GetString(gl::GL_EXTENSIONS));
  if (exts == nullptr) {
    return false;
  }
  return std::strstr(exts, name) != nullptr;
}

void query_caps() {
  // Desktop ES 3.0 core: swizzle (GL_TEXTURE_SWIZZLE_*), sRGB, integer textures
  // and R16I are all guaranteed. Compressed formats and anisotropy are optional.
  g_hasCoreFeatures = true;
  g_textureComponentSwizzleSupported = true;
  g_bcTexturesSupported = has_gl_extension("GL_EXT_texture_compression_s3tc");
  g_astcTexturesSupported = has_gl_extension("GL_KHR_texture_compression_astc_ldr") ||
                            has_gl_extension("GL_OES_texture_compression_astc");

  uint16_t anisotropy = 0;
  if (has_gl_extension("GL_EXT_texture_filter_anisotropic")) {
    gl::GLint maxAniso = 0;
    gl::gl.GetIntegerv(gl::GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
    anisotropy = static_cast<uint16_t>(maxAniso > 0 ? maxAniso : 0);
  }
  g_graphicsConfig.textureAnisotropy = anisotropy;

  gl::GLint uboAlign = 256;
  gl::gl.GetIntegerv(gl::GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uboAlign);
  g_uniformBufferOffsetAlignment = uboAlign > 0 ? static_cast<uint32_t>(uboAlign) : 256u;
}
} // namespace

AuroraBackend g_backendType = BACKEND_OPENGLES;
GraphicsConfig g_graphicsConfig;
TextureWithSampler g_frameBuffer;
TextureWithSampler g_frameBufferResolved;
TextureWithSampler g_depthBuffer;
bool g_hasCoreFeatures = false;
bool g_bcTexturesSupported = false;
bool g_astcTexturesSupported = false;
bool g_textureComponentSwizzleSupported = false;
uint32_t g_uniformBufferOffsetAlignment = 256;

bool initialize(AuroraBackend backend, bool allowCpu) {
  (void)allowCpu;
  if (backend != BACKEND_OPENGLES && backend != BACKEND_WEBGPU && backend != BACKEND_AUTO &&
      backend != BACKEND_VULKAN) {
    Log.warn("Requested backend {} unavailable; the GL backend serves OpenGLES", static_cast<int>(backend));
  }
  g_backendType = BACKEND_OPENGLES;

  g_sdlWindow = window::get_sdl_window();
  if (g_sdlWindow == nullptr) {
    Log.error("[gl] initialize: no SDL window");
    return false;
  }

  // TODO(Phase 6): pick ContextMode::Sdl2Shim when the shim's borrowed-EGL driver
  // is active, feeding the published EGLDisplay/getProc. Desktop is the dev path.
  gl::ContextConfig cfg{
      .mode = gl::ContextMode::Desktop,
      .sdlWindow = g_sdlWindow,
  };
  if (!gl::create_contexts(cfg)) {
    Log.error("[gl] context creation failed");
    return false;
  }

  // The render context is current on this (main) thread right now. Query caps and
  // the UBO alignment, then release it so the render worker can take ownership.
  query_caps();

  const auto size = window::get_window_size();
  g_graphicsConfig.surfaceConfiguration.format = gl::TextureFormat::RGBA8Unorm;
  g_graphicsConfig.surfaceConfiguration.width = size.fb_width;
  g_graphicsConfig.surfaceConfiguration.height = size.fb_height;
  g_graphicsConfig.depthFormat = gl::TextureFormat::Depth32Float;
  g_graphicsConfig.msaaSamples = 1;

  gl::make_none_current();

  Log.info("[gl] backend up: {}x{} RGBA8 (bc={}, astc={}, aniso={})", size.fb_width, size.fb_height,
           g_bcTexturesSupported, g_astcTexturesSupported, g_graphicsConfig.textureAnisotropy);
  return true;
}

void shutdown() {
  g_frameBuffer = {};
  g_frameBufferResolved = {};
  g_depthBuffer = {};
  gl::destroy_contexts();
  g_sdlWindow = nullptr;
}

void release_surface() noexcept {}

bool refresh_surface(bool recreate) {
  (void)recreate;
  return true;
}

void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force) {
  (void)nativeWidth;
  (void)nativeHeight;
  (void)force;
  if (width == 0 || height == 0) {
    return;
  }
  g_graphicsConfig.surfaceConfiguration.width = width;
  g_graphicsConfig.surfaceConfiguration.height = height;
  // Phase 2 recreates g_frameBuffer/g_depthBuffer render targets here (on the worker).
}

TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled) {
  // Phase 2: create the offscreen EFB color/depth targets. Phase 1 presents a
  // cleared default framebuffer, so there is no render target yet.
  (void)width;
  (void)height;
  (void)multisampled;
  return {};
}

const TextureWithSampler& present_source() noexcept {
  return g_graphicsConfig.msaaSamples > 1 ? g_frameBufferResolved : g_frameBuffer;
}

void set_resampler(AuroraSampler sampler) noexcept {
  g_Resampler = (sampler == SAMPLER_AREA || sampler == SAMPLER_BILINEAR) ? sampler : SAMPLER_BILINEAR;
}

AuroraSampler get_resampler() noexcept { return g_Resampler; }

Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept {
  if (surface_width == 0 || surface_height == 0 || content_width == 0 || content_height == 0) {
    return {};
  }
  uint32_t viewport_width = surface_width;
  uint32_t viewport_height = std::min<uint32_t>(
      surface_height, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(viewport_width) *
                                                                               static_cast<double>(content_height) /
                                                                               static_cast<double>(content_width)))));
  if (viewport_height == surface_height) {
    viewport_width = std::min<uint32_t>(
        surface_width, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(viewport_height) *
                                                                                static_cast<double>(content_width) /
                                                                                static_cast<double>(content_height)))));
  }
  return {
      .left = static_cast<float>((surface_width - viewport_width) / 2),
      .top = static_cast<float>((surface_height - viewport_height) / 2),
      .width = static_cast<float>(viewport_width),
      .height = static_cast<float>(viewport_height),
      .znear = 0.f,
      .zfar = 1.f,
  };
}

void present_frame() noexcept {
  // Runs on the render worker (owns the GL context). Phase 1: clear the window's
  // default framebuffer to black and swap. Phase 2 blits the EFB/present source
  // here; Phase 6 hands the frame to the EFB present module on the device path.
  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  gl::gl.Disable(gl::GL_SCISSOR_TEST);
  gl::gl.ClearColor(0.f, 0.f, 0.f, 1.f);
  gl::gl.Clear(gl::GL_COLOR_BUFFER_BIT | gl::GL_DEPTH_BUFFER_BIT);
  if (g_sdlWindow != nullptr) {
    SDL_GL_SwapWindow(g_sdlWindow);
  }
}

} // namespace aurora::webgpu

// Public C API (declared extern "C" in <aurora/gfx.h>). The Dawn build set the
// wgpu surface present mode; the GL backend applies vsync via SDL_GL_SetSwapInterval
// on the render worker (Phase 2). Phase 1 records the intent in the surface config.
void aurora_enable_vsync(const bool enabled) {
  aurora::webgpu::g_graphicsConfig.surfaceConfiguration.presentMode = enabled ? 1u : 0u;
}
