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

#include "../gfx/render_worker.hpp"
#include "../internal.hpp"
#include "../window.hpp"
#include "context.hpp"
#include "fbo_cache.hpp"
#include "gl_core.hpp"
#include "program.hpp"
#include "state.hpp"
#include "textures.hpp"

#include <aurora/gfx.h>
#include <SDL3/SDL_video.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

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
// True once the EFB color/depth render targets exist. Until then resize_swapchain only
// records the surface size; gfx::initialize brings the worker online and creates them.
static bool g_targetsInitialized = false;
// The default framebuffer / window drawable size in pixels (SDL_GetWindowSizeInPixels).
// This is the *present* destination and is distinct from surfaceConfiguration, which holds the
// (possibly internalResolutionScale-downscaled) EFB render size -- e.g. 1216x896 drawable vs a
// 608x448 EFB at scale 0.5. present_frame must blit into the native size, not the render size.
static uint32_t g_presentWidth = 0;
static uint32_t g_presentHeight = 0;

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
  g_presentWidth = size.native_fb_width;
  g_presentHeight = size.native_fb_height;
  g_graphicsConfig.depthFormat = gl::TextureFormat::Depth32Float;
  g_graphicsConfig.msaaSamples = 1;

  gl::make_none_current();

  Log.info("[gl] backend up: {}x{} RGBA8 (bc={}, astc={}, aniso={})", size.fb_width, size.fb_height,
           g_bcTexturesSupported, g_astcTexturesSupported, g_graphicsConfig.textureAnisotropy);
  return true;
}

void shutdown() {
  // The render targets are GL objects owned by the context; destroy_contexts frees them.
  g_frameBuffer = {};
  g_frameBufferResolved = {};
  g_depthBuffer = {};
  g_targetsInitialized = false;
  gl::destroy_contexts();
  g_sdlWindow = nullptr;
}

void release_surface() noexcept {}

bool refresh_surface(bool recreate) {
  (void)recreate;
  return true;
}

TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled) {
  (void)multisampled; // MSAA > 1 is out of scope (config clamps to 1); single-sample only.
  const gl::Extent3D size{width, height, 1};
  const auto format = g_graphicsConfig.surfaceConfiguration.format;
  TextureWithSampler result{};
  // GL objects must be created with the render context current. RmlUi's ensure_render_target calls
  // this from the recording thread, so marshal the create to the worker and block (like
  // gfx::create_gl_texture) rather than returning an empty handle.
  const auto build = [&] {
    gl::Texture texture = gl::create_texture(format, size, 1, /*renderable=*/true);
    // create_texture binds GL_TEXTURE_2D directly; drop the state cache's texture-unit shadow.
    gl::invalidate_texture_bindings();
    gl::SamplerDescriptor samplerDesc{};
    samplerDesc.magFilter = gl::FilterMode::Linear;
    samplerDesc.minFilter = gl::FilterMode::Linear;
    gl::Sampler sampler = gl::create_sampler(samplerDesc, g_graphicsConfig.textureAnisotropy > 0);
    result = TextureWithSampler{
        .texture = texture,
        .view = texture, // WebGPU's separate view collapses into the texture on GL
        .size = size,
        .format = format,
        .sampler = sampler,
    };
  };
  if (!gfx::render_worker::is_running() || gfx::render_worker::is_worker_thread()) {
    build();
  } else {
    gfx::render_worker::enqueue_work(build);
    gfx::render_worker::synchronize();
  }
  return result;
}

static TextureWithSampler create_depth_texture(uint32_t width, uint32_t height) {
  const gl::Extent3D size{width, height, 1};
  const auto format = g_graphicsConfig.depthFormat;
  gl::Texture texture = gl::create_texture(format, size, 1, /*renderable=*/true);
  return TextureWithSampler{
      .texture = texture,
      .view = texture,
      .size = size,
      .format = format,
  };
}

static void destroy_render_targets() {
  // Requires the context current. FBO cache entries reference these attachments; drop
  // them first so a recycled texture name can't alias a stale FBO.
  gl::clear_framebuffer_cache();
  for (TextureWithSampler* t : {&g_frameBuffer, &g_frameBufferResolved, &g_depthBuffer}) {
    gl::destroy_texture(t->texture);
    gl::destroy_sampler(t->sampler);
    *t = {};
  }
}

// (Re)create the EFB color + depth targets at the configured surface size. Must run on
// the render worker (context current): gfx::initialize calls it once the worker is up,
// and resize_swapchain marshals it here on later window resizes.
void create_frame_targets() {
  const uint32_t width = g_graphicsConfig.surfaceConfiguration.width;
  const uint32_t height = g_graphicsConfig.surfaceConfiguration.height;
  if (width == 0 || height == 0) {
    return;
  }
  destroy_render_targets();
  g_frameBuffer = create_render_texture(width, height, g_graphicsConfig.msaaSamples > 1);
  if (g_graphicsConfig.msaaSamples > 1) {
    g_frameBufferResolved = create_render_texture(width, height, false);
  }
  g_depthBuffer = create_depth_texture(width, height);
  g_targetsInitialized = true;
  Log.info("[gl] render targets {}x{} (color RGBA8, depth {})", width, height,
           static_cast<int>(g_graphicsConfig.depthFormat));
}

void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force) {
  if (width == 0 || height == 0) {
    return;
  }
  if (nativeWidth > 0 && nativeHeight > 0) {
    g_presentWidth = nativeWidth;
    g_presentHeight = nativeHeight;
  }
  const bool sizeChanged =
      g_graphicsConfig.surfaceConfiguration.width != width || g_graphicsConfig.surfaceConfiguration.height != height;
  g_graphicsConfig.surfaceConfiguration.width = width;
  g_graphicsConfig.surfaceConfiguration.height = height;

  // The first creation is gfx::initialize's job (it owns the worker-online moment). Early
  // window-init resize calls arrive before the worker exists; just record the size.
  if (!g_targetsInitialized) {
    return;
  }
  if (!force && !sizeChanged) {
    return;
  }
  if (gfx::render_worker::is_worker_thread()) {
    create_frame_targets();
  } else {
    gfx::render_worker::enqueue_work([] { create_frame_targets(); });
    gfx::render_worker::synchronize();
  }
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

// Read the window's default-framebuffer back buffer and write it as a binary PPM. Must run
// on the render worker (context current), before SDL_GL_SwapWindow makes the back buffer
// undefined. This is the Phase-2 DoD headless A/B tool (Phase 5 wires it to F12 / console).
void screenshot(const char* path) noexcept {
  const uint32_t w = g_presentWidth > 0 ? g_presentWidth : g_graphicsConfig.surfaceConfiguration.width;
  const uint32_t h = g_presentHeight > 0 ? g_presentHeight : g_graphicsConfig.surfaceConfiguration.height;
  if (w == 0 || h == 0) {
    return;
  }
  gl::gl.BindFramebuffer(gl::GL_READ_FRAMEBUFFER, 0);
  gl::gl.ReadBuffer(gl::GL_BACK);
  gl::gl.PixelStorei(gl::GL_PACK_ALIGNMENT, 1);
  std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
  gl::gl.ReadPixels(0, 0, static_cast<gl::GLsizei>(w), static_cast<gl::GLsizei>(h), gl::GL_RGBA, gl::GL_UNSIGNED_BYTE,
                    pixels.data());
  gl::reset_state_cache();

  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) {
    Log.warn("[gl] screenshot: cannot open {}", path);
    return;
  }
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  // GL is bottom-left origin; PPM is top-left. Flip rows and drop the alpha channel.
  std::vector<uint8_t> row(static_cast<size_t>(w) * 3);
  for (uint32_t y = 0; y < h; ++y) {
    const uint8_t* src = pixels.data() + static_cast<size_t>(h - 1 - y) * w * 4;
    for (uint32_t x = 0; x < w; ++x) {
      row[x * 3 + 0] = src[x * 4 + 0];
      row[x * 3 + 1] = src[x * 4 + 1];
      row[x * 3 + 2] = src[x * 4 + 2];
    }
    std::fwrite(row.data(), 1, row.size(), f);
  }
  std::fclose(f);
  Log.info("[gl] screenshot written: {} ({}x{})", path, w, h);
}

// Native drawable size (window pixels), NOT the EFB render size, for the present rects.
uint32_t present_surface_width() noexcept {
  return g_presentWidth > 0 ? g_presentWidth : g_graphicsConfig.surfaceConfiguration.width;
}
uint32_t present_surface_height() noexcept {
  return g_presentHeight > 0 ? g_presentHeight : g_graphicsConfig.surfaceConfiguration.height;
}

// Fullscreen-triangle textured-quad program for the UI overlay composite. Every RmlUi pass now
// preserves orientation (rmlui/pipeline.cpp), so the finished UI target lands GL-native (row 0 =
// content bottom), the same layout as the GX scene target. This composite therefore samples it
// straight, exactly like the scene blit: uv(0,0) at NDC(-1,-1) (window bottom = texel row 0), so
// both put memory row 0 at the content-rect bottom -> upright and mutually aligned.
constexpr char kPresentUiVertex[] = R"(#version 300 es
out vec2 v_uv;
void main() {
  const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  const vec2 uvs[3] = vec2[3](vec2(0.0, 0.0), vec2(2.0, 0.0), vec2(0.0, 2.0));
  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
  v_uv = uvs[gl_VertexID];
}
)";
constexpr char kPresentUiFragment[] = R"(#version 300 es
precision highp float;
uniform sampler2D tex;
in vec2 v_uv;
out vec4 out_color;
void main() { out_color = texture(tex, v_uv); }
)";
gl::GLuint g_presentUiProgram = 0;

// Compile the present-UI program on demand (worker context current).
static gl::GLuint present_ui_program() {
  if (g_presentUiProgram == 0) {
    g_presentUiProgram = gl::compile_program(kPresentUiVertex, kPresentUiFragment, "Present UI Composite");
    if (g_presentUiProgram != 0) {
      gl::gl.UseProgram(g_presentUiProgram);
      const gl::GLint loc = gl::gl.GetUniformLocation(g_presentUiProgram, "tex");
      if (loc >= 0) {
        gl::gl.Uniform1i(loc, 0);
      }
      gl::gl.UseProgram(0);
    }
  }
  return g_presentUiProgram;
}

void present_frame() noexcept {
  // Runs on the render worker (owns the GL context). Desktop present: clear the window's
  // default framebuffer to black (letterbox bars) and blit the finished color target into
  // the centered content rect with a linear filter. No Y-flip -- both the source EFB and
  // the default framebuffer are GL bottom-left origin (S1c). Does NOT swap; the caller
  // composites the UI overlay + imgui, then calls present_swap(). Phase 6 replaces this with
  // the SDL2-shim EFB slot hand-off on the device path.
  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  gl::gl.Disable(gl::GL_SCISSOR_TEST);
  gl::gl.ColorMask(gl::GL_TRUE, gl::GL_TRUE, gl::GL_TRUE, gl::GL_TRUE);
  gl::gl.ClearColor(0.f, 0.f, 0.f, 1.f);
  gl::gl.Clear(gl::GL_COLOR_BUFFER_BIT);

  const auto& source = present_source();
  const uint32_t surfaceWidth = present_surface_width();
  const uint32_t surfaceHeight = present_surface_height();
  if (source.texture.id != 0 && surfaceWidth > 0 && surfaceHeight > 0) {
    const auto viewport = calculate_present_viewport(surfaceWidth, surfaceHeight, source.size.width, source.size.height);
    const gl::GLuint readFbo = gl::get_framebuffer(source.texture);
    gl::gl.BindFramebuffer(gl::GL_READ_FRAMEBUFFER, readFbo);
    gl::gl.BindFramebuffer(gl::GL_DRAW_FRAMEBUFFER, 0);
    const auto dstX0 = static_cast<gl::GLint>(viewport.left);
    const auto dstY0 = static_cast<gl::GLint>(viewport.top);
    const auto dstX1 = dstX0 + static_cast<gl::GLint>(viewport.width);
    const auto dstY1 = dstY0 + static_cast<gl::GLint>(viewport.height);
    gl::gl.BlitFramebuffer(0, 0, static_cast<gl::GLint>(source.size.width), static_cast<gl::GLint>(source.size.height),
                           dstX0, dstY0, dstX1, dstY1, gl::GL_COLOR_BUFFER_BIT, gl::GL_LINEAR);
    gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  }
}

void composite_ui_overlay(const gl::Texture& texture, const gl::Sampler& sampler, bool overlay) noexcept {
  const gl::GLuint program = present_ui_program();
  if (program == 0 || texture.id == 0) {
    return;
  }
  const auto& source = present_source();
  const uint32_t surfaceWidth = present_surface_width();
  const uint32_t surfaceHeight = present_surface_height();
  if (source.texture.id == 0 || surfaceWidth == 0 || surfaceHeight == 0) {
    return;
  }
  // Same centered content rect the scene was blitted into (letterbox preserved).
  const auto viewport = calculate_present_viewport(surfaceWidth, surfaceHeight, source.size.width, source.size.height);

  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  gl::gl.Disable(gl::GL_SCISSOR_TEST);
  gl::gl.ColorMask(gl::GL_TRUE, gl::GL_TRUE, gl::GL_TRUE, gl::GL_TRUE);
  gl::gl.Viewport(static_cast<gl::GLint>(viewport.left), static_cast<gl::GLint>(viewport.top),
                  static_cast<gl::GLsizei>(viewport.width), static_cast<gl::GLsizei>(viewport.height));
  if (overlay) {
    // Premultiplied-alpha blend: UI (src) over the scene already in the framebuffer.
    gl::gl.Enable(gl::GL_BLEND);
    gl::gl.BlendEquationSeparate(gl::GL_FUNC_ADD, gl::GL_FUNC_ADD);
    gl::gl.BlendFuncSeparate(gl::GL_ONE, gl::GL_ONE_MINUS_SRC_ALPHA, gl::GL_ONE, gl::GL_ONE_MINUS_SRC_ALPHA);
  } else {
    // Backdrop case: the UI target already contains the scene, so draw it opaque.
    gl::gl.Disable(gl::GL_BLEND);
  }
  gl::gl.UseProgram(program);
  gl::gl.ActiveTexture(gl::GL_TEXTURE0);
  gl::gl.BindTexture(gl::GL_TEXTURE_2D, texture.id);
  gl::gl.BindSampler(0, sampler.id);
  gl::gl.BindVertexArray(0);
  gl::gl.DrawArrays(gl::GL_TRIANGLES, 0, 3);
  gl::gl.BindSampler(0, 0);
  gl::gl.Disable(gl::GL_BLEND);
}

void present_swap() noexcept {
  // The clear + blit + composite + imgui all touched GL outside the state cache; forget the
  // shadow so the next frame's first pass re-issues everything.
  gl::reset_state_cache();
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
