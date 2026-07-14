#pragma once

#include <aurora/aurora.h>
#include <aurora/math.hpp>

#include "../gl/handles.hpp"
#include "../gl/pass.hpp"
#include "../gl/textures.hpp"
#include "../gl/buffers.hpp"

#include <array>
#include <cstdint>

struct SDL_Window;

// The hand-rolled GLES backend. This interface kept the historical
// `aurora::webgpu` namespace and function names through the Dawn->GL cutover so
// the ~150 call sites did not have to churn; the *types* are now GL
// (gl::Texture/Buffer/Sampler, gl vocabulary) instead of wgpu::. lib/gl/device.cpp
// implements it. (Phase 7 renames the namespace/paths to gl::.)
namespace aurora::webgpu {
struct SurfaceConfiguration {
  gl::TextureFormat format = gl::TextureFormat::RGBA8Unorm;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t presentMode = 0; // vsync hint; GL present uses SDL_GL_SetSwapInterval
};
struct GraphicsConfig {
  SurfaceConfiguration surfaceConfiguration;
  gl::TextureFormat depthFormat = gl::TextureFormat::Depth32Float;
  uint32_t msaaSamples = 1;
  uint16_t textureAnisotropy = 0;
};
// One texture (+ its sampler). WebGPU's separate TextureView collapses into the
// texture on GL, so `view` is just a copy of `texture`; call sites that passed a
// view as a pass attachment now pass the texture handle.
struct TextureWithSampler {
  gl::Texture texture;
  gl::Texture view;
  gl::Extent3D size;
  gl::TextureFormat format = gl::TextureFormat::Undefined;
  gl::Sampler sampler;
};
struct Viewport {
  float left;
  float top;
  float width;
  float height;
  float znear;
  float zfar;

  bool operator==(const Viewport& rhs) const {
    return left == rhs.left && top == rhs.top && width == rhs.width && height == rhs.height && znear == rhs.znear &&
           zfar == rhs.zfar;
  }
  bool operator!=(const Viewport& rhs) const { return !(*this == rhs); }
};

extern AuroraBackend g_backendType; // resolves to OpenGLES on every live path
extern GraphicsConfig g_graphicsConfig;
extern TextureWithSampler g_frameBuffer;
extern TextureWithSampler g_frameBufferResolved;
extern TextureWithSampler g_depthBuffer;
extern bool g_hasCoreFeatures;
extern bool g_bcTexturesSupported;
extern bool g_astcTexturesSupported;
extern bool g_textureComponentSwizzleSupported;
// GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, queried at init (replaces Dawn's
// minUniformBufferOffsetAlignment limit). Feeds gfx::align_uniform.
extern uint32_t g_uniformBufferOffsetAlignment;

bool initialize(AuroraBackend backend, bool allowCpu);
void shutdown();
void release_surface() noexcept;
bool refresh_surface(bool recreate = true);
void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force = false);
TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled);
const TextureWithSampler& present_source() noexcept;
void set_resampler(AuroraSampler sampler) noexcept;
AuroraSampler get_resampler() noexcept;
Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept;
// Present the finished frame. Runs on the render worker (owns the GL context).
// Phase 1: binds the default framebuffer, clears to black and swaps -- the EFB /
// swapchain blit + UI overlay come back in Phases 2/5/6.
void present_frame() noexcept;

} // namespace aurora::webgpu
