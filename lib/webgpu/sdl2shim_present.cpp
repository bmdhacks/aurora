#include "sdl2shim_present.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>

#include <SDL3/SDL_video.h>
#include <magic_enum.hpp>
#include <tracy/Tracy.hpp>

#include "../gfx/common.hpp"
#include "../internal.hpp"
#include "../window.hpp"
#include "gpu.hpp"

#ifdef WEBGPU_DAWN
#include <dawn/native/OpenGLBackend.h>
#endif

namespace aurora::webgpu::sdl2shim_present {
namespace {
static Module Log("aurora::sdl2shim_present");

// Two slots: the worker renders into one while the main thread scans out the other. A third would
// only let the worker run further ahead of the display, which we don't want.
constexpr uint32_t SlotCount = 2;
// How long the main thread will wait for the worker to hand over the frame it owes us. Generous:
// exceeding it means the worker is wedged, and we'd rather log and keep the game responsive.
constexpr auto PresentWaitTimeout = std::chrono::milliseconds{500};

// GL / EGL enumerants. Declared here so aurora needn't include vendor GLES headers.
constexpr uint32_t GL_TEXTURE_2D = 0x0DE1;
constexpr uint32_t GL_RGBA = 0x1908;
constexpr int32_t GL_RGBA8 = 0x8058;
constexpr uint32_t GL_UNSIGNED_BYTE = 0x1401;
constexpr uint32_t GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr uint32_t GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr uint32_t GL_TEXTURE_WRAP_S = 0x2802;
constexpr uint32_t GL_TEXTURE_WRAP_T = 0x2803;
constexpr uint32_t GL_TEXTURE_MAX_LEVEL = 0x813D;
constexpr int32_t GL_NEAREST = 0x2600;
constexpr int32_t GL_LINEAR = 0x2601;
constexpr int32_t GL_CLAMP_TO_EDGE = 0x812F;
constexpr uint32_t GL_READ_FRAMEBUFFER = 0x8CA8;
constexpr uint32_t GL_DRAW_FRAMEBUFFER = 0x8CA9;
constexpr uint32_t GL_FRAMEBUFFER = 0x8D40;
constexpr uint32_t GL_COLOR_ATTACHMENT0 = 0x8CE0;
constexpr uint32_t GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
constexpr uint32_t GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr uint32_t GL_SCISSOR_TEST = 0x0C11;
constexpr uint32_t GL_NO_ERROR = 0;

constexpr uint32_t EGL_NONE = 0x3038;
constexpr uint32_t EGL_GL_TEXTURE_2D_KHR = 0x30B1;
constexpr uint32_t EGL_GL_TEXTURE_LEVEL_KHR = 0x30BC;
constexpr uint32_t EGL_IMAGE_PRESERVED_KHR = 0x30D2;
constexpr uint32_t EGL_SYNC_FENCE_KHR = 0x30F9;
constexpr int32_t EGL_TRUE = 1;

using GlGenTexturesFn = void (*)(int32_t, uint32_t*);
using GlBindTextureFn = void (*)(uint32_t, uint32_t);
using GlTexImage2DFn = void (*)(uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t,
                                const void*);
using GlTexParameteriFn = void (*)(uint32_t, uint32_t, int32_t);
using GlDeleteTexturesFn = void (*)(int32_t, const uint32_t*);
using GlGenFramebuffersFn = void (*)(int32_t, uint32_t*);
using GlBindFramebufferFn = void (*)(uint32_t, uint32_t);
using GlFramebufferTexture2DFn = void (*)(uint32_t, uint32_t, uint32_t, uint32_t, int32_t);
using GlCheckFramebufferStatusFn = uint32_t (*)(uint32_t);
using GlDeleteFramebuffersFn = void (*)(int32_t, const uint32_t*);
using GlBlitFramebufferFn = void (*)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                     uint32_t, uint32_t);
using GlDisableFn = void (*)(uint32_t);
using GlFlushFn = void (*)();
using GlGetErrorFn = uint32_t (*)();

using EglCreateImageKHRFn = void* (*)(void*, void*, uint32_t, void*, const int32_t*);
using EglDestroyImageKHRFn = uint32_t (*)(void*, void*);
using EglCreateSyncKHRFn = void* (*)(void*, uint32_t, const int32_t*);
using EglWaitSyncKHRFn = uint32_t (*)(void*, void*, int32_t);
using EglGetCurrentContextFn = void* (*)();

// The firmware SDL2's own swap entry point, republished by the SDL3 shim. This is the exact call
// the Dawn present path used to reach the display with, so prefer it over routing back out through
// SDL3 -> shim -> SDL2.
using Sdl2SwapWindowFn = void (*)(void*);
constexpr const char* SDL2_SHIM_WINDOW_PROP = "SDL.window.sdl2_backend.window";
constexpr const char* SDL2_SHIM_GL_SWAP_WINDOW_PROP = "SDL.window.sdl2_backend.gl_swap_window";

struct GlFns {
  GlGenTexturesFn GenTextures = nullptr;
  GlBindTextureFn BindTexture = nullptr;
  GlTexImage2DFn TexImage2D = nullptr;
  GlTexParameteriFn TexParameteri = nullptr;
  GlDeleteTexturesFn DeleteTextures = nullptr;
  GlGenFramebuffersFn GenFramebuffers = nullptr;
  GlBindFramebufferFn BindFramebuffer = nullptr;
  GlFramebufferTexture2DFn FramebufferTexture2D = nullptr;
  GlCheckFramebufferStatusFn CheckFramebufferStatus = nullptr;
  GlDeleteFramebuffersFn DeleteFramebuffers = nullptr;
  GlBlitFramebufferFn BlitFramebuffer = nullptr;
  GlDisableFn Disable = nullptr;
  GlFlushFn Flush = nullptr;
  GlGetErrorFn GetError = nullptr;

  EglCreateImageKHRFn CreateImageKHR = nullptr;
  EglDestroyImageKHRFn DestroyImageKHR = nullptr;
  EglCreateSyncKHRFn CreateSyncKHR = nullptr;
  EglWaitSyncKHRFn WaitSyncKHR = nullptr;
  EglGetCurrentContextFn GetCurrentContext = nullptr;

  [[nodiscard]] bool complete() const {
    return GenTextures && BindTexture && TexImage2D && TexParameteri && DeleteTextures && GenFramebuffers &&
           BindFramebuffer && FramebufferTexture2D && CheckFramebufferStatus && DeleteFramebuffers &&
           BlitFramebuffer && Disable && Flush && GetError && CreateImageKHR && DestroyImageKHR &&
           CreateSyncKHR && WaitSyncKHR && GetCurrentContext;
  }
};

enum class SlotState : uint8_t {
  Free,      // the worker may render into it
  Rendering, // the worker owns it
  Ready,     // submitted; the main thread may present it
};

struct Slot {
  uint32_t glTexture = 0;    // lives in the shim context
  uint32_t readFbo = 0;      // lives in the shim context, wraps glTexture
  void* eglImage = nullptr;  // the bridge between the two contexts
  wgpu::Texture texture;     // Dawn's import of eglImage
  wgpu::TextureView view;
  void* fwdSync = nullptr;   // Dawn finished rendering this slot
  void* revSync = nullptr;   // main finished reading this slot
  SlotState state = SlotState::Free;
};

bool g_active = false;
void* g_display = nullptr;
uint32_t g_width = 0;
uint32_t g_height = 0;
GlFns g_gl;
void* g_sdl2Window = nullptr;
Sdl2SwapWindowFn g_sdl2SwapWindow = nullptr;
std::array<Slot, SlotCount> g_slots;

std::mutex g_mutex;
std::condition_variable g_slotFreed;  // main -> worker
std::condition_variable g_frameReady; // worker -> main
uint32_t g_nextSlot = 0;
int32_t g_framesOwed = 0;     // end_frames enqueued but not yet presented/dropped
bool g_hasReadyFrame = false;
uint32_t g_readySlot = 0;
bool g_aborting = false;

WGPUDevice device_handle() { return g_device.Get(); }

void destroy_sync(void*& sync) {
#ifdef WEBGPU_DAWN
  if (sync != nullptr) {
    dawn::native::opengl::PortmasterDestroyEGLSync(device_handle(), sync);
    sync = nullptr;
  }
#endif
}

// Requires the shim context to be current (main thread).
void* create_shim_fence() {
  void* sync = g_gl.CreateSyncKHR(g_display, EGL_SYNC_FENCE_KHR, nullptr);
  if (sync == nullptr) {
    return nullptr;
  }
  // The fence must be flushed before Dawn's context can wait on it.
  g_gl.Flush();
  return sync;
}

bool load_procs(ProcAddressFn getProc) {
#define LOAD(name, member)                                                                                   \
  g_gl.member = reinterpret_cast<decltype(g_gl.member)>(getProc(name));                                      \
  if (g_gl.member == nullptr) {                                                                              \
    Log.warn("[sdl2shim-efb] missing entry point {}", name);                                                 \
  }
  LOAD("glGenTextures", GenTextures)
  LOAD("glBindTexture", BindTexture)
  LOAD("glTexImage2D", TexImage2D)
  LOAD("glTexParameteri", TexParameteri)
  LOAD("glDeleteTextures", DeleteTextures)
  LOAD("glGenFramebuffers", GenFramebuffers)
  LOAD("glBindFramebuffer", BindFramebuffer)
  LOAD("glFramebufferTexture2D", FramebufferTexture2D)
  LOAD("glCheckFramebufferStatus", CheckFramebufferStatus)
  LOAD("glDeleteFramebuffers", DeleteFramebuffers)
  LOAD("glBlitFramebuffer", BlitFramebuffer)
  LOAD("glDisable", Disable)
  LOAD("glFlush", Flush)
  LOAD("glGetError", GetError)
  LOAD("eglCreateImageKHR", CreateImageKHR)
  LOAD("eglDestroyImageKHR", DestroyImageKHR)
  LOAD("eglCreateSyncKHR", CreateSyncKHR)
  LOAD("eglWaitSyncKHR", WaitSyncKHR)
  LOAD("eglGetCurrentContext", GetCurrentContext)
#undef LOAD
  return g_gl.complete();
}

// Allocates the shim-side GL texture + read FBO, bridges it as an EGLImage, and imports it into
// Dawn. Runs on the main thread with the shim context current.
bool create_slot(Slot& slot, void* shimContext, wgpu::TextureFormat format) {
#ifndef WEBGPU_DAWN
  return false;
#else
  g_gl.GenTextures(1, &slot.glTexture);
  g_gl.BindTexture(GL_TEXTURE_2D, slot.glTexture);
  g_gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<int32_t>(g_width), static_cast<int32_t>(g_height),
                  0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  // EGL_KHR_gl_texture_2D_image requires a complete texture; clamp it to the single level we defined.
  g_gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
  g_gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  g_gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  g_gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  g_gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  g_gl.BindTexture(GL_TEXTURE_2D, 0);
  if (const uint32_t err = g_gl.GetError(); err != GL_NO_ERROR) {
    Log.error("[sdl2shim-efb] texture allocation failed (GL error 0x{:x})", err);
    return false;
  }

  const std::array<int32_t, 5> imageAttribs{
      static_cast<int32_t>(EGL_GL_TEXTURE_LEVEL_KHR), 0,
      static_cast<int32_t>(EGL_IMAGE_PRESERVED_KHR),  EGL_TRUE,
      static_cast<int32_t>(EGL_NONE),
  };
  slot.eglImage = g_gl.CreateImageKHR(g_display, shimContext, EGL_GL_TEXTURE_2D_KHR,
                                      reinterpret_cast<void*>(static_cast<uintptr_t>(slot.glTexture)),
                                      imageAttribs.data());
  if (slot.eglImage == nullptr) {
    Log.error("[sdl2shim-efb] eglCreateImageKHR failed");
    return false;
  }

  g_gl.GenFramebuffers(1, &slot.readFbo);
  g_gl.BindFramebuffer(GL_FRAMEBUFFER, slot.readFbo);
  g_gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, slot.glTexture, 0);
  const uint32_t status = g_gl.CheckFramebufferStatus(GL_FRAMEBUFFER);
  g_gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    Log.error("[sdl2shim-efb] read framebuffer incomplete (0x{:x})", status);
    return false;
  }

  const wgpu::TextureDescriptor textureDescriptor{
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = wgpu::Extent3D{.width = g_width, .height = g_height, .depthOrArrayLayers = 1},
      .format = format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  dawn::native::opengl::ExternalImageDescriptorEGLImage imageDescriptor;
  imageDescriptor.cTextureDescriptor = reinterpret_cast<const WGPUTextureDescriptor*>(&textureDescriptor);
  imageDescriptor.isInitialized = false;
  imageDescriptor.image = slot.eglImage;

  WGPUTexture wrapped = dawn::native::opengl::WrapExternalEGLImage(device_handle(), &imageDescriptor);
  if (wrapped == nullptr) {
    Log.error("[sdl2shim-efb] WrapExternalEGLImage failed");
    return false;
  }
  slot.texture = wgpu::Texture::Acquire(wrapped);
  slot.view = slot.texture.CreateView();
  if (!slot.view) {
    Log.error("[sdl2shim-efb] CreateView failed on imported texture");
    return false;
  }
  return true;
#endif
}

void destroy_slot(Slot& slot);

bool create_slots(void* shimContext, wgpu::TextureFormat format) {
  for (uint32_t i = 0; i < SlotCount; ++i) {
    if (!create_slot(g_slots[i], shimContext, format)) {
      for (uint32_t j = 0; j <= i; ++j) {
        destroy_slot(g_slots[j]);
      }
      return false;
    }
    g_slots[i].state = SlotState::Free;
  }
  return true;
}

void destroy_slot(Slot& slot) {
  destroy_sync(slot.fwdSync);
  destroy_sync(slot.revSync);
  slot.view = {};
  slot.texture = {};
  if (slot.eglImage != nullptr && g_gl.DestroyImageKHR != nullptr) {
    g_gl.DestroyImageKHR(g_display, slot.eglImage);
    slot.eglImage = nullptr;
  }
  if (slot.readFbo != 0 && g_gl.DeleteFramebuffers != nullptr) {
    g_gl.DeleteFramebuffers(1, &slot.readFbo);
    slot.readFbo = 0;
  }
  if (slot.glTexture != 0 && g_gl.DeleteTextures != nullptr) {
    g_gl.DeleteTextures(1, &slot.glTexture);
    slot.glTexture = 0;
  }
  slot.state = SlotState::Free;
}

// Blit the finished slot into the window's default framebuffer and swap. Main thread, shim context.
void blit_and_swap(Slot& slot) {
  ZoneScopedN("EFB blit + swap");
  SDL_Window* window = window::get_sdl_window();
  int drawableWidth = static_cast<int>(g_width);
  int drawableHeight = static_cast<int>(g_height);
  SDL_GetWindowSizeInPixels(window, &drawableWidth, &drawableHeight);

  // Wait, on the GPU, for Dawn to finish rendering this slot. This does not block the CPU.
  if (slot.fwdSync != nullptr) {
    g_gl.WaitSyncKHR(g_display, slot.fwdSync, 0);
  }

  g_gl.BindFramebuffer(GL_READ_FRAMEBUFFER, slot.readFbo);
  g_gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  g_gl.Disable(GL_SCISSOR_TEST);
  // Reverse-Y: WebGPU's origin is top-left, the default framebuffer's is bottom-left. Mirrors the
  // blit the Dawn SwapChainEGL present path used to do.
  g_gl.BlitFramebuffer(0, 0, static_cast<int32_t>(g_width), static_cast<int32_t>(g_height), 0, drawableHeight,
                       drawableWidth, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);
  g_gl.BindFramebuffer(GL_READ_FRAMEBUFFER, 0);

  if (g_sdl2SwapWindow != nullptr && g_sdl2Window != nullptr) {
    g_sdl2SwapWindow(g_sdl2Window);
  } else {
    SDL_GL_SwapWindow(window);
  }
}

} // namespace

bool active() noexcept { return g_active; }

bool initialize(ProcAddressFn getProc, void* eglDisplay, uint32_t width, uint32_t height,
                wgpu::TextureFormat format) {
#ifndef WEBGPU_DAWN
  return false;
#else
  if (g_active) {
    return true;
  }
  if (getProc == nullptr || eglDisplay == nullptr || width == 0 || height == 0 || !g_device) {
    return false;
  }
  // The shared textures are allocated as GL_RGBA8, so the present pass must render that format.
  // webgpu::initialize() pins the surface configuration to match on this path.
  if (format != wgpu::TextureFormat::RGBA8Unorm) {
    Log.error("[sdl2shim-efb] unsupported present format {}; keeping swapchain present",
              magic_enum::enum_name(format));
    return false;
  }
  g_display = eglDisplay;
  g_width = width;
  g_height = height;

  if (!load_procs(getProc)) {
    Log.error("[sdl2shim-efb] required EGL/GLES entry points unavailable; keeping swapchain present");
    return false;
  }

  void* shimContext = g_gl.GetCurrentContext();
  if (shimContext == nullptr) {
    Log.error("[sdl2shim-efb] no EGL context current on the main thread; keeping swapchain present");
    return false;
  }

  const SDL_PropertiesID props = SDL_GetWindowProperties(window::get_sdl_window());
  g_sdl2Window = SDL_GetPointerProperty(props, SDL2_SHIM_WINDOW_PROP, nullptr);
  g_sdl2SwapWindow =
      reinterpret_cast<Sdl2SwapWindowFn>(SDL_GetPointerProperty(props, SDL2_SHIM_GL_SWAP_WINDOW_PROP, nullptr));
  Log.info("[sdl2shim-efb] present swap via {}", g_sdl2SwapWindow != nullptr && g_sdl2Window != nullptr
                                                     ? "firmware SDL2 SDL_GL_SwapWindow"
                                                     : "SDL3 SDL_GL_SwapWindow");

  if (!create_slots(shimContext, format)) {
    return false;
  }

  g_nextSlot = 0;
  g_framesOwed = 0;
  g_hasReadyFrame = false;
  g_aborting = false;
  g_active = true;
  Log.info("[sdl2shim-efb] {}x{} {} EFB slots shared with Dawn via EGLImage; present runs on the main thread",
           width, height, SlotCount);
  return true;
#endif
}

bool resize(uint32_t width, uint32_t height, wgpu::TextureFormat format) {
  if (!g_active) {
    return false;
  }
  if (width == g_width && height == g_height) {
    return true;
  }
  if (width == 0 || height == 0) {
    return false;
  }

  // Callers reach here through gfx::gpu_synchronize(), so the worker is idle and both slots are
  // ours to tear down.
  void* shimContext = g_gl.GetCurrentContext();
  for (Slot& slot : g_slots) {
    destroy_slot(slot);
  }
  g_width = width;
  g_height = height;
  g_nextSlot = 0;
  g_framesOwed = 0;
  g_hasReadyFrame = false;

  if (shimContext == nullptr || !create_slots(shimContext, format)) {
    Log.error("[sdl2shim-efb] failed to reallocate EFB slots at {}x{}; presentation is now disabled", width,
              height);
    g_active = false;
    return false;
  }
  Log.info("[sdl2shim-efb] reallocated EFB slots at {}x{}", width, height);
  return true;
}

void shutdown() {
  if (!g_active) {
    return;
  }
  {
    std::lock_guard lock{g_mutex};
    g_aborting = true;
  }
  g_slotFreed.notify_all();
  g_frameReady.notify_all();

  g_active = false;
  for (Slot& slot : g_slots) {
    destroy_slot(slot);
  }
  g_hasReadyFrame = false;
  g_framesOwed = 0;
  g_display = nullptr;
  g_sdl2Window = nullptr;
  g_sdl2SwapWindow = nullptr;
  g_gl = {};
}

std::optional<AcquiredFrame> acquire() {
  if (!g_active) {
    return std::nullopt;
  }
  ZoneScopedN("EFB acquire");
  const uint32_t slotIndex = g_nextSlot;
  {
    std::unique_lock lock{g_mutex};
    // Wait for the main thread to finish scanning this slot out. In steady state it already has.
    g_slotFreed.wait(lock, [&] { return g_aborting || g_slots[slotIndex].state == SlotState::Free; });
    if (g_aborting) {
      return std::nullopt;
    }
    g_slots[slotIndex].state = SlotState::Rendering;
  }

  Slot& slot = g_slots[slotIndex];
#ifdef WEBGPU_DAWN
  // Make Dawn's context wait until the main thread's blit of this slot has retired on the GPU.
  if (slot.revSync != nullptr) {
    dawn::native::opengl::PortmasterServerWaitEGLSync(device_handle(), slot.revSync);
    destroy_sync(slot.revSync);
  }
#endif
  g_nextSlot = (slotIndex + 1) % SlotCount;
  return AcquiredFrame{.slot = slotIndex, .view = slot.view};
}

void publish(uint32_t slot) {
  if (!g_active || slot >= SlotCount) {
    return;
  }
#ifdef WEBGPU_DAWN
  // Fence Dawn's just-submitted work so the shim context can order its blit behind it.
  g_slots[slot].fwdSync = dawn::native::opengl::PortmasterCreateFrameEGLSync(device_handle());
#endif

  // The main thread consumes the previous frame before enqueueing the work that produces this one,
  // so the mailbox is always empty here. If that invariant ever breaks, drop the stale frame rather
  // than overwrite it — an overwritten entry would leave its slot Ready forever and wedge acquire().
  uint32_t stale = SlotCount;
  {
    std::lock_guard lock{g_mutex};
    if (g_hasReadyFrame && g_readySlot != slot) {
      stale = g_readySlot;
    }
    g_slots[slot].state = SlotState::Ready;
    g_readySlot = slot;
    g_hasReadyFrame = true;
  }

  if (stale < SlotCount) {
    Log.warn("[sdl2shim-efb] dropping unpresented frame in slot {}", stale);
    Slot& staleSlot = g_slots[stale];
    destroy_sync(staleSlot.fwdSync);
    {
      std::lock_guard lock{g_mutex};
      staleSlot.state = SlotState::Free;
      if (g_framesOwed > 0) {
        --g_framesOwed;
      }
    }
    g_slotFreed.notify_one();
  }

  g_frameReady.notify_one();
}

void publish_empty() {
  if (!g_active) {
    return;
  }
  {
    std::lock_guard lock{g_mutex};
    if (g_framesOwed > 0) {
      --g_framesOwed;
    }
  }
  g_frameReady.notify_one();
}

void note_frame_enqueued() {
  if (!g_active) {
    return;
  }
  std::lock_guard lock{g_mutex};
  ++g_framesOwed;
}

void flush_present() {
  if (!g_active) {
    return;
  }
  ZoneScopedN("EFB flush present");

  uint32_t slotIndex = 0;
  {
    std::unique_lock lock{g_mutex};
    if (g_framesOwed <= 0 && !g_hasReadyFrame) {
      return; // nothing in flight (first frame, or the worker reported an empty frame)
    }
    // g_framesOwed drops to zero when the worker reports a frame it could not present, which is
    // also a reason to stop waiting.
    const auto ready = [&] { return g_aborting || g_hasReadyFrame || g_framesOwed <= 0; };
    if (!g_frameReady.wait_for(lock, PresentWaitTimeout, ready)) {
      Log.warn("[sdl2shim-efb] timed out waiting for the render worker to submit a frame");
      return;
    }
    if (g_aborting || !g_hasReadyFrame) {
      return;
    }
    slotIndex = g_readySlot;
    g_hasReadyFrame = false;
    if (g_framesOwed > 0) {
      --g_framesOwed;
    }
  }

  Slot& slot = g_slots[slotIndex];
  const bool presentable = window::is_presentable();
  if (presentable) {
    blit_and_swap(slot);
  }
  destroy_sync(slot.fwdSync);

  {
    std::lock_guard lock{g_mutex};
    // Only fence the slot back if we actually read from it; otherwise the worker has nothing to
    // wait on and can reuse it immediately.
    slot.revSync = presentable ? create_shim_fence() : nullptr;
    slot.state = SlotState::Free;
  }
  g_slotFreed.notify_one();

  if (presentable) {
    gfx::after_present();
  }
}

void recycle_pending() {
  if (!g_active) {
    return;
  }
  uint32_t slotIndex = 0;
  {
    std::lock_guard lock{g_mutex};
    if (!g_hasReadyFrame) {
      return;
    }
    slotIndex = g_readySlot;
    g_hasReadyFrame = false;
    if (g_framesOwed > 0) {
      --g_framesOwed;
    }
  }

  // Dropped without a read, so the worker needs no reverse fence to reuse the slot.
  Slot& slot = g_slots[slotIndex];
  destroy_sync(slot.fwdSync);
  {
    std::lock_guard lock{g_mutex};
    slot.state = SlotState::Free;
  }
  g_slotFreed.notify_one();
}

} // namespace aurora::webgpu::sdl2shim_present
