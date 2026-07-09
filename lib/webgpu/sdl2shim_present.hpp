#pragma once

#include <cstdint>
#include <optional>

#include <webgpu/webgpu_cpp.h>

namespace aurora::webgpu::sdl2shim_present {

// Dusklight (P4): double-buffered EFB present for the Mali/SDL2-shim device path.
//
// The render worker owns Dawn's EGL context and does the expensive work (game render + Submit).
// Presentation, though, is display-thread-bound on Mali's kmsdrm/gbm stack: a swap issued from the
// worker posts a buffer that is never scanned out (silent blank screen). So the worker renders the
// finished frame into one of two shared "EFB" textures and the main thread blits that texture to
// the window and swaps.
//
// The two threads run *different* EGL contexts (Dawn's, and the firmware's borrowed one), which is
// what lets them run concurrently: Dawn holds its exclusive make-current mutex for the whole of
// Submit, so anything presenting through Dawn would serialize against it. The textures are shared
// across the contexts as EGLImages and ordered with EGL fences.
//
// Inert unless the SDL2-shim video driver is in use; desktop keeps presenting from the worker.
struct AcquiredFrame {
  uint32_t slot = 0;
  wgpu::TextureView view;
};

// Resolves EGL and GLES entry points (the shim's loader; see gpu.cpp).
using ProcAddressFn = void* (*)(const char*);

// True once initialize() has succeeded. Every other entry point is a no-op when false.
[[nodiscard]] bool active() noexcept;

// Main thread, during webgpu::initialize(), with the shim's EGL context current.
// `width`/`height` must match the surface configuration the present pass renders against.
bool initialize(ProcAddressFn getProc, void* eglDisplay, uint32_t width, uint32_t height,
                wgpu::TextureFormat format);
// Reallocate the shared textures for a new window size. No-op when the size is unchanged.
// Main thread, shim context current, with no frame in flight.
bool resize(uint32_t width, uint32_t height, wgpu::TextureFormat format);
void shutdown();

// --- Render worker ---
// Claim the next EFB slot to render into, blocking until the main thread is done reading it.
// Returns nullopt when inactive or on failure; the caller then falls back to the swapchain path.
std::optional<AcquiredFrame> acquire();
// Fence Dawn's context and hand the slot to the main thread. Must follow a successful acquire().
void publish(uint32_t slot);
// Report that this frame produced nothing to present (keeps the main thread's accounting exact).
void publish_empty();

// --- Main thread ---
// Called once per frame immediately before the end-of-frame work is enqueued, so flush_present()
// knows a frame is owed and how long to wait for it.
void note_frame_enqueued();
// Present the frame the worker produced last: wait for it, blit it to the window, swap.
void flush_present();
// Drop any frame the worker is holding out to us without presenting it, and unblock a worker that
// is waiting for a slot. Must run before any render_worker::synchronize() on the main thread.
void recycle_pending();

} // namespace aurora::webgpu::sdl2shim_present
