#include "BackendBinding.hpp"

#include "../internal.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <linux/fb.h>
#include <memory>
#include <sys/ioctl.h>
#include <unistd.h>

#if !defined(SDL_PLATFORM_MACOS) && !defined(SDL_PLATFORM_IOS) && !defined(SDL_PLATFORM_TVOS)
#include <SDL3/SDL_loadso.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>
#endif

namespace aurora::webgpu::utils {
std::shared_ptr<wgpu::ChainedStruct> SetupWindowAndGetSurfaceDescriptorCocoa(SDL_Window* window);

namespace {
Module Log("aurora::dawn");

#if defined(SDL_PLATFORM_LINUX)
constexpr const char* SDL2_SHIM_EGL_DISPLAY_PROP = "SDL.window.sdl2_backend.egl_display";
constexpr const char* SDL2_SHIM_EGL_SURFACE_PROP = "SDL.window.sdl2_backend.egl_surface";
constexpr const char* SDL2_SHIM_EGL_CONTEXT_PROP = "SDL.window.sdl2_backend.egl_context";
constexpr const char* SDL2_SHIM_DRAWABLE_WIDTH_PROP = "SDL.window.sdl2_backend.drawable_width";
constexpr const char* SDL2_SHIM_DRAWABLE_HEIGHT_PROP = "SDL.window.sdl2_backend.drawable_height";
constexpr const char* SDL2_SHIM_GL_GET_PROC_PROP = "SDL.window.sdl2_backend.gl_get_proc";
constexpr const char* SDL2_SHIM_WINDOW_PROP = "SDL.window.sdl2_backend.window";
constexpr const char* SDL2_SHIM_CONTEXT_PROP = "SDL.window.sdl2_backend.context";
constexpr const char* SDL2_SHIM_GL_MAKE_CURRENT_PROP = "SDL.window.sdl2_backend.gl_make_current";
constexpr const char* SDL2_SHIM_GL_SWAP_WINDOW_PROP = "SDL.window.sdl2_backend.gl_swap_window";
constexpr const char* SDL2_SHIM_EGL_MAKE_CURRENT_PROP = "SDL.window.sdl2_backend.egl_make_current";
constexpr int SDL2_EGL_DRAW = 0x3059;

using Sdl2GlMakeCurrentFn = int (*)(void*, void*);
using EglGetCurrentDisplayFn = void* (*)();
using EglGetCurrentSurfaceFn = void* (*)(int);
using EglGetCurrentContextFn = void* (*)();

struct PortmasterSdl2ShimEglSurface {
  uint32_t magic;
  uint32_t version;
  void* eglDisplay;
  void* eglSurface;
  void* sdl2Window;
  void* sdl2Context;
  void* glGetProc;
  void* glMakeCurrent;
  void* glSwapWindow;
  void* eglMakeCurrent;
  void* eglContext;
  uint32_t flags;
  uint32_t drawableWidth;
  uint32_t drawableHeight;
  uint32_t reserved;
};

constexpr uint32_t PortmasterSdl2ShimEglSurfaceMagic = 0x44533245; // DS2E
constexpr uint32_t PortmasterSdl2ShimEglSurfaceVersion = 2;
constexpr uint32_t PortmasterSdl2ShimFlagOwnedEgl = 1u << 0;
constexpr uint32_t PortmasterSdl2ShimFlagSdlSwap = 1u << 1;
constexpr uint32_t PortmasterSdl2ShimFlagDawnContextPresent = 1u << 2;
PortmasterSdl2ShimEglSurface g_portmasterSdl2ShimSurface{};
SDL_SharedObject* g_portmasterSdl2ShimEglLibrary = nullptr;
EglGetCurrentDisplayFn g_portmasterEglGetCurrentDisplay = nullptr;
EglGetCurrentSurfaceFn g_portmasterEglGetCurrentSurface = nullptr;
EglGetCurrentContextFn g_portmasterEglGetCurrentContext = nullptr;

struct PortmasterFbdevWindow {
  unsigned short width;
  unsigned short height;
};

PortmasterFbdevWindow g_portmasterFbdevWindow{};

bool init_portmaster_fbdev_window() {
  int fd = open("/dev/fb0", O_RDWR, 0);
  fb_var_screeninfo vinfo{};
  if (fd < 0 || ioctl(fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
    if (fd >= 0) {
      close(fd);
    }
    return false;
  }
  close(fd);
  g_portmasterFbdevWindow.width = static_cast<unsigned short>(vinfo.xres);
  g_portmasterFbdevWindow.height = static_cast<unsigned short>(vinfo.yres);
  return true;
}

template <typename T>
T load_sdl2shim_egl_function(const char* name) {
  if (g_portmasterSdl2ShimEglLibrary == nullptr) {
    for (const char* libraryName : {"libEGL.so.1", "libEGL.so"}) {
      g_portmasterSdl2ShimEglLibrary = SDL_LoadObject(libraryName);
      if (g_portmasterSdl2ShimEglLibrary != nullptr) {
        Log.info("SDL2-shim EGL handle fallback opened {}", libraryName);
        break;
      }
    }
  }
  if (g_portmasterSdl2ShimEglLibrary == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(SDL_LoadFunction(g_portmasterSdl2ShimEglLibrary, name));
}

void fill_sdl2shim_current_egl_handles(SDL_PropertiesID props, void* sdl2Window, void* sdl2Context,
                                       void* glMakeCurrent, void*& eglDisplay, void*& eglSurface,
                                       void*& eglContext) {
  if ((eglDisplay != nullptr && eglSurface != nullptr && eglContext != nullptr) || sdl2Window == nullptr ||
      sdl2Context == nullptr || glMakeCurrent == nullptr) {
    return;
  }

  if (reinterpret_cast<Sdl2GlMakeCurrentFn>(glMakeCurrent)(sdl2Window, sdl2Context) != 0) {
    Log.warn("SDL2-shim EGL handle fallback could not make SDL2 context current");
    return;
  }

  if (g_portmasterEglGetCurrentDisplay == nullptr) {
    g_portmasterEglGetCurrentDisplay = load_sdl2shim_egl_function<EglGetCurrentDisplayFn>("eglGetCurrentDisplay");
  }
  if (g_portmasterEglGetCurrentSurface == nullptr) {
    g_portmasterEglGetCurrentSurface = load_sdl2shim_egl_function<EglGetCurrentSurfaceFn>("eglGetCurrentSurface");
  }
  if (g_portmasterEglGetCurrentContext == nullptr) {
    g_portmasterEglGetCurrentContext = load_sdl2shim_egl_function<EglGetCurrentContextFn>("eglGetCurrentContext");
  }

  if (eglDisplay == nullptr && g_portmasterEglGetCurrentDisplay != nullptr) {
    eglDisplay = g_portmasterEglGetCurrentDisplay();
  }
  if (eglSurface == nullptr && g_portmasterEglGetCurrentSurface != nullptr) {
    eglSurface = g_portmasterEglGetCurrentSurface(SDL2_EGL_DRAW);
  }
  if (eglContext == nullptr && g_portmasterEglGetCurrentContext != nullptr) {
    eglContext = g_portmasterEglGetCurrentContext();
  }

  if (eglDisplay != nullptr) {
    SDL_SetPointerProperty(props, SDL2_SHIM_EGL_DISPLAY_PROP, eglDisplay);
  }
  if (eglSurface != nullptr) {
    SDL_SetPointerProperty(props, SDL2_SHIM_EGL_SURFACE_PROP, eglSurface);
  }
  if (eglContext != nullptr) {
    SDL_SetPointerProperty(props, SDL2_SHIM_EGL_CONTEXT_PROP, eglContext);
  }
  Log.info("SDL2-shim EGL handle fallback resolved display={} surface={} context={}", eglDisplay, eglSurface,
           eglContext);
}
#endif
} // namespace

std::shared_ptr<wgpu::ChainedStruct> SetupWindowAndGetSurfaceDescriptor(SDL_Window* window) {
#if defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_TVOS)
  return SetupWindowAndGetSurfaceDescriptorCocoa(window);
#else
  const auto props = SDL_GetWindowProperties(window);
#if defined(SDL_PLATFORM_ANDROID)
  std::shared_ptr<wgpu::SurfaceSourceAndroidNativeWindow> desc =
      std::make_shared<wgpu::SurfaceSourceAndroidNativeWindow>();
  desc->window = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr);
  return std::move(desc);
#elif defined(SDL_PLATFORM_WIN32)
  std::shared_ptr<wgpu::SurfaceSourceWindowsHWND> desc = std::make_shared<wgpu::SurfaceSourceWindowsHWND>();
  desc->hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
  desc->hinstance = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
  return std::move(desc);
#elif defined(SDL_PLATFORM_LINUX)
  if (std::getenv("DUSKLIGHT_PORTMASTER_EGL_FBDEV_SURFACE") != nullptr && init_portmaster_fbdev_window()) {
    Log.info("Using PortMaster fbdev sentinel Dawn surface: {}x{}", g_portmasterFbdevWindow.width,
             g_portmasterFbdevWindow.height);
    std::shared_ptr<wgpu::SurfaceSourceXlibWindow> desc = std::make_shared<wgpu::SurfaceSourceXlibWindow>();
    desc->display = nullptr;
    desc->window = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&g_portmasterFbdevWindow));
    return std::move(desc);
  }
  const char* driver = SDL_GetCurrentVideoDriver();
  if (driver == nullptr) {
    Log.error("SDL has no current video driver while creating Dawn surface descriptor");
    return nullptr;
  }
  Log.info("Creating Dawn surface descriptor for SDL video driver: {}", driver);
  if (SDL_strcmp(driver, "wayland") == 0) {
    std::shared_ptr<wgpu::SurfaceSourceWaylandSurface> desc = std::make_shared<wgpu::SurfaceSourceWaylandSurface>();
    desc->display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
    desc->surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    Log.info("Wayland Dawn surface properties: display={} surface={}", desc->display, desc->surface);
    if (desc->display == nullptr || desc->surface == nullptr) {
      Log.error("Wayland SDL driver did not expose display/surface pointers for Dawn");
      return nullptr;
    }
    return std::move(desc);
  }
  if (SDL_strcmp(driver, "x11") == 0) {
    std::shared_ptr<wgpu::SurfaceSourceXlibWindow> desc = std::make_shared<wgpu::SurfaceSourceXlibWindow>();
    desc->display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
    desc->window = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
    Log.info("X11 Dawn surface properties: display={} window={}", desc->display, desc->window);
    if (desc->display == nullptr || desc->window == 0) {
      Log.error("X11 SDL driver did not expose display/window handles for Dawn");
      return nullptr;
    }
    return std::move(desc);
  }
  if (SDL_strcmp(driver, "sdl2") == 0) {
    void* eglDisplay = SDL_GetPointerProperty(props, SDL2_SHIM_EGL_DISPLAY_PROP, nullptr);
    void* eglSurface = SDL_GetPointerProperty(props, SDL2_SHIM_EGL_SURFACE_PROP, nullptr);
    void* eglContext = SDL_GetPointerProperty(props, SDL2_SHIM_EGL_CONTEXT_PROP, nullptr);
    void* glGetProc = SDL_GetPointerProperty(props, SDL2_SHIM_GL_GET_PROC_PROP, nullptr);
    void* sdl2Window = SDL_GetPointerProperty(props, SDL2_SHIM_WINDOW_PROP, nullptr);
    void* sdl2Context = SDL_GetPointerProperty(props, SDL2_SHIM_CONTEXT_PROP, nullptr);
    void* glMakeCurrent = SDL_GetPointerProperty(props, SDL2_SHIM_GL_MAKE_CURRENT_PROP, nullptr);
    void* glSwapWindow = SDL_GetPointerProperty(props, SDL2_SHIM_GL_SWAP_WINDOW_PROP, nullptr);
    void* eglMakeCurrent = SDL_GetPointerProperty(props, SDL2_SHIM_EGL_MAKE_CURRENT_PROP, nullptr);
    int drawableWidth = static_cast<int>(SDL_GetNumberProperty(props, SDL2_SHIM_DRAWABLE_WIDTH_PROP, 0));
    int drawableHeight = static_cast<int>(SDL_GetNumberProperty(props, SDL2_SHIM_DRAWABLE_HEIGHT_PROP, 0));
    if (drawableWidth <= 0 || drawableHeight <= 0) {
      SDL_GetWindowSizeInPixels(window, &drawableWidth, &drawableHeight);
    }
    uint32_t flags = 0;
    if (std::getenv("DUSKLIGHT_PORTMASTER_SDL2SHIM_OWNED_EGL") != nullptr) {
      flags |= PortmasterSdl2ShimFlagOwnedEgl;
    }
    const char* swapPresent = std::getenv("DUSKLIGHT_PORTMASTER_SDL2SHIM_SWAP_PRESENT");
    if (swapPresent == nullptr || (swapPresent[0] != '\0' && swapPresent[0] != '0')) {
      flags |= PortmasterSdl2ShimFlagSdlSwap;
    }
    if (std::getenv("DUSKLIGHT_PORTMASTER_SDL2SHIM_DAWN_CONTEXT_PRESENT") != nullptr) {
      flags |= PortmasterSdl2ShimFlagDawnContextPresent;
    }
    fill_sdl2shim_current_egl_handles(props, sdl2Window, sdl2Context, glMakeCurrent, eglDisplay, eglSurface,
                                      eglContext);
    Log.info("SDL2-shim Dawn surface properties: display={} surface={} eglContext={} getProc={} window={} "
             "context={} makeCurrent={} swapWindow={} eglMakeCurrent={} drawable={}x{} flags=0x{:x} table={}",
             eglDisplay, eglSurface, eglContext, glGetProc, sdl2Window, sdl2Context, glMakeCurrent, glSwapWindow,
             eglMakeCurrent, drawableWidth, drawableHeight, flags, static_cast<void*>(&g_portmasterSdl2ShimSurface));
    if (eglDisplay == nullptr || eglSurface == nullptr || glGetProc == nullptr || sdl2Window == nullptr ||
        sdl2Context == nullptr || glMakeCurrent == nullptr || glSwapWindow == nullptr) {
      Log.error("SDL2-shim did not expose enough EGL/context handles for Dawn");
      return nullptr;
    }
    ::setenv("DUSKLIGHT_PORTMASTER_SDL2SHIM_EGL_SURFACE", "1", 1);
    g_portmasterSdl2ShimSurface = {
        .magic = PortmasterSdl2ShimEglSurfaceMagic,
        .version = PortmasterSdl2ShimEglSurfaceVersion,
        .eglDisplay = eglDisplay,
        .eglSurface = eglSurface,
        .sdl2Window = sdl2Window,
        .sdl2Context = sdl2Context,
        .glGetProc = glGetProc,
        .glMakeCurrent = glMakeCurrent,
        .glSwapWindow = glSwapWindow,
        .eglMakeCurrent = eglMakeCurrent,
        .eglContext = eglContext,
        .flags = flags,
        .drawableWidth = drawableWidth > 0 ? static_cast<uint32_t>(drawableWidth) : 0,
        .drawableHeight = drawableHeight > 0 ? static_cast<uint32_t>(drawableHeight) : 0,
        .reserved = 0,
    };
    char tablePtr[32];
    std::snprintf(tablePtr, sizeof(tablePtr), "0x%llx",
                  static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&g_portmasterSdl2ShimSurface)));
    ::setenv("DUSKLIGHT_PORTMASTER_SDL2SHIM_EGL_SURFACE_INFO", tablePtr, 1);
    std::shared_ptr<wgpu::SurfaceSourceXlibWindow> desc = std::make_shared<wgpu::SurfaceSourceXlibWindow>();
    desc->display = nullptr;
    desc->window = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&g_portmasterSdl2ShimSurface));
    return std::move(desc);
  }
  if (SDL_strcmp(driver, "kmsdrm") == 0) {
    const auto deviceIndex = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_KMSDRM_DEVICE_INDEX_NUMBER, -1);
    const auto drmFd = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_KMSDRM_DRM_FD_NUMBER, -1);
    void* gbmDevice = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_KMSDRM_GBM_DEVICE_POINTER, nullptr);
    Log.info("KMSDRM Dawn surface properties: device_index={} drm_fd={} gbm_device={}", deviceIndex, drmFd,
             gbmDevice);
    Log.error("KMSDRM/GBM Dawn surface creation is not implemented in this Dawn binding yet");
    return nullptr;
  }
  Log.error("Unsupported SDL video driver for Dawn surface: {}", driver);
#endif
  return nullptr;
#endif
}

} // namespace aurora::webgpu::utils
