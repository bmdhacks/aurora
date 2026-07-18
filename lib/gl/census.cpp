#include "census.hpp"

#include "../internal.hpp"

#include <chrono>

namespace aurora::gl::census {
namespace {
Module Log("aurora::gl");
// Only touched from log_tick on the render worker; no synchronization needed.
std::chrono::steady_clock::time_point s_lastLog{};

double mb(const Counter& c) { return static_cast<double>(c.bytes.load(std::memory_order_relaxed)) / (1024.0 * 1024.0); }
} // namespace

Counter textures;
Counter buffers;
Counter programs;
Counter fbos;
Counter samplers;

void log_tick() noexcept {
  const auto now = std::chrono::steady_clock::now();
  if (s_lastLog.time_since_epoch().count() != 0 && now - s_lastLog < std::chrono::seconds(10)) {
    return;
  }
  s_lastLog = now;
  Log.info("[gl-census] tex {} ({:.1f} MB) buf {} ({:.1f} MB) prog {} fbo {} smp {}",
           textures.count.load(std::memory_order_relaxed), mb(textures),
           buffers.count.load(std::memory_order_relaxed), mb(buffers),
           programs.count.load(std::memory_order_relaxed), fbos.count.load(std::memory_order_relaxed),
           samplers.count.load(std::memory_order_relaxed));
}

} // namespace aurora::gl::census
