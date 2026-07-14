#pragma once

#include "aurora/aurora.h"
#include "gfx/common.hpp"
#include "webgpu/gpu.hpp"

#include <SDL3/SDL_events.h>
#include <aurora/rmlui.hpp>

namespace aurora::rmlui {

struct RecordedFrame {
  gfx::BindGroupRef bindGroup = 0; // 0 = no overlay recorded
  bool overlay = false;
};

void initialize(const AuroraWindowSize& size) noexcept;
void handle_event(SDL_Event& event) noexcept;
RecordedFrame record_frame(const webgpu::Viewport& presentViewport) noexcept;
void shutdown() noexcept;

} // namespace aurora::rmlui
