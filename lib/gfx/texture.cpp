#include "common.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "aurora/aurora.h"
#include "texture.hpp"
#include "texture_convert.hpp"
#include "../gx/gx_fmt.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx {
using webgpu::g_device;
using webgpu::g_queue;

namespace {
Module Log("aurora::gfx");

wgpu::Extent3D physical_size(wgpu::Extent3D size, TextureFormatInfo info) {
  const uint32_t width = ((size.width + info.blockWidth - 1) / info.blockWidth) * info.blockWidth;
  const uint32_t height = ((size.height + info.blockHeight - 1) / info.blockHeight) * info.blockHeight;
  return {.width = width, .height = height, .depthOrArrayLayers = size.depthOrArrayLayers};
}

bool setup_swizzle(wgpu::TextureComponentSwizzleDescriptor& swizzle, u32 format) noexcept {
  if (!webgpu::g_textureComponentSwizzleSupported) {
    return false;
  }

  switch (format) {
  case GX_TF_R8_PC:
    swizzle.swizzle.r = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.g = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.b = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.a = wgpu::ComponentSwizzle::R;
    return true;
  case GX_TF_RG8_PC:
    swizzle.swizzle.r = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.g = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.b = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.a = wgpu::ComponentSwizzle::G;
    return true;
  default:
    return false;
  }
}
} // namespace

TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 format, ArrayRef<uint8_t> data,
                                    bool tlut, const char* label) noexcept {
  ZoneScoped;

  auto handle = new_dynamic_texture_2d(width, height, mips, format, label);
  auto& ref = *handle;

  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    if (tlut) {
      CHECK(ref.size.height == 1, "new_static_texture_2d[{}]: expected tlut height 1, got {}", label, ref.size.height);
      CHECK(ref.mipCount == 1, "new_static_texture_2d[{}]: expected tlut mipCount 1, got {}", label, ref.mipCount);
      converted = convert_tlut(ref.gxFormat, ref.size.width, data);
    } else {
      converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    }
    if (!converted.data.empty()) {
      data = converted.data;
      ref.hasArbitraryMips = converted.hasArbitraryMips;
    }
  }

  uint32_t offset = 0;
  // Use ref.mipCount (already clamped in new_dynamic_texture_2d), not the raw mips
  // argument — otherwise we'd write past the texture's actual mip levels.
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const wgpu::Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(offset + dataSize <= data.size(), "new_static_texture_2d[{}]: expected at least {} bytes, got {}", label,
          offset + dataSize, data.size());
    const wgpu::TexelCopyTextureInfo dstView{
        .texture = ref.texture,
        .mipLevel = mip,
    };
    if constexpr (UseTextureBuffer) {
      queue_texture_upload_data(data.data() + offset, bytesPerRow, heightBlocks, std::move(dstView), physicalSize);
    } else {
      const wgpu::TexelCopyBufferLayout dataLayout{
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_queue.WriteTexture(&dstView, data.data() + offset, dataSize, &dataLayout, &physicalSize);
    }
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("new_static_texture_2d[{}]: texture used {} bytes, but given {} bytes", label, offset, data.size());
  }
  return handle;
}

TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                     const char* label) noexcept {
  ZoneScopedS(3);
  // A zero-sized texture makes Mali's glTexStorage2D fail with GL_INVALID_VALUE and lose
  // the device (desktop Mesa tolerates it). GX can request 0-dim copies/targets when a
  // subsystem is scaled to nothing (e.g. shadows under shadowResolutionMultiplier=0), so
  // clamp to at least 1x1 — the texture is valid and the draw becomes an effective no-op.
  width = width < 1u ? 1u : width;
  height = height < 1u ? 1u : height;
  // Mali's GLES rejects glTexStorage2D with GL_INVALID_VALUE when the requested mip
  // count exceeds floor(log2(max(w,h)))+1. Some GX textures over-specify their mip
  // chain (harmless on desktop Mesa, fatal on Mali), so clamp to the legal maximum.
  {
    uint32_t maxMips = 1;
    for (uint32_t d = (width > height ? width : height); d > 1u; d >>= 1) {
      ++maxMips;
    }
    if (mips > maxMips) {
      Log.warn("Clamping texture '{}' mip levels {} -> {} for {}x{} (Mali glTexStorage2D limit)", label, mips, maxMips,
               width, height);
      mips = maxMips;
    }
    if (mips == 0) {
      mips = 1;
    }
  }
  const auto wgpuFormat = to_wgpu(gxFormat);
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TextureDescriptor textureDescriptor{
      .label = label,
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = wgpuFormat,
      .mipLevelCount = mips,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);
  const auto viewLabel = fmt::format("{} view", label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = wgpuFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = mips,
  };
  wgpu::TextureComponentSwizzleDescriptor swizzle;
  if (setup_swizzle(swizzle, gxFormat)) {
    textureViewDescriptor.nextInChain = &swizzle;
  }
  auto textureView = texture.CreateView(&textureViewDescriptor);
  return std::make_shared<TextureRef>(std::move(texture), std::move(textureView), wgpu::TextureView{}, size, wgpuFormat,
                                      mips, gxFormat);
}

TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;

  // Clamp to at least 1x1 — a 0-sized target (e.g. a shadow EFB-resolve copy under
  // shadowResolutionMultiplier=0) crashes Mali's glTexStorage2D with GL_INVALID_VALUE.
  width = width < 1u ? 1u : width;
  height = height < 1u ? 1u : height;
  const auto wgpuFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TextureDescriptor textureDescriptor{
      .label = label,
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = wgpuFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);

  // Create texture view for color attachments
  const auto viewLabel = fmt::format("{} view", label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = wgpuFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
  };
  auto attachmentTextureView = texture.CreateView(&textureViewDescriptor);
  wgpu::TextureView sampleTextureView = attachmentTextureView;
  return std::make_shared<TextureRef>(std::move(texture), std::move(sampleTextureView),
                                      std::move(attachmentTextureView), size, wgpuFormat, 1, gxFormat);
}

TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;

  // Clamp to at least 1x1 — a 0-sized copy-conv target crashes Mali's glTexStorage2D.
  width = width < 1u ? 1u : width;
  height = height < 1u ? 1u : height;
  const auto wgpuFormat = to_wgpu(gxFormat);
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TextureDescriptor textureDescriptor{
      .label = label,
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = wgpuFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);

  // Create texture view for color attachments
  const auto viewLabel = fmt::format("{} view", label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = wgpuFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
  };
  auto attachmentTextureView = texture.CreateView(&textureViewDescriptor);
  wgpu::TextureView sampleTextureView = attachmentTextureView;
  return std::make_shared<TextureRef>(std::move(texture), std::move(sampleTextureView),
                                      std::move(attachmentTextureView), size, wgpuFormat, 1, gxFormat);
}

void write_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept {
  ZoneScoped;

  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    ref.hasArbitraryMips = converted.hasArbitraryMips;
    if (!converted.data.empty()) {
      data = converted.data;
    }
  }

  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const wgpu::Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(offset + dataSize <= data.size(), "write_texture: expected at least {} bytes, got {}", offset + dataSize,
          data.size());
    const wgpu::TexelCopyTextureInfo dstView{
        .texture = ref.texture,
        .mipLevel = mip,
    };
    if constexpr (UseTextureBuffer) {
      queue_texture_upload_data(data.data() + offset, bytesPerRow, heightBlocks, std::move(dstView), physicalSize);
    } else {
      const wgpu::TexelCopyBufferLayout dataLayout{
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_queue.WriteTexture(&dstView, data.data() + offset, dataSize, &dataLayout, &physicalSize);
    }
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("write_texture: texture used {} bytes, but given {} bytes", offset, data.size());
  }
}
} // namespace aurora::gfx
