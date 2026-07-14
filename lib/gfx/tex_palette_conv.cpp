#include "tex_palette_conv.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "texture.hpp"

#include <vector>

namespace aurora::gfx::tex_palette_conv {
static Module Log("aurora::gfx::tex_palette_conv");

// Phase 4: these WGSL sources are dead literals under the GL backend. Phase 4
// rewrites them to GLSL (fullscreen triangle vtx + index/TLUT lookup frag) and
// compiles real programs via the GL program cache.
static constexpr std::string_view ShaderPreambleVtx = R"(
struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);
var<private> uvs: array<vec2f, 3> = array(
    vec2f(0.0, 0.0),
    vec2f(0.0, 2.0),
    vec2f(2.0, 0.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

fn intensity(rgb: vec3f) -> f32 {
    // ITU-R BT.601 luma coefficients
    return dot(rgb, vec3f(0.257, 0.504, 0.098)) + 16.0 / 255.0;
}
)"sv;

// Direct: R16Sint index texture + TLUT -> RGBA8
static constexpr std::string_view ShaderDirect = R"(
@group(0) @binding(0) var src_samp: sampler;
@group(0) @binding(1) var src: texture_2d<i32>;
@group(0) @binding(2) var tlut: texture_2d<f32>;

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let texSize = vec2f(textureDimensions(src));
    let coord = vec2i(floor(in.uv * texSize));
    let idx = textureLoad(src, coord, 0).r;
    return textureLoad(tlut, vec2i(idx, 0), 0);
}
)"sv;

// FromFloat8: f32 texture (R8Unorm) -> 8-bit index -> TLUT -> RGBA8
static constexpr std::string_view ShaderFromFloat8 = R"(
@group(0) @binding(0) var src_samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var tlut: texture_2d<f32>;

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let texSize = vec2f(textureDimensions(src));
    let coord = vec2i(floor(in.uv * texSize));
    let r = textureLoad(src, coord, 0).r;
    return textureLoad(tlut, vec2i(i32(r * 255.0), 0), 0);
}
)"sv;

// FromFloat4: f32 texture (R8Unorm) -> 4-bit index -> TLUT -> RGBA8
static constexpr std::string_view ShaderFromFloat4 = R"(
@group(0) @binding(0) var src_samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var tlut: texture_2d<f32>;

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let texSize = vec2f(textureDimensions(src));
    let coord = vec2i(floor(in.uv * texSize));
    let r = textureLoad(src, coord, 0).r;
    return textureLoad(tlut, vec2i(i32(r * 15.0), 0), 0);
}
)"sv;

// Phase 4: wgpu carried a per-variant BindGroupLayout alongside the pipeline; GL
// has no layout object, so PipelineInfo collapses to just the pipeline handle.
struct PipelineInfo {
  gl::Pipeline pipeline;
};

static PipelineInfo g_directPipeline;
static PipelineInfo g_fromFloat8Pipeline;
static PipelineInfo g_fromFloat4Pipeline;
static gl::Sampler g_sampler;

static PipelineInfo create_pipeline(std::string_view fragBindingsAndShader, const char* label) {
  // Assemble the full WGSL source (preamble + per-variant frag bindings/shader).
  // Backend-agnostic string plumbing kept intact; Phase 4 lowers this to GLSL and
  // compiles a real program instead of the stub below.
  std::string shaderSource;
  shaderSource.reserve(ShaderPreambleVtx.size() + fragBindingsAndShader.size());
  shaderSource += ShaderPreambleVtx;
  shaderSource += fragBindingsAndShader;
  (void)shaderSource;
  (void)label;

  // Phase 4: compile the program and bake real fixed-function state. For now bake
  // what we statically know (single RGBA8 color target, triangle-list, no blend/
  // depth/cull) and return a program-less (0) pipeline.
  gl::Pipeline pipeline{};
  pipeline.state.topology = gl::PrimitiveTopology::TriangleList;
  return PipelineInfo{.pipeline = pipeline};
}

static const PipelineInfo& pipeline_for_variant(Variant variant) {
  switch (variant) {
  case Variant::Direct:
    return g_directPipeline;
  case Variant::FromFloat8:
    return g_fromFloat8Pipeline;
  case Variant::FromFloat4:
    return g_fromFloat4Pipeline;
  }
  FATAL("invalid palette conv variant {}", static_cast<int>(variant));
}

void initialize() {
  g_directPipeline = create_pipeline(ShaderDirect, "TexPaletteConv Direct");
  g_fromFloat8Pipeline = create_pipeline(ShaderFromFloat8, "TexPaletteConv FromFloat8");
  g_fromFloat4Pipeline = create_pipeline(ShaderFromFloat4, "TexPaletteConv FromFloat4");
  // Phase 4: create the real nearest-filter (NonFiltering) sampler used to sample
  // the index/TLUT textures. Descriptor intent preserved for the later phase.
  const gl::SamplerDescriptor samplerDesc{
      .magFilter = gl::FilterMode::Nearest,
      .minFilter = gl::FilterMode::Nearest,
  };
  (void)samplerDesc;
  g_sampler = gl::Sampler{};
}

void shutdown() {
  g_directPipeline = {};
  g_fromFloat8Pipeline = {};
  g_fromFloat4Pipeline = {};
  g_sampler = {};
}

void run(const ConvRequest& req) {
  const auto& info = pipeline_for_variant(req.variant);

  // Phase 4: build the binding set (sampler @0, src index texture @1, tlut @2),
  // begin a render pass into req.dst (load=Clear, store=Store, clear 0,0,0,0),
  // set the pipeline + bindings and Draw(3) the fullscreen triangle to resolve
  // the paletted texture. Binding intent preserved below for the later phase.
  gl::BindingSet bindings{};
  bindings.textures[0] = {req.src->sampleTextureView.id, g_sampler.id};
  bindings.textures[1] = {req.tlut->sampleTextureView.id, g_sampler.id};
  (void)info;
  (void)bindings;
  (void)req.dst->attachmentTextureView;
}

} // namespace aurora::gfx::tex_palette_conv
