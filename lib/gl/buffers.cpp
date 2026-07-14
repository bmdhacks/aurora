#include "buffers.hpp"

namespace aurora::gl {

Buffer create_buffer(GLenum target, uint64_t size, bool dynamic) {
  GLuint id = 0;
  gl.GenBuffers(1, &id);
  gl.BindBuffer(target, id);
  gl.BufferData(target, static_cast<GLsizeiptr>(size), nullptr, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
  return Buffer{.id = id, .target = target, .size = size};
}

void upload_buffer(const Buffer& buffer, uint64_t offset, const void* data, uint64_t size) {
  if (buffer.id == 0 || size == 0) {
    return;
  }
  gl.BindBuffer(buffer.target, buffer.id);
  gl.BufferSubData(buffer.target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), data);
}

void destroy_buffer(Buffer& buffer) noexcept {
  if (buffer.id != 0) {
    gl.DeleteBuffers(1, &buffer.id);
    buffer.id = 0;
  }
}

} // namespace aurora::gl
