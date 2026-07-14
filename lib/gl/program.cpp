#include "program.hpp"

#include "../internal.hpp"

#include <string>
#include <vector>

namespace aurora::gl {
namespace {
Module Log("aurora::gl");

GLuint compile_stage(GLenum stage, const char* source, const char* label) {
  const GLuint shader = gl.CreateShader(stage);
  if (shader == 0) {
    Log.error("compile_program({}): glCreateShader failed", label);
    return 0;
  }
  const GLchar* sources[]{source};
  gl.ShaderSource(shader, 1, sources, nullptr);
  gl.CompileShader(shader);

  GLint status = GL_FALSE;
  gl.GetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    GLint logLength = 0;
    gl.GetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::string infoLog(logLength > 0 ? static_cast<size_t>(logLength) : 1, '\0');
    gl.GetShaderInfoLog(shader, static_cast<GLsizei>(infoLog.size()), nullptr, infoLog.data());
    Log.error("compile_program({}): {} shader compile failed: {}", label,
              stage == GL_VERTEX_SHADER ? "vertex" : "fragment", infoLog.c_str());
    gl.DeleteShader(shader);
    return 0;
  }
  return shader;
}
} // namespace

GLuint compile_program(const char* vertexSource, const char* fragmentSource, const char* label) {
  const GLuint vs = compile_stage(GL_VERTEX_SHADER, vertexSource, label);
  if (vs == 0) {
    return 0;
  }
  const GLuint fs = compile_stage(GL_FRAGMENT_SHADER, fragmentSource, label);
  if (fs == 0) {
    gl.DeleteShader(vs);
    return 0;
  }

  const GLuint program = gl.CreateProgram();
  gl.AttachShader(program, vs);
  gl.AttachShader(program, fs);
  gl.LinkProgram(program);
  // The shader objects are reference-counted by the program; drop our references
  // regardless of link result.
  gl.DeleteShader(vs);
  gl.DeleteShader(fs);

  GLint status = GL_FALSE;
  gl.GetProgramiv(program, GL_LINK_STATUS, &status);
  if (status != GL_TRUE) {
    GLint logLength = 0;
    gl.GetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::string infoLog(logLength > 0 ? static_cast<size_t>(logLength) : 1, '\0');
    gl.GetProgramInfoLog(program, static_cast<GLsizei>(infoLog.size()), nullptr, infoLog.data());
    Log.error("compile_program({}): link failed: {}", label, infoLog.c_str());
    gl.DeleteProgram(program);
    return 0;
  }
  return program;
}

} // namespace aurora::gl
