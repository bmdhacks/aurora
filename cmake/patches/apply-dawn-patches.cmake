# Applies the Dusklight Mali/OpenGL-backend patch stack to a freshly-fetched Dawn source tree.
# Invoked as a FetchContent PATCH_COMMAND:
#   cmake -DDAWN_SOURCE_DIR=<dir> -P apply-dawn-patches.cmake
#
# Design goals (this is the fix for the stale-Dawn-source class of bug):
#   * Idempotent: re-running against an already-patched tree is a no-op, never a double-apply.
#   * Staleness-guarded: if the tree was patched with a *different* patch set than the one on disk
#     now (e.g. a patch was edited but FetchContent reused the cached source), fail loudly with the
#     exact remedy rather than silently building stale code.
#   * Fails hard on a genuine apply failure — no partial patching.

if (NOT DEFINED DAWN_SOURCE_DIR)
  message(FATAL_ERROR "apply-dawn-patches: DAWN_SOURCE_DIR not set")
endif ()
if (NOT IS_DIRECTORY "${DAWN_SOURCE_DIR}")
  message(FATAL_ERROR "apply-dawn-patches: DAWN_SOURCE_DIR '${DAWN_SOURCE_DIR}' is not a directory")
endif ()

find_package(Git REQUIRED)

# The fetched Dawn source usually lives under the consuming project's build/ dir, i.e. *inside* an
# outer git work tree (dusklight). Left alone, `git apply` discovers that outer repo and checks the
# patch against its index instead of the extracted files on disk — which makes --reverse --check
# spuriously succeed and the idempotency guard skip every patch. Cap repo discovery at the source
# tree's parent so git apply always operates purely on the files here.
file(REAL_PATH "${DAWN_SOURCE_DIR}" _dawn_real)
get_filename_component(_git_ceiling "${_dawn_real}" DIRECTORY)

file(GLOB _patches LIST_DIRECTORIES FALSE "${CMAKE_CURRENT_LIST_DIR}/dawn/*.patch")
list(SORT _patches)
if (NOT _patches)
  message(FATAL_ERROR "apply-dawn-patches: no patches found in ${CMAKE_CURRENT_LIST_DIR}/dawn/")
endif ()

# Fingerprint the exact patch set (names + contents), so the stamp detects any edit/add/remove.
set(_fingerprint_input "")
foreach (_patch IN LISTS _patches)
  get_filename_component(_name "${_patch}" NAME)
  file(SHA256 "${_patch}" _hash)
  string(APPEND _fingerprint_input "${_name}:${_hash}\n")
endforeach ()
string(SHA256 _patchset_hash "${_fingerprint_input}")

set(_stamp "${DAWN_SOURCE_DIR}/.dusklight-dawn-patches.stamp")
if (EXISTS "${_stamp}")
  file(READ "${_stamp}" _stamped_hash)
  string(STRIP "${_stamped_hash}" _stamped_hash)
  if (_stamped_hash STREQUAL _patchset_hash)
    message(STATUS "aurora: Dawn patches already applied (fingerprint match); skipping")
    return()
  endif ()
  message(FATAL_ERROR
    "aurora: Dawn source at '${DAWN_SOURCE_DIR}' was patched with a DIFFERENT patch set than the one "
    "in cmake/patches/dawn/ now. FetchContent reuses a cached source tree, so an edited/added/removed "
    "patch will NOT re-extract it. Remove the fetched source and reconfigure:\n"
    "    rm -rf \"${DAWN_SOURCE_DIR}\"\n"
    "(usually build/*/_deps/dawn-src), then re-run cmake.")
endif ()

# Fresh tree: apply each patch in sorted order. --reverse --check first so a tree that is already
# patched (but missing the stamp, e.g. hand-applied) is detected and skipped rather than failing.
foreach (_patch IN LISTS _patches)
  get_filename_component(_name "${_patch}" NAME)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "GIT_CEILING_DIRECTORIES=${_git_ceiling}"
            "${GIT_EXECUTABLE}" apply --reverse --check -p1 "${_patch}"
    WORKING_DIRECTORY "${DAWN_SOURCE_DIR}"
    RESULT_VARIABLE _already_applied
    ERROR_QUIET OUTPUT_QUIET)
  if (_already_applied EQUAL 0)
    message(STATUS "aurora: Dawn patch ${_name} already applied; skipping")
    continue()
  endif ()
  message(STATUS "aurora: Applying Dawn patch ${_name}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "GIT_CEILING_DIRECTORIES=${_git_ceiling}"
            "${GIT_EXECUTABLE}" apply -p1 "${_patch}"
    WORKING_DIRECTORY "${DAWN_SOURCE_DIR}"
    RESULT_VARIABLE _apply_rc
    ERROR_VARIABLE _apply_err)
  if (NOT _apply_rc EQUAL 0)
    message(FATAL_ERROR
      "aurora: failed to apply Dawn patch ${_name} to '${DAWN_SOURCE_DIR}':\n${_apply_err}\n"
      "The patch stack targets AURORA_DAWN_REF; a pin change needs the patches regenerated.")
  endif ()
endforeach ()

file(WRITE "${_stamp}" "${_patchset_hash}\n")
message(STATUS "aurora: Dawn patch stack applied (${_patches})")
