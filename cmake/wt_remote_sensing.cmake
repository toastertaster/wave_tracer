# Realistic remote sensing: Earth-observation scenes, atmospheres and detector
# models, developed on the fork's remote-sensing branch. Unlikely to be merged
# upstream, so all of it is compiled only with this option.
#
# Included by the top-level CMakeLists.txt, and by builds that embed wave_tracer
# (e.g. the Rust bindings), after creating their wave_tracer target and before
# cmake/wt_gpu.cmake's wt_add_gpu_backends(), which reads the option:
#
#   include(<wave_tracer>/cmake/wt_remote_sensing.cmake)
#   wt_add_remote_sensing(<target>)
#
# Option (OFF by default; without it nothing here is built):
#   REALISTIC_REMOTE_SENSING  compiles src/remote_sensing/ into <target> and
#                             defines REALISTIC_REMOTE_SENSING, PUBLIC on <target>
#                             and in the GPU backends and kernels
#
# Code outside src/ and include/wt/remote_sensing/ that only remote sensing
# needs is wrapped in #if defined(REALISTIC_REMOTE_SENSING). Embedding builds
# that compile wave_tracer's sources by globbing must exclude src/remote_sensing/.

option(REALISTIC_REMOTE_SENSING "Build the realistic remote sensing features" OFF)

set(WT_REMOTE_SENSING_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

function(wt_add_remote_sensing target)
    if(NOT REALISTIC_REMOTE_SENSING)
        return()
    endif()
    file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${WT_REMOTE_SENSING_ROOT}/src/remote_sensing/*.cpp")
    target_sources(${target} PRIVATE ${sources})
    target_compile_definitions(${target} PUBLIC REALISTIC_REMOTE_SENSING)
    message(STATUS "Realistic remote sensing: ON")
endfunction()
