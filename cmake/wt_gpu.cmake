# GPU backends (experimental).
#
# Included by the top-level CMakeLists.txt, and by builds that embed wave_tracer
# (e.g. the Rust bindings), after creating their wave_tracer target:
#
#   include(<wave_tracer>/cmake/wt_gpu.cmake)
#   wt_add_gpu_backends(<target>)
#
# Options (all OFF by default; without them nothing here is built and no GPU
# toolchain is needed):
#   GPU_CUDA           CUDA backend: runs the shared kernels of include/wt/gpu/kernel
#   GPU_OPTIX          ray queries through OptiX's hardware traversal (needs GPU_CUDA
#                      and the OptiX SDK; see OptiX_INSTALL_DIR)
#   GPU_METAL          Metal backend (macOS on Apple silicon): the shared kernels
#                      compiled as Metal Shading Language 3.2; needs the Metal
#                      toolchain (xcodebuild -downloadComponent MetalToolchain)
#   GPU_DBL_PRECISION  64-bit floating point in GPU kernels (independent of
#                      DBL_PRECISION; OptiX ray queries are then not used; not
#                      available with GPU_METAL)
# One backend per build: GPU_CUDA and GPU_METAL are exclusive.
# The resulting compile definitions (WT_GPU_CUDA, WT_GPU_OPTIX, WT_GPU_METAL,
# WT_GPU_DBL_PRECISION) are PUBLIC on <target>.

option(GPU_CUDA "Build the CUDA GPU backend (experimental)" OFF)
option(GPU_OPTIX "Use OptiX for GPU ray queries (requires GPU_CUDA)" OFF)
option(GPU_METAL "Build the Metal GPU backend (macOS, experimental)" OFF)
option(GPU_DBL_PRECISION "Use 64-bit floating point in GPU kernels" OFF)
set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}" CACHE PATH
    "OptiX SDK directory, containing include/optix.h (searched in /opt/optix* if empty)")

set(WT_GPU_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

if(GPU_OPTIX AND NOT GPU_CUDA)
    message(FATAL_ERROR "GPU_OPTIX requires GPU_CUDA=ON")
endif()

if(GPU_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "GPU_METAL: the Metal backend builds only on macOS")
    endif()
    if(GPU_CUDA)
        message(FATAL_ERROR "GPU_METAL and GPU_CUDA are exclusive: a build has one GPU backend")
    endif()
    if(GPU_DBL_PRECISION)
        message(FATAL_ERROR "GPU_METAL: Metal has no 64-bit floating point; build without GPU_DBL_PRECISION")
    endif()
    # Since Xcode 26 the Metal compiler is a separately downloaded component.
    execute_process(COMMAND xcrun -sdk macosx --find metal
                    OUTPUT_VARIABLE WT_METAL_COMPILER OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE metal_found ERROR_QUIET)
    execute_process(COMMAND xcrun -sdk macosx --find metallib
                    OUTPUT_VARIABLE WT_METALLIB_LINKER OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE metallib_found ERROR_QUIET)
    if(NOT metal_found EQUAL 0 OR NOT metallib_found EQUAL 0)
        message(FATAL_ERROR "GPU_METAL: the Metal toolchain was not found; install it with "
                            "`xcodebuild -downloadComponent MetalToolchain`")
    endif()
    execute_process(COMMAND "${WT_METAL_COMPILER}" --version
                    OUTPUT_VARIABLE metal_version ERROR_QUIET)
    string(REGEX MATCH "metal version [0-9.]+" metal_version "${metal_version}")
    message(STATUS "GPU: Metal (${metal_version}), Metal Shading Language 3.2")
    enable_language(OBJCXX)
endif()

if(GPU_CUDA)
    # enable_language() must run at file scope, and the architectures must be
    # chosen before it: default to the GPUs of the building machine.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "GPU: CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(GPU_OPTIX)
    # CUDA_OPTIX_COMPILATION
    cmake_minimum_required(VERSION 3.27)

    if(NOT OptiX_INSTALL_DIR)
        # Newest first. OptiX SDKs need a recent enough driver: OptiX 9.1 needs
        # a newer one than 9.0; set OptiX_INSTALL_DIR to choose.
        file(GLOB WT_OPTIX_CANDIDATES "/opt/optix*" "/opt/NVIDIA-OptiX-SDK*")
        list(SORT WT_OPTIX_CANDIDATES ORDER DESCENDING)
        foreach(dir ${WT_OPTIX_CANDIDATES})
            if(EXISTS "${dir}/include/optix.h")
                set(OptiX_INSTALL_DIR "${dir}" CACHE PATH "" FORCE)
                break()
            endif()
        endforeach()
    endif()
    if(NOT EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
        message(FATAL_ERROR "GPU_OPTIX: OptiX SDK not found; set OptiX_INSTALL_DIR to its directory")
    endif()
    message(STATUS "GPU: OptiX SDK ${OptiX_INSTALL_DIR}")
endif()

function(wt_add_gpu_backends target)
    set(defines)
    if(GPU_CUDA)
        list(APPEND defines WT_GPU_CUDA)
    endif()
    if(GPU_OPTIX)
        list(APPEND defines WT_GPU_OPTIX)
    endif()
    if(GPU_METAL)
        list(APPEND defines WT_GPU_METAL)
    endif()
    if(GPU_DBL_PRECISION)
        list(APPEND defines WT_GPU_DBL_PRECISION)
    endif()
    if(defines)
        target_compile_definitions(${target} PUBLIC ${defines})
    endif()
    if(GPU_METAL)
        wt_add_metal_backend(${target} "${defines}")
    endif()
    if(NOT GPU_CUDA)
        return()
    endif()

    # The backend is its own library: GPU sources must not get the wave_tracer
    # target's C++20 flags and definitions, and see only the shared kernel
    # headers and wt/gpu/backend.hpp.
    add_library(wt_gpu_cuda STATIC "${WT_GPU_ROOT}/src/gpu/cuda/cuda_backend.cu")
    set_target_properties(wt_gpu_cuda PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(wt_gpu_cuda PRIVATE "${WT_GPU_ROOT}/include")
    target_compile_definitions(wt_gpu_cuda PRIVATE ${defines})
    # The CUDA runtime is linked statically, so applications need only the driver.
    target_link_libraries(wt_gpu_cuda PRIVATE CUDA::cudart_static)

    if(GPU_OPTIX AND NOT GPU_DBL_PRECISION)
        # Device programs, compiled to OptiX-IR and embedded into the library.
        add_library(wt_gpu_optix_ir OBJECT "${WT_GPU_ROOT}/src/gpu/optix/optix_programs.cu")
        set_target_properties(wt_gpu_optix_ir PROPERTIES
            CUDA_OPTIX_COMPILATION ON
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_include_directories(wt_gpu_optix_ir PRIVATE
            "${WT_GPU_ROOT}/include" "${OptiX_INSTALL_DIR}/include")
        target_compile_definitions(wt_gpu_optix_ir PRIVATE ${defines})

        set(embedded "${CMAKE_CURRENT_BINARY_DIR}/wt_optix_programs.cpp")
        add_custom_command(
            OUTPUT "${embedded}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=$<TARGET_OBJECTS:wt_gpu_optix_ir>"
                "-DOUTPUT=${embedded}"
                -DNAME=wt_optix_programs
                -P "${WT_GPU_ROOT}/cmake/wt_embed_file.cmake"
            DEPENDS wt_gpu_optix_ir $<TARGET_OBJECTS:wt_gpu_optix_ir> "${WT_GPU_ROOT}/cmake/wt_embed_file.cmake"
            VERBATIM)

        target_sources(wt_gpu_cuda PRIVATE "${WT_GPU_ROOT}/src/gpu/optix/optix_rays.cpp" "${embedded}")
        target_include_directories(wt_gpu_cuda PRIVATE "${OptiX_INSTALL_DIR}/include")
        # optixInit() loads the OptiX implementation from the driver.
        target_link_libraries(wt_gpu_cuda PRIVATE ${CMAKE_DL_LIBS})
    elseif(GPU_OPTIX)
        message(STATUS "GPU: GPU_DBL_PRECISION is ON, so ray queries use the CUDA traversal instead of OptiX")
    endif()

    target_link_libraries(${target} PRIVATE wt_gpu_cuda)
endfunction()


# The Metal backend: src/gpu/metal/*.metal compiled into a metallib embedded in
# the backend library, and the Objective-C++ host code.
function(wt_add_metal_backend target defines)
    set(metal_dir "${WT_GPU_ROOT}/src/gpu/metal")
    set(kernel_sources geometry)
    # Kernels need IEEE-conforming arithmetic (wt/gpu/kernel/math.hpp): no fast math.
    set(metal_flags -std=metal3.2 -fno-fast-math -Wno-c++17-extensions
                    -mmacosx-version-min=15.0
                    "-I${WT_GPU_ROOT}/include" "-I${metal_dir}")
    set(air_files)
    foreach(name ${kernel_sources})
        set(air "${CMAKE_CURRENT_BINARY_DIR}/wt_metal/${name}.air")
        add_custom_command(
            OUTPUT "${air}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/wt_metal"
            COMMAND "${WT_METAL_COMPILER}" ${metal_flags} -MMD -MF "${air}.d"
                    -c "${metal_dir}/${name}.metal" -o "${air}"
            DEPENDS "${metal_dir}/${name}.metal"
            DEPFILE "${air}.d"
            COMMENT "Compiling Metal kernels ${name}.metal"
            VERBATIM)
        list(APPEND air_files "${air}")
    endforeach()

    set(metallib "${CMAKE_CURRENT_BINARY_DIR}/wt_metal/wt_metal_kernels.metallib")
    add_custom_command(
        OUTPUT "${metallib}"
        COMMAND "${WT_METALLIB_LINKER}" ${air_files} -o "${metallib}"
        DEPENDS ${air_files}
        COMMENT "Linking the Metal kernels"
        VERBATIM)
    set(embedded "${CMAKE_CURRENT_BINARY_DIR}/wt_metal/wt_metal_kernels.cpp")
    add_custom_command(
        OUTPUT "${embedded}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${metallib}" "-DOUTPUT=${embedded}" -DNAME=wt_metal_kernels
                -P "${WT_GPU_ROOT}/cmake/wt_embed_file.cmake"
        DEPENDS "${metallib}" "${WT_GPU_ROOT}/cmake/wt_embed_file.cmake"
        VERBATIM)

    # Its own library, as the CUDA backend: it sees only the shared kernel
    # headers and wt/gpu/backend.hpp, not the wave_tracer target's settings.
    add_library(wt_gpu_metal STATIC "${metal_dir}/metal_backend.mm" "${embedded}")
    set_target_properties(wt_gpu_metal PROPERTIES
        OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(wt_gpu_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_include_directories(wt_gpu_metal PRIVATE "${WT_GPU_ROOT}/include")
    target_compile_definitions(wt_gpu_metal PRIVATE ${defines})
    target_link_libraries(wt_gpu_metal PRIVATE "-framework Metal" "-framework Foundation")

    target_link_libraries(${target} PRIVATE wt_gpu_metal)
endfunction()
