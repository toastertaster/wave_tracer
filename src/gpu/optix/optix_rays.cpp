/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#include "optix_rays.hpp"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "optix_params.hpp"
#include "../cuda/cuda_util.hpp"

// The device programs, compiled to OptiX-IR (cmake/wt_embed_file.cmake).
extern const unsigned char wt_optix_programs[];
extern const std::size_t wt_optix_programs_size;

using namespace wt::gpu;
using namespace wt::gpu::optix;
using wt::gpu::cuda::check;
using wt::gpu::cuda::device_buffer_t;

namespace {

void check_optix(OptixResult result, const char* what, const char* log = nullptr) {
    if (result==OPTIX_SUCCESS)
        return;
    std::string message = std::string("OptiX error ") + what + ": " + optixGetErrorName(result) +
                          " (" + optixGetErrorString(result) + ")";
    if (log && *log)
        message += "\n" + std::string(log);
    throw std::runtime_error(message);
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) sbt_record_t {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

CUdeviceptr device_ptr(const void* p) noexcept { return CUdeviceptr(reinterpret_cast<std::uintptr_t>(p)); }

}


struct context_t::impl_t {
    int ordinal;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygen_closest = nullptr, raygen_shadow = nullptr, miss = nullptr, hit = nullptr;
    OptixPipeline pipeline = nullptr;

    device_buffer_t<sbt_record_t> records;   // raygen closest, raygen shadow, miss, hit group
    OptixShaderBindingTable sbt_closest{}, sbt_shadow{};

    ~impl_t() noexcept {
        if (pipeline) optixPipelineDestroy(pipeline);
        for (auto* group : { raygen_closest, raygen_shadow, miss, hit })
            if (group) optixProgramGroupDestroy(group);
        if (module) optixModuleDestroy(module);
        if (context) optixDeviceContextDestroy(context);
    }
};

context_t::context_t(int ordinal) : pimpl(std::make_unique<impl_t>()) {
    auto& p = *pimpl;
    p.ordinal = ordinal;

    cuda::set_device(ordinal);
    check(cudaFree(nullptr), "initializing the device");
    check_optix(optixInit(), "initializing OptiX (the driver may be too old for this OptiX SDK)");
    OptixDeviceContextOptions context_options{};
    check_optix(optixDeviceContextCreate(nullptr, &context_options, &p.context), "creating the context");

    // module
    OptixModuleCompileOptions module_options{};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options{};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 5;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[2048];
    std::size_t log_size = sizeof(log);
    log[0] = 0;
    check_optix(optixModuleCreate(p.context, &module_options, &pipeline_options,
                                  reinterpret_cast<const char*>(wt_optix_programs), wt_optix_programs_size,
                                  log, &log_size, &p.module),
                "creating the module", log);

    // program groups
    OptixProgramGroupOptions group_options{};
    const auto create_group = [&](const OptixProgramGroupDesc& desc, OptixProgramGroup& group) {
        log_size = sizeof(log);
        log[0] = 0;
        check_optix(optixProgramGroupCreate(p.context, &desc, 1, &group_options, log, &log_size, &group),
                    "creating a program group", log);
    };
    OptixProgramGroupDesc desc{};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = p.module;
    desc.raygen.entryFunctionName = "__raygen__closest";
    create_group(desc, p.raygen_closest);
    desc.raygen.entryFunctionName = "__raygen__shadow";
    create_group(desc, p.raygen_shadow);

    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = p.module;
    desc.miss.entryFunctionName = "__miss__ray";
    create_group(desc, p.miss);

    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleCH = p.module;
    desc.hitgroup.entryFunctionNameCH = "__closesthit__ray";
    create_group(desc, p.hit);

    // pipeline
    const OptixProgramGroup groups[] = { p.raygen_closest, p.raygen_shadow, p.miss, p.hit };
    OptixPipelineLinkOptions link_options{};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    log[0] = 0;
    check_optix(optixPipelineCreate(p.context, &pipeline_options, &link_options, groups, 4,
                                    log, &log_size, &p.pipeline),
                "creating the pipeline", log);

    OptixStackSizes stack_sizes{};
    for (auto* group : groups)
        check_optix(optixUtilAccumulateStackSizes(group, &stack_sizes, p.pipeline), "sizing the stack");
    unsigned dc_from_traversal, dc_from_state, continuation;
    check_optix(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0,
                                           &dc_from_traversal, &dc_from_state, &continuation),
                "sizing the stack");
    check_optix(optixPipelineSetStackSize(p.pipeline, dc_from_traversal, dc_from_state, continuation, 1),
                "setting the stack size");

    // shader binding tables: the two ray kinds differ only in their raygen record
    sbt_record_t records[4];
    check_optix(optixSbtRecordPackHeader(p.raygen_closest, &records[0]), "packing a record");
    check_optix(optixSbtRecordPackHeader(p.raygen_shadow,  &records[1]), "packing a record");
    check_optix(optixSbtRecordPackHeader(p.miss,           &records[2]), "packing a record");
    check_optix(optixSbtRecordPackHeader(p.hit,            &records[3]), "packing a record");
    p.records.upload(records, 4);

    const auto base = device_ptr(p.records.get());
    for (auto* sbt : { &p.sbt_closest, &p.sbt_shadow }) {
        sbt->missRecordBase = base + 2*sizeof(sbt_record_t);
        sbt->missRecordStrideInBytes = sizeof(sbt_record_t);
        sbt->missRecordCount = 1;
        sbt->hitgroupRecordBase = base + 3*sizeof(sbt_record_t);
        sbt->hitgroupRecordStrideInBytes = sizeof(sbt_record_t);
        sbt->hitgroupRecordCount = 1;
    }
    p.sbt_closest.raygenRecord = base;
    p.sbt_shadow.raygenRecord  = base + sizeof(sbt_record_t);
}

context_t::~context_t() noexcept = default;

std::string context_t::version() const {
    return std::to_string(OPTIX_VERSION/10000) + "." + std::to_string((OPTIX_VERSION%10000)/100);
}


struct ray_geometry_t::impl_t {
    const context_t::impl_t* context;
    device_buffer_t<float> vertices;
    device_buffer_t<unsigned char> gas;
    OptixTraversableHandle handle = 0;
    const kernel::bvh_tri* tris = nullptr;     // owned by the CUDA geometry
    device_buffer_t<launch_params_t> params;

    void launch(const OptixShaderBindingTable& sbt, launch_params_t p, std::size_t count) {
        params.upload(&p, 1);
        check_optix(optixLaunch(context->pipeline, nullptr, device_ptr(params.get()), sizeof(launch_params_t),
                                &sbt, unsigned(count), 1, 1),
                    "launching ray queries");
    }
};

ray_geometry_t::ray_geometry_t(const context_t& context, const bvh_arrays_t& bvh,
                               const kernel::bvh_tri* device_tris)
    : pimpl(std::make_unique<impl_t>())
{
    auto& p = *pimpl;
    p.context = &context.impl();
    p.tris = device_tris;
    cuda::set_device(p.context->ordinal);

    // one vertex triple per triangle, in the BVH's triangle order
    std::vector<float> vertices;
    vertices.reserve(bvh.tris.size()*9);
    for (const auto& t : bvh.tris)
        for (const auto& v : { t.a, t.b, t.c })
            vertices.insert(vertices.end(), { v.x, v.y, v.z });
    p.vertices.upload(vertices.data(), vertices.size());

    const CUdeviceptr vertex_buffer = device_ptr(p.vertices.get());
    const unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = 3*sizeof(float);
    input.triangleArray.numVertices = unsigned(bvh.tris.size()*3);
    input.triangleArray.vertexBuffers = &vertex_buffer;
    input.triangleArray.flags = &flags;
    input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes{};
    check_optix(optixAccelComputeMemoryUsage(p.context->context, &options, &input, 1, &sizes),
                "sizing the acceleration structure");
    device_buffer_t<unsigned char> temp;
    temp.resize(sizes.tempSizeInBytes);
    p.gas.resize(sizes.outputSizeInBytes);
    check_optix(optixAccelBuild(p.context->context, nullptr, &options, &input, 1,
                                device_ptr(temp.get()), sizes.tempSizeInBytes,
                                device_ptr(p.gas.get()), sizes.outputSizeInBytes,
                                &p.handle, nullptr, 0),
                "building the acceleration structure");
    check(cudaDeviceSynchronize(), "building the acceleration structure");
}

ray_geometry_t::~ray_geometry_t() noexcept = default;

void ray_geometry_t::intersect(const ray_query_t* queries, std::size_t count, kernel::ray_hit* hits) const {
    pimpl->launch(pimpl->context->sbt_closest,
                  { pimpl->handle, queries, pimpl->tris, hits, nullptr }, count);
}

void ray_geometry_t::occluded(const ray_query_t* queries, std::size_t count, std::uint8_t* occluded) const {
    pimpl->launch(pimpl->context->sbt_shadow,
                  { pimpl->handle, queries, pimpl->tris, nullptr, occluded }, count);
}
