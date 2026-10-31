
Building and using
=======================================

GPU support is optional twice over: at compile time, the GPU backends are built only when their CMake options are set; at run time, a build with them still renders on the CPU unless a GPU is asked for.


Requirements
------------

For the CUDA backend:

* An NVIDIA GPU and its driver. The CUDA runtime is linked statically, so the driver is all that is needed to run.
* To build: the CUDA toolkit (``nvcc``), with a host compiler it accepts.
* For OptiX: CMake 3.27 or newer and the `OptiX SDK <https://developer.nvidia.com/designworks/optix/download>`_ (headers only; OptiX itself ships with the driver).
  Every OptiX SDK needs a driver at least as new as its release: OptiX 9.1, for example, needs a newer driver than 9.0. With an older driver, build against an older SDK.

For the Metal backend:

* macOS 15 or later, on Apple silicon: an M1 or later GPU (Metal GPU family Apple7, for float atomics and SIMD-group functions). Ray queries use the GPU's ray-tracing hardware on M3 and later.
* To build: Xcode or its Command Line Tools, and the Metal toolchain, which Xcode 26 and later download separately: ``xcodebuild -downloadComponent MetalToolchain`` (the Rust bindings' ``just install-metal-toolchain``). The kernels are compiled as Metal Shading Language 3.2 and embedded in the library, so nothing else is needed to run.

The GPU code was developed and measured on an NVIDIA GB10 (compute capability 12.1) with CUDA 13.0 and OptiX 9.0, and on an Apple M4 (macOS 27.2).


Build configurations
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 32 22 28

   * - Configuration
     - CMake options
     - Rust bindings features
     - The GPU it reports
   * - CPU only (default)
     - none
     - none
     - ``--gpu`` fails: *this build of wave_tracer has no GPU backend*
   * - CUDA
     - ``-DGPU_CUDA=ON``
     - ``gpu-cuda``
     - ``CUDA on <GPU> (compute capability <x.y>)``
   * - CUDA and OptiX
     - ``-DGPU_CUDA=ON -DGPU_OPTIX=ON``
     - ``gpu-optix``
     - ``CUDA + OptiX <version> on <GPU> (...)``
   * - 64-bit GPU kernels
     - add ``-DGPU_DBL_PRECISION=ON``
     - add ``gpu-double-precision``
     - ``CUDA, 64-bit on <GPU> (...)``
   * - Metal (macOS)
     - ``-DGPU_METAL=ON``
     - ``gpu-metal``
     - ``Metal + hardware ray tracing on <GPU>`` (M3 and later), ``Metal on <GPU>``

``GPU_CUDA``
   Builds the CUDA backend. Kernels are compiled for the GPUs of the building machine (``CMAKE_CUDA_ARCHITECTURES=native``), unless ``CMAKE_CUDA_ARCHITECTURES`` is set, e.g. to ``86;89;120`` to build for other GPUs.

``GPU_OPTIX``
   Requires ``GPU_CUDA``. Traces ray queries with OptiX's hardware ray tracing: the batched ray queries of the GPU geometry API, and ``plt_bdpt``'s connection shadow rays.
   The rendering kernels themselves traverse the BVH with CUDA code in both configurations (see :doc:`design`), so OptiX speeds up only ``plt_bdpt`` renders (see :doc:`performance`).
   The SDK is searched for in ``/opt/optix*`` and ``/opt/NVIDIA-OptiX-SDK*`` (newest first), or given with ``-DOptiX_INSTALL_DIR=<dir>`` or the environment variable of the same name.

``GPU_METAL``
   Builds the Metal backend, on macOS only; exclusive with ``GPU_CUDA`` (a build has one GPU backend) and with ``GPU_DBL_PRECISION``.
   The kernels (``src/gpu/metal/*.metal``, including the shared ``include/wt/gpu/kernel/`` headers) are compiled with the Metal toolchain (``xcrun metal``) into a metallib embedded in the library.
   Ray queries, of the geometry API and of the rendering kernels, use Metal's hardware ray tracing on GPUs that have it (M3 and later); there is no build option for it.
   Metal has no 64-bit floating point: films accumulate in 32-bit floats on the GPU, added into double-precision films on the host every 64 samples per element (see :doc:`design`).

``GPU_DBL_PRECISION``
   Computes in 64-bit floating point in the GPU kernels, independently of the CPU's ``DBL_PRECISION``.
   OptiX is then not used, as its traversal is 32-bit.
   Most GPUs compute 64-bit floating point at a small fraction of their 32-bit rate: on the GB10, renders take 3 to 8 times longer.
   Use it to check whether a difference between GPU and CPU renders comes from floating-point precision.

Builds that embed wave_tracer (such as the Rust bindings) add the GPU backends to their wave_tracer target with:

.. code-block:: cmake

   include(<wave_tracer>/cmake/wt_gpu.cmake)
   wt_add_gpu_backends(<target>)

which honours the same options and defines ``WT_GPU_CUDA``, ``WT_GPU_OPTIX``, ``WT_GPU_METAL`` and ``WT_GPU_DBL_PRECISION`` on the target.


Command line
------------

``render`` and ``renderui`` take ``--gpu``, which renders every sensor of the scene on the first CUDA device:

.. code-block:: sh

   ./wave_tracer render --gpu scene.xml -o out -D spp=256
   ./wave_tracer render --gpu --ray-tracing scene.xml -o out
   ./wave_tracer render --gpu --no-optix scene.xml -o out    # CUDA builds
   ./wave_tracer render --gpu --no-hw-rt scene.xml -o out    # Metal builds
   CUDA_VISIBLE_DEVICES=1 ./wave_tracer render --gpu scene.xml   # another GPU

* ``--ray-tracing`` renders in ray-tracing mode, as on the CPU.
* ``--no-optix`` traces rays with the CUDA traversal, as builds without ``GPU_OPTIX`` do, rather than with OptiX; the program then reports the GPU as ``CUDA on <GPU>``. Use it to compare the two, or to rule OptiX out when investigating a difference: OptiX may report either triangle when a ray grazes an edge or vertex they share. It has no effect in builds without OptiX.
* ``--no-hw-rt``, in Metal builds, traces rays with the ported BVH traversal rather than with Metal's hardware ray tracing; the program then reports the GPU as ``Metal on <GPU>``. As with OptiX, rays grazing an edge or vertex may hit either triangle. It has no effect on GPUs without ray-tracing hardware (before M3).
* Outputs, previews (``--tev``, and the GUI of ``renderui``), progress, pausing and termination work as on the CPU. They update between the GPU's *passes*: a pass renders whole samples of all sensor elements, about :math:`2^{22}` paths (see :doc:`design`).
* ``--block_size`` and ``--block_samples`` do not apply; ``--threads`` sets the CPU threads that load the scene and build its acceleration structure.
* The reported render time (``Elapsed``) excludes loading the scene and building its BVH (both on the CPU), opening the GPU, and compiling and uploading the scene to it.


C++
---

The GPU is a ``wt::gpu::device_t``, given to the renderer through the context; a ``scene_renderer_t`` renders on ``wt_context_t::gpu_device`` whenever it is set:

.. code-block:: cpp

   #include <wt/gpu/gpu.hpp>

   // throws wt::gpu::gpu_error_t, explaining why, if the build or the machine has no usable GPU
   auto gpu = wt::gpu::device_t::create({ .ordinal = 0, .use_hardware_ray_tracing = true });
   std::cout << gpu->description() << '\n';

   wt::wt_context_t context;
   // ... thread pool, paths, etc., as for the CPU
   context.gpu_device = gpu.get();
   context.renderer_force_ray_tracing = false;

   // load the scene with `context`, then render it with a scene_renderer_t, as on the CPU

* ``wt::gpu::compiled_backends()`` names the backends a build includes (``"CUDA, OptiX"``, ``"CUDA"``, ``"Metal"`` or ``""``).
* ``use_hardware_ray_tracing = false`` traces rays with the ported BVH traversal rather than with OptiX (the command line's ``--no-optix``) or Metal's ray tracing (``--no-hw-rt``). ``ordinal`` is a CUDA device ordinal, or an index into Metal's devices.
* Constructing a ``wt::gpu::renderer_t`` (done by ``scene_renderer_t`` for each sensor) compiles the scene; it throws ``gpu_error_t`` naming the first unsupported scene element.
* ``wt::gpu::geometry_t`` (from ``device_t::upload``) answers batches of the acceleration structure's queries on the GPU (closest-hit and shadow rays, elliptic cones and balls), with the CPU's results up to floating-point rounding.
* A render uploads the scene's acceleration structure once, for all its sensors. An application that has already uploaded it, for its own queries, can give that upload to its renders, which then share it rather than upload the scene again:

  .. code-block:: cpp

     std::shared_ptr<const wt::gpu::geometry_t> geometry = gpu->upload(ads);
     // ... geometry->intersect(rays, range), etc.

     wt::scene::render_opts_t opts;
     opts.gpu_geometry = geometry;   // must be `ads` uploaded to context.gpu_device
     wt::scene_renderer_t renderer{ scene, context, ads, std::launch::async, std::move(opts) };

  Queries and renders may then run concurrently; queries wait for the GPU work queued before them.
* Destroy the ``device_t`` before ``main`` returns: the statically linked CUDA runtime shuts down before global objects are destroyed.


Rust bindings
-------------

The `Rust bindings <https://github.com/toastertaster/wave_tracers>`_ build the GPU backends with the features ``gpu-cuda``, ``gpu-optix``, ``gpu-double-precision`` and ``gpu-metal``. A context uses the GPU only when asked:

.. code-block:: rust

   use wave_tracers::{Context, Device, Scene};

   let context = Context::builder()
       .device(Device::Gpu)
       .ray_tracing_only(false)
       .build()?;
   println!("{:?}", context.gpu()); // Some("CUDA + OptiX 9.0 on NVIDIA GB10 (compute capability 12.1)")
   let scene = Scene::load(&context, "scene.xml", &[("spp", "64")])?;
   let result = scene.start_render()?.wait()?;

``wave_tracers::gpu_backends()`` names the compiled backends, and ``ContextBuilder::hardware_ray_tracing(false)`` turns OptiX or Metal's ray tracing off (the command line's ``--no-optix`` and ``--no-hw-rt``).
A scene loaded with a GPU context uploads its geometry once, for its geometry queries, and its renders share that upload.
The bindings' examples take ``--gpu``, ``--no-hw-rt`` (or ``--no-optix``) and ``--ray-tracing`` (``examples/render.rs``), and ``examples/gpu_compare.rs`` compares a scene's GPU and CPU renders statistically.


Supported scenes
----------------

The GPU renders scenes made of the following elements. Using any other element fails before rendering, with a ``gpu_error_t`` that names it.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Element
     - Supported on the GPU
   * - Integrators
     - :doc:`../scenes/integrators/plt_path`, forward and backward, with beams (and free-space diffraction) or in ray-tracing mode; :doc:`../scenes/integrators/plt_bdpt`, with beams (and its Fraunhofer free-space diffraction) or in ray-tracing mode. Their options (maximal depth, Russian roulette, MIS, direct sampling, FSD) apply.
   * - Sensors
     - ``virtual_plane`` (coverage) sensors; ``perspective`` sensors, scalar or polarimetric (Stokes parameters), with backward ``plt_path`` or ``plt_bdpt``. Any response function (monochromatic, RGB, XYZ, multichannel; up to 4 channels), and reconstruction filters of a limited radius. Sensors must source isotropic beams.
   * - Emitters
     - ``point``, ``spot``, ``directional``, and ``area`` emitters of a constant radiance. Any emission spectra.
   * - BSDFs
     - ``surface_spm`` with a smooth (Dirac) or rough (``fractal`` or ``gaussian``) surface profile; ``dielectric``; ``diffuse``; wrapped, optionally, in ``twosided``, ``composite``, ``scale`` and ``normalmap``.
   * - Textures
     - ``constant`` and ``bitmap`` (filtered and wrapped as on the CPU), optionally ``transform``-ed or ``scale``-d by a constant, for diffuse reflectances, BSDF scales, surface roughness (perceptual roughness) and normal maps. Bitmaps are uploaded as the CPU stores them (8- and 16-bit PNGs as integers, EXRs as 32-bit floats), up to 16 GiB in all.
   * - Acceleration structure
     - :doc:`../ads/bvh8w`.

For example, ``scenes/sionna_etoile`` (both its coverage map and its optical preview), ``scenes/veach_mis``, ``scenes/cornell-box``, ``scenes/bidir_room``, ``scenes/colourchecker``, ``scenes/diffraction_simple`` and ``scenes/sionna_munich`` render on the GPU, while ``scenes/cornell-box/box_empty.xml`` fails with:

.. code-block:: text

   GPU rendering supports only bitmap, constant, scale and transform textures so far; BSDF <...> has a 'function' texture

At run time, the GPU stops with an error if a BVH traversal overflows its stack, and warns if paths are stopped after :math:`2^{14}` steps (a safeguard against endless paths).
