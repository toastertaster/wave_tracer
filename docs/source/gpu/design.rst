
Design
=======================================

The GPU renderer ports **wave_tracer**'s light transport, rather than approximating it: its kernels trace the CPU's beams (elliptic cones), with the same BVH traversals, interaction models, free-space diffraction and sampling decisions.
What differs is how the work is organized, so that thousands of paths run at once, and the random numbers the paths draw.
The CPU still loads the scene, builds its BVH, and develops and writes the rendered films.


Code layout
-----------

``include/wt/gpu/kernel/``
   The kernels: math, ray/triangle and cone/triangle intersections, the ``bvh8w`` traversals (``bvh.hpp``), beams (``beam.hpp``), polarimetry, free-space diffraction (``fsd.hpp``, ``fraunhofer.hpp``), textures, and the integrators (``plt_path.hpp``, ``plt_bdpt.hpp``).
   They are written in a subset of C++17, without wave_tracer's unit types or the standard library (see ``config.hpp``), so that the same headers compile with ``nvcc``, as Metal Shading Language 3.2, and as host C++.
   Metal requires address spaces on pointers and references (``WT_GPU_GLOBAL``, ``WT_GPU_THREAD``, ``WT_GPU_CONSTANT``), and namespaces declared one level at a time.
   ``real`` is ``float``, or ``double`` with ``GPU_DBL_PRECISION`` (CUDA only).

``include/wt/gpu/gpu.hpp``, ``include/wt/gpu/renderer.hpp``
   The API the rest of wave_tracer uses: ``gpu::device_t``, ``gpu::geometry_t`` (batched acceleration-structure queries) and ``gpu::renderer_t`` (renders a sensor).
   ``src/gpu/renderer.cpp`` compiles a scene into the kernels' plain structs (``kernel/scene.hpp``), refusing what the kernels do not implement.

``include/wt/gpu/backend.hpp``, ``src/gpu/cuda/``, ``src/gpu/optix/``
   The interface a backend implements, and the CUDA backend with its OptiX ray tracing.

``src/gpu/metal/``
   The Metal backend (Apple silicon): the kernels' entry points (``geometry.metal``, ``plt.metal``), compiled into a metallib embedded in the library; the host side in Objective-C++ (``metal_backend.mm``, ``metal_renderer.mm``); the structs the two exchange (``plt_shared.hpp``), whose layouts the backend checks at start-up against the Metal compiler's (``layout.hpp``); and ray queries with Metal's ray tracing (``rt_rays.hpp``).
   Its host code mirrors the CUDA backend's, with the differences described below.

``cmake/wt_gpu.cmake``
   The build options, shared by builds that embed wave_tracer.


Rendering loop
--------------

``scene_renderer_t`` renders on the GPU when the context has a GPU device.
For each sensor, it constructs a ``gpu::renderer_t``, which compiles the scene for that sensor and uploads it, then renders the sensor in **passes**.
A pass renders whole samples of every sensor element: :math:`\max(1, \lfloor 2^{22} / \text{elements} \rfloor)` samples per element, about four million paths, so a 256×256 film renders 64 samples per pixel in a pass and a 2048×2048 film one.
Passes run one after another; between them, the renderer handles interrupts (pause, resume, terminate, capture), updates progress and previews.
Interrupts wait for the current pass to end: with an expensive scene, a pass can take seconds or more (see :doc:`performance`).

Every path is one sample of one element: path :math:`s \cdot \text{elements} + e` is sample :math:`s` of element :math:`e` (forward paths, which start at emitters, ignore their element, as on the CPU).
The GPU accumulates the paths' splats into films (with atomic additions), as the CPU's integrators splat them: backward ``plt_path`` into image blocks with reconstruction weights, forward ``plt_path`` directly, and ``plt_bdpt`` both, the latter into a separate light image.
With CUDA, the films are in double precision on the GPU.
Metal has no 64-bit floating point, so the Metal backend accumulates in 32-bit floats on the GPU, and adds them into double-precision films on the host (see `Precision and limits`_).
When a render completes, or a preview or capture is asked for, the films are downloaded into the sensor's film storage, and developed, tonemapped and written by the CPU's code.


Random numbers
--------------

Each path draws from a counter-based generator, Philox4x32-10, keyed by the path's index.
A path's random numbers therefore do not depend on which thread runs it, or when; a path that must be traced again (see below) replays exactly.
The CPU's samplers (``uniform``, ``sobolld``) are not used on the GPU, so GPU and CPU renders agree statistically, not pixel for pixel.

Paths draw their wavenumbers as the CPU's do (``scene_t::sample_spectrum``), from tables the CPU prepares for each pass: 4096 stratified draws, divided equally among the emitters, each with what depends on the wavenumber compiled (emitted power, the sensor's response, materials' indices of refraction, RGB-to-spectrum uplift).
A path picks one of its emitter's draws at random.
Stratification makes the draws cover the spectrum evenly, so that a pass's common spectral error stays small.
Sensors sensitive to a single wavelength use a single draw for all passes.


Executing paths
---------------

Paths differ greatly in their work: their lengths, the number of triangles their beams overlap, free-space diffraction's rejection sampling.
The backend organizes them so that the GPU keeps busy despite that:

**Beams of plt_path: a wavefront.**
   A pool of :math:`2^{18}` paths in flight advances a step at a time, by two kernels: one traces the paths' beams (BVH cone queries), the other computes their interactions with what they hit (scattering, free-space diffraction, splats).
   Splitting the traversal from the interaction code keeps the traversal's register use low and its warps coherent.
   Most paths end after a step or two; their slots are refilled with new paths, of the same pass or of the next, so passes overlap and the GPU does not wait for a pass's longest paths.
   When at most 1024 paths remain, one kernel finishes them in a single launch, rather than two launches per step.

**Ray tracing and plt_bdpt: whole paths, by persistent warps.**
   A ray-traced step is too cheap to be worth storing a path's state between kernel launches, and a ``plt_bdpt`` path holds two subpaths of vertices.
   These paths run whole: a launch starts as many threads as the GPU can run at once (``plt_bdpt``: as many as 512 MiB of vertex memory allows), and each warp takes the next 32 paths from a global counter once all its lanes have finished their previous ones.
   Launches trace up to :math:`2^{20}` paths each.
   On Metal, SIMD-groups (of 32 lanes) take paths as warps do, with ``simd_ballot`` and ``simd_broadcast``; Metal reports no occupancy, so a launch runs 2048 threads per GPU core (the core count is read from the IORegistry), more than a core runs at once.
   ``plt_bdpt``'s vertex memory is a quarter of the GPU's recommended working set, up to 2 GiB, as unified memory affords it.
   macOS stops command buffers that run too long, so Metal launches are sized, from the previous launch's rate, to take about a quarter of a second of GPU time.

**Free-space diffraction's rejection sampling** shares its tries among a warp's (or SIMD-group's) lanes, so that a lane whose tries are rejected does not hold back the others.

**Scratch memory.**
   A beam query returns all the triangles and edges within the beam's footprint, into a per-path buffer with room for 64.
   A path whose query finds more stops, and is recorded with the room that query needed (rounded up to a power-of-four multiple of 64, up to :math:`2^{20}` entries); after the pass, it is traced again from its start, with the same random numbers, skipping the splats its earlier attempt made (``plt_bdpt`` paths that overflow splat nothing).
   Re-traces run in a pool of paths asking for the same room, the least first; a path that overflows again, at a later query, asks for more and joins those paths. Paths that ask for 1024 entries or more (in ``sphere_polarization`` at 128×128 pixels and 8 samples per pixel, 18,829 of 131,072) are traced by the CPU's threads instead, with the kernels' own code compiled as host C++ (``src/gpu/cpu_paths.hpp``), while the GPU re-traces the others: their beams overlap so much geometry that a GPU thread takes seconds for one, and their memory leaves room for few at once, so that the GPU's last re-traces waited on a handful of them, while a CPU core traces them many times faster.
   They are the paths the GPU would have traced: on the M4, a ``sphere_polarization`` render whose 18,000 heaviest paths the CPU traced differs from a GPU-only one by a relative L1 difference of :math:`10^{-5}`.
   The CUDA backend keeps the compiled scene and the BVH in host memory for them; the Metal backend reads its buffers, in unified memory, from the CPU.

**plt_bdpt connections.**
   ``plt_bdpt`` connects the vertices of its sensor and emitter subpaths, each connection needing a shadow ray.
   With OptiX, a launch defers these shadow rays, with the splats they would make, into a batch of up to :math:`2^{22}`; OptiX traces the batch on the GPU's ray-tracing cores, and a kernel then makes the splats of the unoccluded connections.
   Without OptiX, paths trace their shadow rays themselves, with the CUDA traversal.

Everything else (cone and ray traversals of the paths, edge queries of free-space diffraction) uses the CUDA port of ``bvh8w``: the same node layout and visiting order as the CPU's, with leaves encoded in their parents' child pointers.
OptiX cannot replace these traversals, as beams are cones, not rays, and OptiX rays cannot be traced from within the long-running path kernels.

**Hardware ray tracing on Metal.**
   On GPUs with ray-tracing hardware (Apple M3 and later; earlier Apple GPUs emulate Metal's ray tracing, and the backend does not use it), the Metal backend builds a primitive acceleration structure of the BVH's triangles, in the BVH's order, so that a hit's primitive is its BVH triangle.
   Unlike OptiX's, Metal's ray queries (``intersector``) run within any compute kernel, so the path kernels trace their rays with it directly, without deferring them: the closest hits of ballistic segments and of ray tracing, and shadow rays (``plt_bdpt``'s connections, next-event estimation), as well as the geometry API's ray queries.
   Cone queries still traverse the BVH.
   The path kernels' pipelines are specialized by a function constant, ``hardware_rays``, so that each is compiled with either kind of ray query.
   Hits agree with the BVH traversal up to floating-point rounding, as OptiX's do, but for one case: ``plt_path``'s free-space diffraction of beams tests the visibility of the point a beam hit, on its surface, and whether such a shadow ray is blocked by the surface it ends on turns on the rounding of the ray-triangle test, which the BVH traversal shares with the CPU and the hardware does not. Those shadow rays, of ``plt_path`` with beams and free-space diffraction, use the BVH traversal.
   ``--no-hw-rt`` (``use_hardware_ray_tracing = false``) turns hardware ray tracing off.


Precision and limits
--------------------

* Kernels compute in 32-bit floating point by default, as the CPU's default build does; ``GPU_DBL_PRECISION`` switches them to 64 bits (CUDA only: Apple GPUs have no 64-bit floating point). Films accumulate in 64 bits either way.
* **Films on Metal (macOS).** Metal has no 64-bit floating point, neither arithmetic nor atomics, so the Metal backend's kernels splat into 32-bit float films with float atomic additions.
  Between launches, while the GPU is idle, the host adds these float films into its double-precision films and clears them: after every 64 samples per film element (paths started, on average; launches of whole paths trace no more than that), when the render finishes, and before a film is read (previews, captures).
  A float element therefore sums only a bounded number of splats, about 64 samples' worth (for films smaller than about 64×64 elements, a launch's worth), before its sum moves to double precision, rather than all of a render's samples; the relative rounding error of a flushed sum is at most of the order of :math:`64 \cdot 2^{-24} \approx 4 \cdot 10^{-6}`, and in practice nearer :math:`\sqrt{64} \cdot 2^{-24}`.
  Over the render, the double-precision films accumulate as the CUDA backend's do.
  A flush reads and clears every float element: it takes a few milliseconds for a 2048×2048 RGB film, once per 64 samples per element.
* BVH traversals have fixed-size stacks: an overflow stops the render with an error.
* Paths stop after :math:`2^{14}` steps, with a warning.
* Scenes are compiled per sensor (emitters, materials, film), but share one upload of the geometry: a render uploads the BVH once for all its sensors, or uses an upload it is given (``render_opts_t::gpu_geometry``; the Rust bindings give their scene's).
* One GPU renders; CPU threads trace only the heaviest of the re-traced paths (see `Executing paths`_).
