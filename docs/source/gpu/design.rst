
Design
=======================================

The GPU renderer ports **wave_tracer**'s light transport, rather than approximating it: its kernels trace the CPU's beams (elliptic cones), with the same BVH traversals, interaction models, free-space diffraction and sampling decisions.
What differs is how the work is organized, so that thousands of paths run at once, and the random numbers the paths draw.
The CPU still loads the scene, builds its BVH, and develops and writes the rendered films.


Code layout
-----------

``include/wt/gpu/kernel/``
   The kernels: math, ray/triangle and cone/triangle intersections, the ``bvh8w`` traversals (``bvh.hpp``), beams (``beam.hpp``), polarimetry, free-space diffraction (``fsd.hpp``, ``fraunhofer.hpp``), textures, and the integrators (``plt_path.hpp``, ``plt_bdpt.hpp``).
   They are written in a subset of C++17, without wave_tracer's unit types or the standard library (see ``config.hpp``), so that the same headers compile with ``nvcc`` and as host C++, and can be ported to other GPU languages (such as Metal) later.
   ``real`` is ``float``, or ``double`` with ``GPU_DBL_PRECISION``.

``include/wt/gpu/gpu.hpp``, ``include/wt/gpu/renderer.hpp``
   The API the rest of wave_tracer uses: ``gpu::device_t``, ``gpu::geometry_t`` (batched acceleration-structure queries) and ``gpu::renderer_t`` (renders a sensor).
   ``src/gpu/renderer.cpp`` compiles a scene into the kernels' plain structs (``kernel/scene.hpp``), refusing what the kernels do not implement.

``include/wt/gpu/backend.hpp``, ``src/gpu/cuda/``, ``src/gpu/optix/``
   The interface a backend implements, and the CUDA backend with its OptiX ray tracing.

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
The GPU accumulates the paths' splats into films in double precision (with atomic additions), as the CPU's integrators splat them: backward ``plt_path`` into image blocks with reconstruction weights, forward ``plt_path`` directly, and ``plt_bdpt`` both, the latter into a separate light image.
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

**Free-space diffraction's rejection sampling** shares its tries among a warp's lanes, so that a lane whose tries are rejected does not hold back the others.

**Scratch memory.**
   A beam query returns all the triangles and edges within the beam's footprint, into a per-path buffer with room for 64.
   A path whose query finds more is recorded and traced again, from its start, after the pass, with four times the room (up to :math:`2^{20}` entries); it skips the splats its earlier attempt made (``plt_bdpt`` paths that overflow splat nothing the first time).

**plt_bdpt connections.**
   ``plt_bdpt`` connects the vertices of its sensor and emitter subpaths, each connection needing a shadow ray.
   With OptiX, a launch defers these shadow rays, with the splats they would make, into a batch of up to :math:`2^{22}`; OptiX traces the batch on the GPU's ray-tracing cores, and a kernel then makes the splats of the unoccluded connections.
   Without OptiX, paths trace their shadow rays themselves, with the CUDA traversal.

Everything else (cone and ray traversals of the paths, edge queries of free-space diffraction) uses the CUDA port of ``bvh8w``: the same node layout and visiting order as the CPU's, with leaves encoded in their parents' child pointers.
OptiX cannot replace these traversals, as beams are cones, not rays, and OptiX rays cannot be traced from within the long-running path kernels.


Precision and limits
--------------------

* Kernels compute in 32-bit floating point by default, as the CPU's default build does; ``GPU_DBL_PRECISION`` switches them to 64 bits. Films accumulate in 64 bits either way.
* BVH traversals have fixed-size stacks: an overflow stops the render with an error.
* Paths stop after :math:`2^{14}` steps, with a warning.
* Scenes are compiled per sensor: each sensor of a scene uploads its own copy of the geometry.
* One GPU renders; CPU threads do not take part in rendering.
