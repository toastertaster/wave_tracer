
Performance
=======================================

Times on this page are render times on an NVIDIA GB10 (48 SMs, compute capability 12.1; CUDA 13.0, OptiX 9.0; 20 Arm CPU cores), except those of `Metal on Apple silicon`_, of single runs, which vary by a few percent from run to run.
They exclude loading the scene, building its BVH and uploading it to the GPU.
Renders on the GPU also use the CPU, for the heaviest of the paths re-traced after running out of memory (see `Re-traced paths on the CPU`_); unless noted, times are of the command line's ``render --gpu`` (``c3ffba8``), whose CPU tracing uses all of the machine's hardware threads.
Scenes are those in ``scenes/`` and the test scenes of the Rust bindings (``tests/scenes``), rendered with the defines given (``-D name=value``); a film of resolution ``res`` is ``res`` pixels wide.


At a glance
-----------

* **Beams cost far more than rays.** ``plt_path`` with beams takes 3 to 18 times longer than in ray-tracing mode; ``plt_bdpt`` with beams, 15 to 150 times longer in the scenes below. Ray tracing renders millions of paths per second on any of these scenes.
* **A beam's cost grows with the geometry its footprint overlaps.** Finely tessellated meshes, and wide beams (of coarse films), make beam queries return many triangles and edges, all of which the path then processes.
* **The CPU traces the heaviest re-traced paths, alongside the GPU.** Paths whose beam queries need 1024 entries or more of scratch memory, a GPU thread's work of seconds each, go to the CPU's cores. This made the overflow-heavy renders 2 to 34 times faster on the GB10: ``bidir_room`` 8.8 s to 0.26 s, ``cornell-box/box.xml`` 89 s to 11 s, ``colourchecker`` 22 s to 2.2 s, ``sphere_polarization`` 9.1 s to 4.6 s. Other renders do not change.
* **OptiX matters only for plt_bdpt.** It traces ``plt_bdpt``'s connection shadow rays, 15% to 20% faster in ray-tracing mode; with beams, whose time goes elsewhere, and in ``plt_path`` renders, the difference is within run-to-run noise.
* **On Apple silicon (Metal)**, an M4's GPU renders ``plt_path`` 1.7 to 7 times faster than its CPU, and hardware ray tracing (M3 and later), which Metal's path kernels use for all their rays, renders ray-traced scenes of thousands of triangles up to twice as fast (see `Metal on Apple silicon`_).
* **64-bit kernels are 3 to 8 times slower** on the GB10, a GPU with a low 64-bit floating-point rate.
* **Small renders do not fill the GPU.** A pass has a fixed cost of tens of milliseconds for ``plt_path``, and of up to a second for ``plt_bdpt`` with beams, whose longest paths dominate a small pass.


Configurations
--------------

The same scenes in the CUDA + OptiX build, with OptiX and without it (``--no-optix``, which traces rays with the CUDA traversal, as the CUDA-only build does), with beams and in ray-tracing mode (``--ray-tracing``):

.. list-table::
   :header-rows: 1
   :widths: 30 20 10 10 10 10 10

   * - Scene
     - Defines
     - Beams, OptiX
     - Beams, no OptiX
     - Rays, OptiX
     - Rays, no OptiX
     - Beams / rays
   * - test ``box`` (plt_path)
     - res=256 spp=64
     - 437 ms
     - 447 ms
     - 90 ms
     - 97 ms
     - 4.9×
   * - test ``glass_box``
     - res=256 spp=64
     - 453 ms
     - 474 ms
     - 61 ms
     - 68 ms
     - 7.4×
   * - test ``lights_box``
     - res=256 spp=64
     - 291 ms
     - 298 ms
     - 47 ms
     - 49 ms
     - 6.2×
   * - test ``textured_box``
     - res=256 spp=64
     - 289 ms
     - 292 ms
     - 93 ms
     - 99 ms
     - 3.1×
   * - test ``rough_box``
     - res=256 spp=64
     - 341 ms
     - 354 ms
     - 65 ms
     - 70 ms
     - 5.2×
   * - test ``polarized_box`` (Stokes)
     - res=256 spp=64
     - 279 ms
     - 284 ms
     - 61 ms
     - 73 ms
     - 4.6×
   * - test ``coverage`` (forward plt_path)
     - res=256 spp=256
     - 482 ms
     - 493 ms
     - 62 ms
     - 65 ms
     - 7.8×
   * - test ``lights_coverage``
     - res=256 spp=256
     - 674 ms
     - 664 ms
     - 65 ms
     - 75 ms
     - 10×
   * - ``veach_mis``
     - res=480 spp=64
     - 747 ms
     - 751 ms
     - 146 ms
     - 141 ms
     - 5.1×
   * - ``diffraction_simple/double_slits``
     - res=512 spp=32
     - 452 ms
     - 470 ms
     - 47 ms
     - 51 ms
     - 9.6×
   * - ``sionna_etoile`` (coverage)
     - res=512 spp=64
     - 1.14 s
     - 1.14 s
     - 63 ms
     - 66 ms
     - 18×
   * - ``cornell-box/sphere_polarization`` (plt_bdpt, Stokes)
     - res=128 spp=8
     - 4.64 s
     - 4.70 s
     - 31 ms
     - 36 ms
     - 150×
   * - ``bidir_room`` (plt_bdpt)
     - res=96 spp=4
     - 259 ms
     - 252 ms
     - 17 ms
     - 21 ms
     - 15×

In ``plt_path`` renders, OptiX makes no difference beyond run-to-run noise, as the path kernels use the CUDA traversals either way.
In ``plt_bdpt`` renders, OptiX traces the connections' shadow rays in batches (see :doc:`design`): ray-traced renders take 15% to 20% less time with it; renders with beams spend their time on beam queries and on re-traced paths, and do not change.


Re-traced paths on the CPU
--------------------------

A path whose beam query finds more triangles or edges than its scratch memory holds stops, and is traced again after the pass with the room it needs (see :doc:`design`).
Paths that need 1024 entries or more are traced by the CPU's threads (one per hardware thread), with the kernels' own code compiled as host C++, while the GPU re-traces the others.
Such paths overlap so much geometry that a GPU thread takes seconds for one, and they used to leave the GPU waiting on a handful of them at the end of each pass; a CPU core traces them many times faster.
They are the same paths, with the same random numbers: renders match those whose re-traces all ran on the GPU up to rounding (``sphere_polarization`` at 256×256 pixels and 4 samples per pixel: a relative L1 difference of :math:`6 \cdot 10^{-5}`).

The GB10, re-tracing on the GPU only (``93344c2``) and with the heaviest re-traces on its CPU (``c3ffba8``), with beams:

.. list-table::
   :header-rows: 1
   :widths: 40 22 13 13 12

   * - Scene
     - Defines
     - GPU only
     - GPU + CPU
     - Speed-up
   * - ``bidir_room`` (plt_bdpt)
     - res=96 spp=4
     - 8.80 s
     - 259 ms
     - 34×
   * - ``bidir_room``
     - res=192 spp=4
     - 15.1 s
     - 661 ms
     - 23×
   * - ``colourchecker`` (backward plt_path, 909 K triangles)
     - res=192 spp=4
     - 22.3 s
     - 2.24 s
     - 10×
   * - ``cornell-box/box.xml`` (plt_bdpt)
     - res=256 spp=64
     - 88.6 s
     - 11.2 s
     - 7.9×
   * - ``cornell-box/sphere_polarization`` (plt_bdpt, Stokes)
     - res=64 spp=4
     - 2.89 s
     - 828 ms
     - 3.5×
   * - ``cornell-box/sphere_polarization``
     - res=128 spp=4
     - 6.28 s
     - 2.53 s
     - 2.5×
   * - ``cornell-box/sphere_polarization``
     - res=128 spp=8
     - 9.13 s
     - 4.64 s
     - 2.0×
   * - ``cornell-box/sphere_polarization``
     - res=256 spp=4
     - 16.2 s
     - 9.46 s
     - 1.7×
   * - ``cornell-box/sphere_polarization``
     - res=192 spp=16
     - 31.2 s
     - 20.7 s
     - 1.5×

The gain is largest where a few paths dominated: ``bidir_room``'s render was almost all re-traces.
``sphere_polarization`` keeps more of its time on the GPU, in its first pass and in the re-traces of moderate size, and gains less as its films grow.
The other scenes of `Configurations`_ re-trace few paths, if any, and render as fast as before (within run-to-run noise; ``sionna_etoile``, 11% faster).


Metal on Apple silicon
----------------------

The Metal backend on an Apple M4 (10-core GPU, which has ray-tracing hardware; 4 performance and 6 efficiency CPU cores, 32 GB; macOS 27.2, Metal toolchain 32023): with hardware ray tracing (the default), without it (``--no-hw-rt``), against the same build rendering on the M4's CPU (all 10 cores), and the GB10 times above (CUDA + OptiX).
Times are of the second of two runs, so that the GPU's compiled pipelines are cached (see below).

With beams:

.. list-table::
   :header-rows: 1
   :widths: 30 16 11 13 10 10

   * - Scene
     - Defines
     - Metal
     - Metal, ``--no-hw-rt``
     - M4 CPU
     - GB10
   * - test ``box`` (plt_path)
     - res=256 spp=64
     - 1.0 s
     - 964 ms
     - 3.5 s
     - 437 ms
   * - test ``glass_box``
     - res=256 spp=64
     - 1.3 s
     - 1.6 s
     - 4.2 s
     - 453 ms
   * - test ``lights_box``
     - res=256 spp=64
     - 794 ms
     - 773 ms
     - 3.2 s
     - 291 ms
   * - test ``textured_box``
     - res=256 spp=64
     - 832 ms
     - 824 ms
     - 3.1 s
     - 289 ms
   * - test ``rough_box``
     - res=256 spp=64
     - 1.0 s
     - 1.1 s
     - 4.0 s
     - 341 ms
   * - test ``polarized_box`` (Stokes)
     - res=256 spp=64
     - 703 ms
     - 611 ms
     - 2.0 s
     - 279 ms
   * - test ``coverage`` (forward plt_path)
     - res=256 spp=256
     - 1.7 s
     - 1.7 s
     - 4.1 s
     - 482 ms
   * - test ``lights_coverage``
     - res=256 spp=256
     - 3.0 s
     - 3.0 s
     - 9.3 s
     - 674 ms
   * - ``diffraction_simple/double_slits``
     - res=512 spp=32
     - 1.3 s
     - 1.2 s
     - 2.5 s
     - 452 ms
   * - ``veach_mis``
     - res=480 spp=64
     - 1.9 s
     - 1.9 s
     - 5.6 s
     - 747 ms
   * - ``sionna_etoile`` (coverage)
     - res=512 spp=64
     - 4.5 s
     - 5.3 s
     - 7.6 s
     - 1.14 s
   * - ``cornell-box/sphere_polarization`` (plt_bdpt, Stokes)
     - res=128 spp=8
     - 7.8 s
     - 8.2 s
     - 7.3 s
     - 4.64 s
   * - ``bidir_room`` (plt_bdpt)
     - res=96 spp=4
     - 360 ms
     - 361 ms
     - 339 ms
     - 259 ms

In ray-tracing mode (``--ray-tracing``), with the scenes' triangle counts:

.. list-table::
   :header-rows: 1
   :widths: 28 14 10 10 13 10 10

   * - Scene
     - Defines
     - Triangles
     - Metal
     - Metal, ``--no-hw-rt``
     - M4 CPU
     - GB10
   * - test ``box`` (plt_path)
     - res=256 spp=64
     - 1290
     - 159 ms
     - 323 ms
     - 1.2 s
     - 90 ms
   * - test ``glass_box``
     - res=256 spp=64
     - 1304
     - 167 ms
     - 226 ms
     - 1.0 s
     - 61 ms
   * - test ``lights_box``
     - res=256 spp=64
     - 14
     - 131 ms
     - 116 ms
     - 784 ms
     - 47 ms
   * - test ``textured_box``
     - res=256 spp=64
     - 38
     - 234 ms
     - 256 ms
     - 938 ms
     - 93 ms
   * - test ``rough_box``
     - res=256 spp=64
     - 38
     - 191 ms
     - 217 ms
     - 1.1 s
     - 65 ms
   * - test ``polarized_box`` (Stokes)
     - res=256 spp=64
     - 10
     - 125 ms
     - 114 ms
     - 547 ms
     - 61 ms
   * - test ``coverage`` (forward plt_path)
     - res=256 spp=256
     - 38
     - 171 ms
     - 166 ms
     - 820 ms
     - 62 ms
   * - test ``lights_coverage``
     - res=256 spp=256
     - 40
     - 194 ms
     - 192 ms
     - 993 ms
     - 65 ms
   * - ``diffraction_simple/double_slits``
     - res=512 spp=32
     - 10
     - 122 ms
     - 103 ms
     - 554 ms
     - 47 ms
   * - ``veach_mis``
     - res=480 spp=64
     - 3892
     - 307 ms
     - 521 ms
     - 2.3 s
     - 146 ms
   * - ``sionna_etoile`` (coverage)
     - res=512 spp=64
     - 13058
     - 148 ms
     - 225 ms
     - 803 ms
     - 63 ms
   * - ``cornell-box/sphere_polarization`` (plt_bdpt, Stokes)
     - res=128 spp=8
     - 20502
     - 41 ms
     - 63 ms
     - 116 ms
     - 31 ms
   * - ``bidir_room`` (plt_bdpt)
     - res=96 spp=4
     - 246244
     - 21 ms
     - 23 ms
     - 24 ms
     - 17 ms

* **plt_path renders 1.7 to 4 times faster with beams, and 3 to 7 times faster with rays, on the M4's GPU than on its CPU**, and 2 to 4.5 times slower than on the GB10, a GPU with several times the M4's 32-bit floating-point throughput.
* **Hardware ray tracing pays off with geometry**: ray-traced renders of the scenes of a thousand triangles or more take 25% to 50% less time with it (``box``: 323 ms to 159 ms; ``veach_mis``: 521 ms to 307 ms; ``sionna_etoile``: 225 ms to 148 ms; ``sphere_polarization``: 63 ms to 41 ms; ``bidir_room``'s 21 ms, mostly a pass's fixed cost, only 9%), and ``glass_box`` and ``sionna_etoile`` with beams 15% to 20% less. On the test scenes of a few dozen triangles, whose BVH is a leaf or two, the ported traversal takes up to 16% less time; and beams, whose cone queries always traverse the BVH, gain less.
* **plt_bdpt with beams is no faster on the GPU than on the CPU** in these two scenes: most of their time goes to the paths that overflow their scratch memory (see :doc:`design`), whose beams overlap thousands of triangles and edges. Before such paths were traced by the CPU, the GPU took 14.7 s for ``sphere_polarization`` (its first pass about 2 s; then thousands of re-traces, the last few hundred each on a single GPU thread for seconds) and 10.9 s for ``bidir_room``; the GB10 took 9.1 s and 8.8 s, and takes 4.64 s and 259 ms since (see `Re-traced paths on the CPU`_).

What was measured, and decided, while tuning the Metal backend:

* **Command buffers.** A wavefront step (path starts, beam traversals, interactions) is one command buffer, which the host waits for, as CUDA's host waits for each step's counters. The GPU is busy for 96% of a render's time (899 ms of GPU time in 66 command buffers, of a 938 ms ``box`` render); indirect dispatch, or Metal 4's command submission, could save at most the rest, so the backend uses neither.
* **Per-thread memory.** Instruments reports about 1.5 KiB of stack and spills per thread in the wavefront's kernels, mostly the BVH traversal stacks (128 entries for cones). Neither reading the scene's parameters in place (in Metal's ``constant`` address space, rather than as a copy), nor loading path states as 16-byte words, changed render times: Metal's compiler handles both.
* **Threadgroups and persistent threads.** Threadgroups of 32 to 256 threads, and 512 to 4096 persistent threads per GPU core, render within 2% of each other; the backend uses 128 and 2048.
* **Free-space diffraction's rejection sampling** shares its tries among a SIMD-group's lanes, as among a CUDA warp's: plt_bdpt's first pass of ``sphere_polarization`` traces 6% more paths per second than with each lane sampling alone.
* **plt_bdpt's memory for paths in flight** is a quarter of the GPU's recommended working set, up to 2 GiB, rather than CUDA's 512 MiB: re-traced paths need much more memory each, and 512 MiB held too few of them to fill the GPU (``sphere_polarization``: 24.6 s with 512 MiB, 16.6 s with 2 GiB; 4 GiB is no faster).
* **The first render after building** includes compiling the kernels' pipelines for the GPU, a second or two (and again the first time with ``--no-hw-rt``, whose pipelines differ); macOS caches them for later runs.

64-bit kernels
--------------

The CUDA build with 32-bit kernels against the same build with ``GPU_DBL_PRECISION=ON`` (measured through the Rust bindings, before the CPU re-traced paths; these ``plt_path`` renders re-trace few paths):

.. list-table::
   :header-rows: 1
   :widths: 28 22 10 10 10 10 10

   * - Scene
     - Defines
     - Beams, 32-bit
     - Beams, 64-bit
     - Rays, 32-bit
     - Rays, 64-bit
     - Slowdown (beams, rays)
   * - test ``box``
     - res=64 spp=64
     - 391 ms
     - 3.05 s
     - 19 ms
     - 90 ms
     - 7.8×, 4.8×
   * - test ``lights_box``
     - res=64 spp=64
     - 59 ms
     - 263 ms
     - 16 ms
     - 45 ms
     - 4.4×, 2.8×
   * - test ``coverage``
     - res=64 spp=256
     - 75 ms
     - 391 ms
     - 17 ms
     - 48 ms
     - 5.2×, 2.8×
   * - ``veach_mis``
     - res=120 spp=16
     - 193 ms
     - 1.29 s
     - 17 ms
     - 49 ms
     - 6.7×, 3.0×

The GB10 computes 64-bit floating point at a small fraction of its 32-bit rate; beams, whose geometry is heavier in floating-point arithmetic, slow down more than rays.
GPUs with a high 64-bit rate (data-centre GPUs) would slow down less.


Scaling with the work per pass
------------------------------

One 256×256 film, with more samples per pixel, with beams (CUDA + OptiX build; measured through the Rust bindings, before the CPU re-traced paths, which these renders do not depend on):

.. list-table::
   :header-rows: 1
   :widths: 16 16 17 17 17 17

   * - spp
     - Paths
     - test ``box``
     - paths/s
     - test ``lights_box``
     - paths/s
   * - 1
     - 65 K
     - 70 ms
     - 0.9 M
     - 50 ms
     - 1.3 M
   * - 4
     - 262 K
     - 81 ms
     - 3.2 M
     - 60 ms
     - 4.4 M
   * - 16
     - 1.0 M
     - 157 ms
     - 6.7 M
     - 109 ms
     - 9.6 M
   * - 64
     - 4.2 M
     - 437 ms
     - 9.6 M
     - 295 ms
     - 14 M
   * - 256
     - 16.8 M
     - 1.50 s
     - 11 M
     - 1.07 s
     - 16 M

A pass has a fixed cost of a few tens of milliseconds.
The GPU reaches its throughput with a few million paths, i.e. once a pass is full (:math:`2^{22}` paths); beyond that, time grows linearly with the samples.

``plt_bdpt`` with beams needs more paths to reach its throughput: its paths are long and their costs vary widely, and the slowest paths of a launch dominate a small render.
``cornell-box/sphere_polarization`` at 4 samples per pixel takes 828 ms at 64×64 (16 K paths), 2.53 s at 128×128 (65 K paths) and 9.46 s at 256×256 (262 K paths), about 36 µs per path at the largest.


What a beam costs
-----------------

A ray query returns the closest hit; a beam query returns every triangle the beam's footprint overlaps, and a path processes them all (and, with free-space diffraction, the edges near its beam).
Beams therefore slow down with:

* **Tessellation.** ``scenes/colourchecker`` (909 K triangles) renders 192×144 pixels at 4 samples per pixel in 2.24 s with beams, and in 16 ms in ray-tracing mode: 140 times faster.
  Paths whose queries find more than 64 triangles or edges are also traced again, with the memory they need, the heaviest by the CPU (see `Re-traced paths on the CPU`_): before the CPU took those, this render took 22.3 s.
* **Wide beams.** A sensor's beams spread with the size of its elements, so coarse films trace wide beams. The test ``box`` scene renders 262 K paths in 81 ms at 256×256 with 4 samples per pixel, but in 391 ms at 64×64 with 64 samples per pixel.
* **plt_bdpt's connections**, between every vertex of the sensor subpath and every vertex of the emitter subpath, each evaluating its beams' interactions and a shadow ray.

In ray-tracing mode, none of this applies, and renders take milliseconds to a few seconds.


Convergence
-----------

Speed per path is not speed to a clean image.
Forward ``plt_path`` renders of large coverage maps with narrow beams, such as ``scenes/sionna_munich`` (beam aperture 0.01°), need many samples: at 960×633 pixels and 128 samples per pixel (15.6 s), only about 5% of its pixels receive any contribution, and the far streets fill in only with thousands of samples per pixel.
``plt_bdpt`` with beams converges slowly too: ``scenes/cornell-box/box.xml`` takes 11.2 s at 256×256 pixels and 64 samples per pixel, and is still noisy (see :doc:`gallery`).


Tips
----

* Preview with ``--ray-tracing``, which renders most scenes in well under a second, before rendering with beams.
* Render enough paths per pass to fill the GPU: films of at least 256×256 elements, or enough samples per element; with ``plt_bdpt``, prefer larger films.
* Keep meshes as coarse as the simulation allows when rendering with beams.
* Build with OptiX for ``plt_bdpt``; it costs nothing otherwise, and ``--no-optix`` turns it off at run time.
* On Apple silicon, hardware ray tracing (M3 and later) is on by default; ``--no-hw-rt`` may be faster for scenes of a few dozen triangles.
* Use ``GPU_DBL_PRECISION`` to investigate precision, not for production renders.
* Interrupts and previews wait for a pass to end; with ``plt_bdpt`` and beams, a pass can take a minute or more.
