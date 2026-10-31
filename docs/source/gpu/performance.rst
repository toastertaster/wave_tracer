
Performance
=======================================

Times on this page are render times on an NVIDIA GB10 (48 SMs, compute capability 12.1; CUDA 13.0, OptiX 9.0), except those of `Metal on Apple silicon`_, of single runs, which vary by a few percent from run to run.
They exclude loading the scene, building its BVH and uploading it to the GPU.
Scenes are those in ``scenes/`` and the test scenes of the Rust bindings (``tests/scenes``), rendered with the defines given (``-D name=value``); a film of resolution ``res`` is ``res`` pixels wide.


At a glance
-----------

* **Beams cost far more than rays.** ``plt_path`` with beams takes 3 to 18 times longer than in ray-tracing mode; ``plt_bdpt`` with beams, hundreds of times longer (380× and 540× below). Ray tracing renders millions of paths per second on any of these scenes.
* **A beam's cost grows with the geometry its footprint overlaps.** Finely tessellated meshes, and wide beams (of coarse films), make beam queries return many triangles and edges, all of which the path then processes.
* **OptiX matters only for plt_bdpt.** It traces ``plt_bdpt``'s connection shadow rays (0 to 20% faster); ``plt_path``, with beams or rays, renders at the same speed with or without it.
* **On Apple silicon (Metal)**, an M4's GPU renders ``plt_path`` 1.7 to 7 times faster than its CPU, and hardware ray tracing (M3 and later), which Metal's path kernels use for all their rays, renders ray-traced scenes of thousands of triangles up to twice as fast (see `Metal on Apple silicon`_).
* **64-bit kernels are 3 to 8 times slower** on the GB10, a GPU with a low 64-bit floating-point rate.
* **Small renders do not fill the GPU.** A pass has a fixed cost of tens of milliseconds for ``plt_path``, and of seconds for ``plt_bdpt`` with beams, whose longest paths dominate a small pass.


Configurations
--------------

The same scenes in the CUDA and the CUDA + OptiX builds, with beams and in ray-tracing mode (``--ray-tracing``):

.. list-table::
   :header-rows: 1
   :widths: 30 20 10 10 10 10 10

   * - Scene
     - Defines
     - Beams, OptiX
     - Beams, CUDA
     - Rays, OptiX
     - Rays, CUDA
     - Beams / rays
   * - test ``box`` (plt_path)
     - res=256 spp=64
     - 451 ms
     - 470 ms
     - 89 ms
     - 92 ms
     - 5.1×
   * - test ``glass_box``
     - res=256 spp=64
     - 467 ms
     - 435 ms
     - 64 ms
     - 63 ms
     - 7.4×
   * - test ``lights_box``
     - res=256 spp=64
     - 291 ms
     - 289 ms
     - 44 ms
     - 46 ms
     - 6.6×
   * - test ``textured_box``
     - res=256 spp=64
     - 291 ms
     - 295 ms
     - 92 ms
     - 96 ms
     - 3.1×
   * - test ``rough_box``
     - res=256 spp=64
     - 346 ms
     - 359 ms
     - 66 ms
     - 72 ms
     - 5.2×
   * - test ``polarized_box`` (Stokes)
     - res=256 spp=64
     - 276 ms
     - 300 ms
     - 64 ms
     - 62 ms
     - 4.3×
   * - test ``coverage`` (forward plt_path)
     - res=256 spp=256
     - 481 ms
     - 488 ms
     - 64 ms
     - 62 ms
     - 7.5×
   * - test ``lights_coverage``
     - res=256 spp=256
     - 661 ms
     - 673 ms
     - 71 ms
     - 66 ms
     - 9.3×
   * - ``veach_mis``
     - res=480 spp=64
     - 779 ms
     - 798 ms
     - 140 ms
     - 142 ms
     - 5.6×
   * - ``diffraction_simple/double_slits``
     - res=512 spp=32
     - 464 ms
     - 467 ms
     - 45 ms
     - 55 ms
     - 10×
   * - ``sionna_etoile`` (coverage)
     - res=512 spp=64
     - 1.30 s
     - 1.33 s
     - 71 ms
     - 59 ms
     - 18×
   * - ``cornell-box/sphere_polarization`` (plt_bdpt, Stokes)
     - res=128 spp=8
     - 12.6 s
     - 12.6 s
     - 33 ms
     - 41 ms
     - 380×
   * - ``bidir_room`` (plt_bdpt)
     - res=96 spp=4
     - 8.45 s
     - 10.3 s
     - 16 ms
     - 18 ms
     - 540×

In ``plt_path`` renders the two builds differ by no more than run-to-run noise, as the path kernels use the CUDA traversals in both.
In ``plt_bdpt`` renders, OptiX traces the connections' shadow rays in batches (see :doc:`design`): ``bidir_room`` renders 18% faster with it, while ``sphere_polarization``, whose cost lies elsewhere, does not change.


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
     - 451 ms
   * - test ``glass_box``
     - res=256 spp=64
     - 1.3 s
     - 1.6 s
     - 4.2 s
     - 467 ms
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
     - 291 ms
   * - test ``rough_box``
     - res=256 spp=64
     - 1.0 s
     - 1.1 s
     - 4.0 s
     - 346 ms
   * - test ``polarized_box`` (Stokes)
     - res=256 spp=64
     - 703 ms
     - 611 ms
     - 2.0 s
     - 276 ms
   * - test ``coverage`` (forward plt_path)
     - res=256 spp=256
     - 1.7 s
     - 1.7 s
     - 4.1 s
     - 481 ms
   * - test ``lights_coverage``
     - res=256 spp=256
     - 3.0 s
     - 3.0 s
     - 9.3 s
     - 661 ms
   * - ``diffraction_simple/double_slits``
     - res=512 spp=32
     - 1.3 s
     - 1.2 s
     - 2.5 s
     - 464 ms
   * - ``veach_mis``
     - res=480 spp=64
     - 1.9 s
     - 1.9 s
     - 5.6 s
     - 779 ms
   * - ``sionna_etoile`` (coverage)
     - res=512 spp=64
     - 4.5 s
     - 5.3 s
     - 7.6 s
     - 1.30 s
   * - ``cornell-box/sphere_polarization`` (plt_bdpt, Stokes)
     - res=128 spp=8
     - 14.4 s
     - 15.2 s
     - 7.3 s
     - 12.6 s
   * - ``bidir_room`` (plt_bdpt)
     - res=96 spp=4
     - 11.0 s
     - 10.6 s
     - 339 ms
     - 8.45 s

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
     - 89 ms
   * - test ``glass_box``
     - res=256 spp=64
     - 1304
     - 167 ms
     - 226 ms
     - 1.0 s
     - 64 ms
   * - test ``lights_box``
     - res=256 spp=64
     - 14
     - 131 ms
     - 116 ms
     - 784 ms
     - 44 ms
   * - test ``textured_box``
     - res=256 spp=64
     - 38
     - 234 ms
     - 256 ms
     - 938 ms
     - 92 ms
   * - test ``rough_box``
     - res=256 spp=64
     - 38
     - 191 ms
     - 217 ms
     - 1.1 s
     - 66 ms
   * - test ``polarized_box`` (Stokes)
     - res=256 spp=64
     - 10
     - 125 ms
     - 114 ms
     - 547 ms
     - 64 ms
   * - test ``coverage`` (forward plt_path)
     - res=256 spp=256
     - 38
     - 171 ms
     - 166 ms
     - 820 ms
     - 64 ms
   * - test ``lights_coverage``
     - res=256 spp=256
     - 40
     - 194 ms
     - 192 ms
     - 993 ms
     - 71 ms
   * - ``diffraction_simple/double_slits``
     - res=512 spp=32
     - 10
     - 122 ms
     - 103 ms
     - 554 ms
     - 45 ms
   * - ``veach_mis``
     - res=480 spp=64
     - 3892
     - 307 ms
     - 521 ms
     - 2.3 s
     - 140 ms
   * - ``sionna_etoile`` (coverage)
     - res=512 spp=64
     - 13058
     - 148 ms
     - 225 ms
     - 803 ms
     - 71 ms
   * - ``cornell-box/sphere_polarization`` (plt_bdpt, Stokes)
     - res=128 spp=8
     - 20502
     - 41 ms
     - 63 ms
     - 116 ms
     - 33 ms
   * - ``bidir_room`` (plt_bdpt)
     - res=96 spp=4
     - 246244
     - 21 ms
     - 23 ms
     - 24 ms
     - 16 ms

* **plt_path renders 1.7 to 4 times faster with beams, and 3 to 7 times faster with rays, on the M4's GPU than on its CPU**, and 2 to 4.5 times slower than on the GB10, a GPU with several times the M4's 32-bit floating-point throughput.
* **Hardware ray tracing pays off with geometry**: ray-traced renders of the scenes of a thousand triangles or more take 25% to 50% less time with it (``box``: 323 ms to 159 ms; ``veach_mis``: 521 ms to 307 ms; ``sionna_etoile``: 225 ms to 148 ms; ``sphere_polarization``: 63 ms to 41 ms; ``bidir_room``'s 21 ms, mostly a pass's fixed cost, only 9%), and ``glass_box`` and ``sionna_etoile`` with beams 15% to 20% less. On the test scenes of a few dozen triangles, whose BVH is a leaf or two, the ported traversal takes up to 16% less time; and beams, whose cone queries always traverse the BVH, gain less.
* **plt_bdpt with beams is slower on the GPU than on the CPU**: ``sphere_polarization`` takes twice as long, and ``bidir_room`` thirty times. In ``sphere_polarization``, the first pass takes about 2 s; the rest is spent re-tracing the paths that overflow their scratch memory (see :doc:`design`), whose beams overlap thousands of triangles and edges: a few hundred such paths, each traced by a single GPU thread, far slower than a CPU core. The GB10 spends most of its 12.6 s on them too.

What was measured, and decided, while tuning the Metal backend:

* **Command buffers.** A wavefront step (path starts, beam traversals, interactions) is one command buffer, which the host waits for, as CUDA's host waits for each step's counters. The GPU is busy for 96% of a render's time (899 ms of GPU time in 66 command buffers, of a 938 ms ``box`` render); indirect dispatch, or Metal 4's command submission, could save at most the rest, so the backend uses neither.
* **Per-thread memory.** Instruments reports about 1.5 KiB of stack and spills per thread in the wavefront's kernels, mostly the BVH traversal stacks (128 entries for cones). Neither reading the scene's parameters in place (in Metal's ``constant`` address space, rather than as a copy), nor loading path states as 16-byte words, changed render times: Metal's compiler handles both.
* **Threadgroups and persistent threads.** Threadgroups of 32 to 256 threads, and 512 to 4096 persistent threads per GPU core, render within 2% of each other; the backend uses 128 and 2048.
* **Free-space diffraction's rejection sampling** shares its tries among a SIMD-group's lanes, as among a CUDA warp's: plt_bdpt's first pass of ``sphere_polarization`` traces 6% more paths per second than with each lane sampling alone.
* **plt_bdpt's memory for paths in flight** is a quarter of the GPU's recommended working set, up to 2 GiB, rather than CUDA's 512 MiB: re-traced paths need much more memory each, and 512 MiB held too few of them to fill the GPU (``sphere_polarization``: 24.6 s with 512 MiB, 16.6 s with 2 GiB; 4 GiB is no faster).
* **The first render after building** includes compiling the kernels' pipelines for the GPU, a second or two (and again the first time with ``--no-hw-rt``, whose pipelines differ); macOS caches them for later runs.

64-bit kernels
--------------

The CUDA build with 32-bit kernels against the same build with ``GPU_DBL_PRECISION=ON``:

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

One 256×256 film, with more samples per pixel, with beams (CUDA + OptiX build):

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

``plt_bdpt`` with beams needs many more paths to reach its throughput: its paths are long and their costs vary widely, and the slowest paths of a launch dominate a small render.
``cornell-box/sphere_polarization`` at 4 samples per pixel takes 3.88 s at 64×64 (16 K paths), 7.60 s at 128×128 (65 K paths) and 20.8 s at 256×256 (262 K paths).


What a beam costs
-----------------

A ray query returns the closest hit; a beam query returns every triangle the beam's footprint overlaps, and a path processes them all (and, with free-space diffraction, the edges near its beam).
Beams therefore slow down with:

* **Tessellation.** ``scenes/colourchecker`` (909 K triangles) renders 192×144 pixels at 4 samples per pixel in 22.7 s with beams, and in 16 ms in ray-tracing mode: 1400 times faster.
  Paths whose queries find more than 64 triangles or edges are also traced a second time, with more memory (see :doc:`design`).
* **Wide beams.** A sensor's beams spread with the size of its elements, so coarse films trace wide beams. The test ``box`` scene renders 262 K paths in 81 ms at 256×256 with 4 samples per pixel, but in 391 ms at 64×64 with 64 samples per pixel.
* **plt_bdpt's connections**, between every vertex of the sensor subpath and every vertex of the emitter subpath, each evaluating its beams' interactions and a shadow ray.

In ray-tracing mode, none of this applies, and renders take milliseconds to a few seconds.


Convergence
-----------

Speed per path is not speed to a clean image.
Forward ``plt_path`` renders of large coverage maps with narrow beams, such as ``scenes/sionna_munich`` (beam aperture 0.01°), need many samples: at 960×633 pixels and 128 samples per pixel (18.8 s), only about 5% of its pixels receive any contribution, and the far streets fill in only with thousands of samples per pixel.
``plt_bdpt`` with beams converges slowly too: ``scenes/cornell-box/box.xml`` takes 77 s at 256×256 pixels and 64 samples per pixel, and is still noisy (see :doc:`gallery`).


Tips
----

* Preview with ``--ray-tracing``, which renders most scenes in well under a second, before rendering with beams.
* Render enough paths per pass to fill the GPU: films of at least 256×256 elements, or enough samples per element; with ``plt_bdpt``, prefer larger films.
* Keep meshes as coarse as the simulation allows when rendering with beams.
* Build with OptiX for ``plt_bdpt``; it costs nothing otherwise, and ``--no-optix`` turns it off at run time.
* On Apple silicon, hardware ray tracing (M3 and later) is on by default; ``--no-hw-rt`` may be faster for scenes of a few dozen triangles.
* Use ``GPU_DBL_PRECISION`` to investigate precision, not for production renders.
* Interrupts and previews wait for a pass to end; with ``plt_bdpt`` and beams, a pass can take a minute or more.
