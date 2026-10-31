
Performance
=======================================

All times on this page are render times on an NVIDIA GB10 (48 SMs, compute capability 12.1; CUDA 13.0, OptiX 9.0), of single runs, which vary by a few percent from run to run.
They exclude loading the scene, building its BVH and uploading it to the GPU.
Scenes are those in ``scenes/`` and the test scenes of the Rust bindings (``tests/scenes``), rendered with the defines given (``-D name=value``); a film of resolution ``res`` is ``res`` pixels wide.


At a glance
-----------

* **Beams cost far more than rays.** ``plt_path`` with beams takes 3 to 18 times longer than in ray-tracing mode; ``plt_bdpt`` with beams, hundreds of times longer (380× and 540× below). Ray tracing renders millions of paths per second on any of these scenes.
* **A beam's cost grows with the geometry its footprint overlaps.** Finely tessellated meshes, and wide beams (of coarse films), make beam queries return many triangles and edges, all of which the path then processes.
* **OptiX matters only for plt_bdpt.** It traces ``plt_bdpt``'s connection shadow rays (0 to 20% faster); ``plt_path``, with beams or rays, renders at the same speed with or without it.
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
* Use ``GPU_DBL_PRECISION`` to investigate precision, not for production renders.
* Interrupts and previews wait for a pass to end; with ``plt_bdpt`` and beams, a pass can take a minute or more.
