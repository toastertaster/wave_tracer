Realistic remote sensing
=======================================

The fork's ``remote-sensing`` branch extends **wave_tracer** to render realistic Earth-observation imagery: a satellite camera hundreds of kilometres above a terrain scene, lit by the sun through an atmosphere, recorded by a modelled detector.
It is developed for simulating several kinds of cameras (frame, and later TDI/pushbroom, event and Geiger-mode APD cameras), from the visible and near infrared first, and later the shortwave, longwave and thermal infrared.
Instrument analysis (stray light, polarization sensitivity, tolerancing) is left to optical design software; the telescope enters as a point-spread function.

These features are unlikely to be merged upstream, so they are compiled only with the CMake option ``REALISTIC_REMOTE_SENSING``:

.. code-block:: sh

   cmake -B build -DCMAKE_BUILD_TYPE=Release -DREALISTIC_REMOTE_SENSING=ON
   cmake --build build
   ./build/wave_tracer version     # prints "with realistic remote sensing"

The option combines with the GPU backends (``-DGPU_CUDA=ON`` or ``-DGPU_METAL=ON``).
Without it, the build is the same as ``main-gpu``'s.

.. toctree::
   :maxdepth: 1

   scene_elements
   tools
   data
   gallery


What is gated
-------------

* Code that only remote sensing needs lives in ``include/wt/remote_sensing/`` and ``src/remote_sensing/``, and is compiled only with the option.
  Hooks it needs elsewhere (scene loaders, integrators, GPU kernels) are wrapped in ``#if defined(REALISTIC_REMOTE_SENSING)``.
* General improvements that remote sensing happens to need (e.g. time-varying scenes, environment emitters, bug fixes) are not gated: they land on ``gpu-experiment``/``main-gpu`` and reach ``remote-sensing`` by merging.

``cmake/wt_remote_sensing.cmake`` defines the option and ``wt_add_remote_sensing(<target>)``, which compiles ``src/remote_sensing/`` into the target and defines ``REALISTIC_REMOTE_SENSING`` on it.
``cmake/wt_gpu.cmake`` passes the definition on to the CUDA and Metal backends and their kernels.
Builds that embed **wave_tracer** (e.g. the Rust bindings) include the module before ``cmake/wt_gpu.cmake``, and exclude ``src/remote_sensing/`` if they glob the sources.


Plan
----

The first goal is stage one of the simulation: the radiance that reaches the telescope's entrance pupil, rendered by **wave_tracer**, with the telescope's blur applied afterwards from point-spread functions.

0. *Done.* Scaffolding: the option, its propagation to the GPU backends and the Rust bindings, and the data survey (:doc:`data`).
1. *Done.* Reference site: a fetch script and a scene builder for a terrain tile (elevation model and orthophoto) at Railroad Valley Playa, Nevada (:doc:`tools`).
2. *Done.* Satellite frame camera: placed by altitude and look angles, with focal length, pixel pitch and pixel count; radiance per band in physical units; precision at orbital distances (:doc:`scene_elements`).
3. *Done.* Sun and atmosphere: the top-of-atmosphere solar spectrum and the sun's position; the atmosphere of the day (from AERONET's aerosol optical depth and water vapour) computed with 6S: the sun's and the camera's transmittances and a sky dome in the scene, the path radiance and the adjacency effect added to the render (:doc:`tools`).
   Matches 6S's top-of-atmosphere radiance of a uniform ground to 0.16%.
4. *Done.* Spectral materials: maps of materials' abundances, unmixed from NAIP, weighting their measured reflectance spectra (the ``spectral_mix`` texture, :doc:`scene_elements`; ``materials.py``, :doc:`tools`).
   Matches 6S's top-of-atmosphere radiance for the materials' mean spectrum to 0.19%.
5. *Done.* Detector: the telescope's and the jitter's MTFs, the pixels' integration of a finer render, conversion to electrons and digital numbers, noise and quantization (``detector.py``, :doc:`tools`).
   Its MTF matches a slanted edge's to 0.006, and its radiometry an independent calculation to 10⁻⁸.
6. *Done.* Time: platform motion (orbital ground speed, motion compensation, crab angle) and jitter (sinusoids, random), over one render; frame cameras' smear, pushbroom and TDI cameras (``detector.py``); event cameras (``event_camera.py``) (:doc:`tools`).
   Matches closed-form MTFs of smear, TDI synchronization errors and jitter to 0.0054, and an event camera's analytic events.
   Not yet: trajectories rendered frame by frame (views changing during an acquisition, parallax).
7. Later: Geiger-mode APD cameras (time-of-flight rendering), and the shortwave, longwave and thermal infrared (thermal emission).


Known issues
------------

* A ``directional`` emitter whose direction is exactly parallel to a coordinate axis (e.g. ``target="0m, -1m, 0m"``) lights a plane perpendicular to it with a pattern of rings, and too dimly; tilted by 0.001°, it lights it exactly. Suns computed from ephemerides are unaffected. Not yet investigated.
* Materials: the default playa spectrum is an analogue's (Stonewall Playa), not Railroad Valley's own (RadCalNet's, which needs registration); NAIP is uncalibrated and 8-bit, so abundances are only as good as the assumption that its values are proportional to radiance; at most four materials per ``spectral_mix``.
* Detector: the telescope's PSF is the same over the field (diffraction and a statistical wavefront error; no aberrations of a design, nor PSFs from optical design software, yet); the pixels have no MTF beyond their area (no diffusion or crosstalk); time is modelled over a single render: footprints move over it along its columns (north–south), without parallax or changes of view, so pushbroom scans of whole scenes with relief, and other headings, need renders per frame (not yet). Event cameras are sampled at fine time steps: seconds of a large array take hours. Renders must be oversampled (``k`` ≥ 2–4, so that the optics' cut-off is within the render's Nyquist frequency; ``detector.py`` warns otherwise), and their Monte Carlo noise kept below the detector's.
* Bands share wavelength samples (see ``band_radiance``): images of several disjoint bands are noisier per band than images of one.
* From a camera hundreds of kilometres away, ~10⁻⁴ of oblique views' samples still shadow themselves with ``near_clip``'s default margin (2 km); nearer margins leave fewer.
