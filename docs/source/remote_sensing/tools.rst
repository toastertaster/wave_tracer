Tools
=======================================

``tools/remote_sensing/`` builds scenes of real sites. The scripts declare their Python dependencies inline: run them with `uv <https://docs.astral.sh/uv/>`_.

.. code-block:: sh

   uv run tools/remote_sensing/fetch_site.py sites/railroad_valley          # ~30 MB, ~20 s
   python tools/remote_sensing/atmosphere.py sites/railroad_valley          # 6S: ~20 s (see below)
   uv run tools/remote_sensing/materials.py sites/railroad_valley           # ~1 min
   uv run tools/remote_sensing/build_scene.py sites/railroad_valley         # ~5 s
   ./build/wave_tracer render --gpu sites/railroad_valley/scene.xml -o out
   uv run tools/remote_sensing/toa.py sites/railroad_valley/scene.json out/camera.exr out/toa.exr

A town needs its buildings and trees: ``--lidar`` adds a surface model from 3DEP's lidar, and ``--sentinel2`` the Sentinel-2 surface reflectance that calibrates NAIP;
``materials.py`` then maps surface classes, each with its own materials, and ``build_scene.py`` meshes the surface model (the :doc:`gallery`'s cities):

.. code-block:: sh

   uv run tools/remote_sensing/fetch_site.py sites/reno --lat 39.533 --lon -119.814 --date 2022-06-15 \
       --lidar --sentinel2                                                       # ~1 GB of point clouds, ~5 min
   python tools/remote_sensing/atmosphere.py sites/reno --time 2022-06-15T18:39:19Z
   uv run tools/remote_sensing/materials.py sites/reno                           # classes; ~3 min
   uv run tools/remote_sensing/build_scene.py sites/reno --time 2022-06-15T18:39:19Z   # 8.4 M triangles, ~1 min
   ./build/wave_tracer render --gpu sites/reno/scene.xml -o out_reno -D spp=64

To image the scene onto a detector, render it on a grid finer than the detector's (here 4 × 4 render pixels per detector pixel), then:

.. code-block:: sh

   uv run tools/remote_sensing/build_scene.py sites/railroad_valley --name scene_k4 \
       --gsd 0.25 --pixel-pitch 1.25e-6 --pixels 4096 --spp 256 --ray-tracing
   ./build/wave_tracer render --gpu sites/railroad_valley/scene_k4.xml -o out_k4                     # ~40 s
   uv run tools/remote_sensing/toa.py sites/railroad_valley/scene_k4.json out_k4/camera.exr out_k4/toa.exr
   uv run tools/remote_sensing/detector.py sites/railroad_valley/scene_k4.json out_k4/toa.exr \
       tools/remote_sensing/instrument.json out_k4/dn.exr                                            # 1024² pixels
   uv run tools/remote_sensing/event_camera.py sites/railroad_valley/scene_k4.json out_k4/toa.exr \
       tools/remote_sensing/event_camera.json out_k4/events.npz --duration 0.01 --preview out_k4/events.png
   uv run tools/remote_sensing/preview.py out_k4/dn.exr out_k4/dn.jpg --per-band                    # a viewable image

``fetch_site.py OUT_DIR [--lat --lon --size --date --aeronet-radius --lidar --lidar-project --lidar-cache --sentinel2 --sentinel2-days]``
   Fetches, for a square area in the conterminous US (default: 2048 m around the RadCalNet site at Railroad Valley Playa), the USGS 3DEP 1 m elevation (``dem.tif``), the NAIP orthophoto nearest ``--date`` (``naip.tif``, default 2022-07-19), the TSIS-1 solar spectrum (``tsis1_hsrs_1nm.csv``), and AERONET's Level 2.0 daily averages of ``--date`` from the nearest site within ``--aeronet-radius`` (``aeronet.csv``, default 50 km; Railroad Valley's is 0.8 km away), with ``site.json`` describing them.
   Both rasters share the area's bounds in its UTM zone (NAD83); only the area is read from the sources (cloud-optimized GeoTIFFs).
   The DEM is the newest 3DEP project that covers the whole area (else the projects' tiles merged, newest first): a project's tiles can stop at a state line.
   See :doc:`data` for the sources and their terms.

   ``--lidar`` adds a surface model, ``dsm.tif`` (``lidar.py``): the highest return of each 1 m cell of a 3DEP lidar collection (the newest that covers the area, or ``--lidar-project``), and its ASPRS class (``lidar_class.tif``: ground, vegetation, building, water, bridge).
   The returns come from 3DEP's Entwine Point Tiles on AWS, of which only the octree's nodes over the area are read, else from 3DEP's LAZ tiles (whole tiles, from a slower server); both are kept in ``--lidar-cache`` (``~/.cache/wave_tracer/lidar``).
   Noise classes are dropped, the collection's vertical datum is matched to the DEM's (by the median difference of its ground returns), cells without returns take the DEM's height, and isolated spikes (a cell with at most 2 of its 24 neighbours within 8 m below it: wires, birds, missed noise) are lowered to their surroundings.
   ``--sentinel2`` adds ``sentinel2.tif``: Sentinel-2's Level-2A surface reflectance (bands 2, 3, 4, 8 and the scene classification) of the clearest pass within ``--sentinel2-days`` (20) of ``--date``, on the area's 10 m grid, which ``materials.py`` calibrates NAIP with.

``build_scene.py SITE_DIR [options]``
   Writes ``terrain.ply`` (the elevation as a mesh, every ``--mesh-spacing`` pixels) and ``scene.xml``.
   The elevation is the lidar surface model if the site has one (``--surface dsm``: roofs and tree crowns, flat-shaded triangles every 1 m, 8.4 M triangles for 2048 m), else the bare-earth DEM (``--surface dem``: every 2 m, ~2 M triangles, with vertex normals).
   With ``materials.py``'s classes, the terrain is split by class instead (``terrain_<class>.ply``: each triangle the class at its centroid; steep triangles, over 60°, that are not vegetation are walls, ``terrain_wall.ply``), each with its own reflectance.
   The scene has:

   * the scene frame of the ``satellite_frame`` sensor (x east, y up, z south, origin on the ground at the area's centre);
   * a ``satellite_frame`` camera (``--altitude``, ``--view-zenith``, ``--view-azimuth``, ``--gsd``, ``--pixel-pitch``, ``--pixels``) recording band-averaged radiance in W/m²/sr/µm in four placeholder bands (blue 450–520, green 520–600, red 630–690, near infrared 760–900 nm);
   * the sun at its position at ``--time`` (default 2022-07-19T18:19:19Z, Sentinel-2B's pass), with the TSIS-1 spectrum scaled to that day's Earth–Sun distance;
   * with ``atmosphere.json`` (unless ``--no-atmosphere``), per wavelength: the sun attenuated by the direct transmittance and the gases' downward transmittance; the sky, ``sky.ply``, a black hemisphere of radius ``--sky-radius`` (100 km) of uniform radiance that gives a horizontal surface 6S's diffuse irradiance, including the light reflected back and forth between the ground (of the terrain's mean reflectance spectrum) and the atmosphere, with a hole around the sun; and the camera's ``band_radiance`` transmittance, the direct upward transmittance and the rest of the gases' two-way transmittance.
     The render is then the top-of-atmosphere radiance that comes straight from the ground; ``scene.json`` gives ``toa.py`` what it needs to add the rest. The ``satellite_frame`` camera's rays start inside the sky.
     Without an atmosphere, radiances are of the surface, under the top-of-atmosphere sun.

   * the terrain's reflectance: with ``materials.json`` (from ``materials.py``), a ``spectral_mix`` of the materials' spectra weighted by their abundances (``materials.exr``); with classes, each class's (``materials_<class>.exr``), and the walls' material.
     Without it, or with ``--reflectance R``, a placeholder, ``reflectance.png``: spectrally flat, scaled from NAIP's brightness so that its median is R (default 0.35).


   The scene's defines ``spp`` and ``ray_tracing`` (beams by default) can be overridden with ``-D``.
   ``--fsd``/``--no-fsd`` turn the beams' free-space diffraction by the terrain's edges on or off: on by default for the DEM, off for the surface model, whose millions of sharp edges make it slow and leave fireflies (and diffraction by metre-sized edges is negligible at these wavelengths).

``atmosphere.py SITE_DIR [--time --view-zenith --view-azimuth --aot550 --water --ozone]``
   Computes the atmosphere at the site with 6S (Vermote et al. 1997), at 400–1000 nm every 2.5 nm (6S's resolution), for the sun at ``--time`` and the camera's view: a continental aerosol model with AERONET's optical depth at 550 nm (from 500 nm and the 440–870 nm Ångström exponent) and precipitable water, ozone 0.30 cm-atm, the ground at the site's elevation. Writes ``atmosphere.json``: the direct and total transmittances down and up, the diffuse upward transmittances of molecules and of aerosols, the gases' transmittances (down, up and two-way: in absorption bands the two-way one is not the product), the spherical albedo and the intrinsic reflectance.
   It needs the 6S executable and Py6S, which are not on PyPI together: e.g. ``micromamba create -n rs -c conda-forge sixs python=3.12 numpy``, then ``pip install Py6S python-dateutil`` in it, and run the script with that environment's Python (``--sixs`` gives the executable's path, if it is not on ``PATH``).
   For 2022-07-19: AOT 0.058 at 550 nm, 1.89 cm of water.

``materials.py SITE_DIR [--material NAME=SOURCE[*SCALE] ...] [--library --colour-scale]``
   Maps the site's materials, as abundances of measured reflectance spectra, by unmixing NAIP: each pixel's four band values are taken to be a non-negative combination of up to four materials' reflectances in those bands (exact non-negative least squares).
   Writes ``materials.exr`` (half-float RGBA: one material's abundance per channel, on NAIP's grid) and ``materials.json`` (the materials' spectra at 350–2500 nm every 1 nm, their sources and mean abundances, and the fit's residual).

   A material's ``SOURCE`` is part of the file name of a spectrum of the USGS Spectral Library 7, convolved to an ASD spectrometer's resolution (downloaded once, 28 MB, to ``--library``, default ``~/.cache/wave_tracer/``), or a CSV file of wavelengths (nm) and reflectances; ``*SCALE`` scales it.
   The defaults suit a dry playa: ``playa`` (Stonewall Playa, Nevada, dry mud: an analogue of Railroad Valley's surface, whose own spectrum, from RadCalNet, can be given as a CSV), ``concrete`` (light grey road: gravel roads and structures) and ``asphalt`` (old black road: dark surfaces).

   NAIP is not radiometrically calibrated: its 8-bit values are taken to be proportional to radiance, its median pixel to be the first material, and each band is scaled accordingly (its bands taken to be the Leica ADS100's boxes: blue 420–492, green 533–587, red 604–664, near infrared 833–920 nm).
   NAIP's mosaics vary in colour over hundreds of metres (vignetting, seams between flight lines), so of its colour only the detail finer than ``--colour-scale`` (default 256 m) is kept; brightness is kept at every scale.
   At Railroad Valley: 99.1% playa, 0.7% concrete, 2.0% asphalt (mean abundances; they need not sum to one), with an RMS residual of 0.0009 in reflectance.

   With ``--classes`` (the default for a site with a lidar surface model and ``sentinel2.tif``), a town's surfaces are mapped by class instead:

   #. NAIP is calibrated to surface reflectance, per band, by a robust linear fit of its means over Sentinel-2's 10 m pixels (clear land) to Sentinel-2's Level-2A reflectance.
   #. Each NAIP pixel gets a class, from the lidar's classes and heights above the bare earth and NAIP's NDVI (``--ndvi``, 0.3; ``--min-height``, 2.5 m): water, building, tree, grass or ground (roads, pavement, bare soil). ``classes.png`` has them.
      Collections can leave most of a river's returns unclassified, so low, dark cells within 300 m of water returns and within 1 m of their height are water too.
   #. NAIP's own shadows are found (the sun's direction during NAIP's flight is fitted: the one whose shadows, cast by the surface model, best match NAIP's dark pixels), and shadowed pixels take the reflectance of the nearest sunlit pixel of their class: the renderer casts its own shadows.
   #. Each class is unmixed into four materials, chosen from its candidates (USGS spectra; ``--class-material CLASS:NAME=SOURCE`` replaces them) as the four whose mixtures best fit the class's sunlit pixels.
      The candidates (``CLASS_CANDIDATES``): for ground, asphalt, concrete, dry mud, lawn, dark and tan paving bricks, dry grass and sand; for grass, green, dry, mixed and cheat grass, dry mud and asphalt; for trees, maple, walnut, oak and aspen leaves, dry grass and asphalt (gaps in the crowns); for buildings, white and green fibreglass roofing, light grey, dark grey and tan asphalt shingles, tar roofing, red and light grey building bricks, concrete and galvanized sheet metal; for water, coastal seawater, water with 0.5 and 5 g/l of clay, and marsh water.
      Here abundances are fractions of the surface: they sum to at most one, the rest being shade finer than NAIP's pixels, so that each material keeps its own brightness.
   #. Walls, which NAIP does not see, get one material, ``--wall-material`` (default a light grey building brick).

   ``materials_<class>.exr`` has each class's abundances.

``toa.py SCENE_JSON RENDER.exr OUT.exr [--components]``
   Adds to a render of a scene with an atmosphere, per band, the path radiance (sunlight scattered towards the camera by the atmosphere) and the adjacency effect: the render averaged with 6S's environment function (a satellite's, corrected for the view zenith; it reaches kilometres, so beyond the image the ground is taken to be as bright as the image's mean) times the ratio of diffuse to direct upward transmittance. ``--components`` also writes them separately.
   The image is taken as a grid on the ground at the nadir sampling distance: off nadir, an approximation.

``detector.py SCENE_JSON RADIANCE.exr INSTRUMENT.json OUT.exr [--electrons E.exr --seed N]``
   Images a render (of band-averaged spectral radiance: ``toa.py``'s output, or a render without an atmosphere) through a telescope onto a moving detector (a frame, pushbroom or TDI camera), and writes its digital numbers: one channel of unsigned integers per band.
   The render's pixel pitch must divide the detector's (``k`` render pixels per detector pixel, across); its focal length and bands are the scene's (``scene.json``). Per band:

   #. The telescope blurs the radiance, by its modulation transfer functions (in the Fourier domain, with the image extended by reflection): diffraction by a circular aperture with a central obscuration, averaged over the band's wavelengths (weighted by their photons); the wavefront error (Shannon's approximation for a random error, 1 − (W/0.18)² (1 − 4(ν − ½)²), W the RMS error in waves, ν the frequency over the cut-off; good to ~0.15 waves); and, as a Gaussian, jitter faster than the time steps (``jitter_urad_rms``).
      The render's own blur beyond its pixels' area, the film's Gaussian reconstruction filter (σ ¼ render pixel), is divided out.
   #. Each detector pixel integrates the radiance over its area (``k`` × ``k`` render pixels: its MTF, and aliasing), where the platform's motion puts its footprint at each time step of its exposure (between render pixels, by cubic spline interpolation).
   #. Radiance becomes electrons: *L* Δλ (λ̄/*hc*) *A* Ω τ η *t*, Δλ and λ̄ the band's width and centre, *A* the pixel's area, Ω the projected solid angle of the (annular) aperture from the focal plane, τ the optics' transmittance, η the quantum efficiency, *t* the integration time.
   #. Noise and quantization: photo-response non-uniformity multiplies the signal and dark current adds to it (with its own non-uniformity), both fixed patterns of the instrument's seed; Poisson shot noise of both; the full well clips; Gaussian read noise adds; the converter divides by its gain, adds its offset, rounds and clips to its bits.

   Time: the scene is rendered once, and the platform's motion moves the pixels' footprints over that render (which ignores the parallax and the change of view during an acquisition: right over a TDI line's few milliseconds, approximate for a pushbroom scan of a whole scene with relief).
   The footprints move along the render's columns (its rows are along track: north–south, as ``satellite_frame`` cameras look at nadir), at the ground speed of a circular orbit at the scene's altitude (or ``platform.ground_speed_m_s``), less ``platform.motion_compensation`` (a fraction: 1 for a staring camera), towards the south (``platform.scan``, or ``"north"``), at ``platform.drift_deg`` from the columns (a crab angle).
   The line of sight jitters: ``platform.jitter`` has sinusoids (``frequency_hz``, ``along_urad``, ``across_urad``, ``phase_deg``) and random jitter (``random_urad_rms`` per axis, of a Lorentzian spectrum of corner ``random_bandwidth_hz``), of a ``seed``. Without ``platform``, nothing moves. The camera (``camera.type``) is

   * ``frame``: every pixel integrates during ``integration_time_s``;
   * ``pushbroom``: a line of pixels across track; each row of the image is a line period later;
   * ``tdi``: time delay and integration, the charge of each row moving with the image across ``camera.stages`` lines of pixels, one per line period; ``camera.readout`` ``"charge"`` (a CCD's: read once) or ``"digital"`` (each line read: √stages times the read noise, and each line's full well).

   The line period is ``camera.line_period_s``, or by default the time the footprints take to move by one pixel, times 1 + ``camera.line_period_error`` (a synchronization error); lines integrate during ``integration_time_s`` (default: the whole period).
   Lines' PRNU and dark current are per line pixel, averaged over the stages a row's charge visits.
   Exposures are sampled at ``camera.time_steps`` (default: steps of at most a tenth of a pixel of motion, and ten per period of the jitter's fastest component).

   ``INSTRUMENT.json`` gives the telescope (``aperture_diameter_m``, ``obscuration``, ``transmittance``, ``wavefront_error_nm_rms``), ``jitter_urad_rms`` (per axis), the detector (``pixel_pitch_m``, ``integration_time_s``, ``quantum_efficiency``, ``full_well_e``, ``read_noise_e``, ``dark_current_e_per_s``, ``prnu``, ``dsnu``, ``gain_e_per_dn``, ``offset_dn``, ``adc_bits``, ``fixed_pattern_seed``), ``camera`` and ``platform``; any number may be given per band, as an object keyed by band.
   Examples in ``tools/remote_sensing/``: ``instrument.json``, a 1 m GSD frame camera at 500 km (2.5 m focal length, 5 µm pixels) with a 0.4 m telescope (f/6.25, obscuration 0.3, 40 nm RMS wavefront error) and a 0.1 ms exposure (0.7 pixels of motion); ``instrument_tdi.json``, its telescope with a 16-stage TDI camera and a 100 Hz jitter line.
   It prints per band the electrons per unit radiance, the mean signal and its SNR, λ*N*/*p*, and the static MTF at the detector's Nyquist frequency; ``--electrons`` also writes the signal's electrons without noise.

   The render's own noise adds to the detector's: render twice and compare, per detector pixel. At Railroad Valley (``k`` = 4), 64 samples per render pixel leave ~2% of noise per detector pixel, 256 ~1%, against the frame camera's 3–4.5%; the TDI camera's 0.6–0.9% needs thousands.

``event_camera.py SCENE_JSON RADIANCE.exr INSTRUMENT.json OUT.npz --duration S [--band NAME --seed N --preview PNG]``
   An event camera's pixels report changes of brightness asynchronously: a pixel emits an ON (OFF) event when the logarithm of its photocurrent rises (falls) by a contrast threshold since its last event.
   Its pixels see what ``detector.py``'s do (the telescope, the platform's motion and jitter), sampled at time steps; per pixel:

   * the photocurrent: the electrons per second of the render's bands (each with its quantum efficiency; or only ``--band``'s), plus the dark current;
   * its logarithm, through the photoreceptor's first-order low-pass filter (``events.cutoff_hz``, optional; solved exactly for a current varying linearly within a step);
   * an event for each crossing of the thresholds (``events.threshold_on``, ``events.threshold_off``: contrasts in the natural logarithm, each pixel's drawn with an RMS of ``events.threshold_sigma``, a fixed pattern) since the pixel's reference level, timed by interpolation within the step; the reference moves by the threshold. During ``events.refractory_s`` after an event, the pixel emits none, and its reference stays;
   * noise events of random polarity at ``events.noise_rate_hz`` per pixel.

   The time steps (``events.time_step_s``) default to a tenth of a pixel of motion and a tenth of the jitter's fastest period.
   ``OUT.npz`` has the events in time order (``x``, ``y``, ``t`` in µs, ``p`` ±1) and the camera's ``width`` and ``height``; ``--preview`` draws their sums per pixel (ON red, OFF blue).
   ``tools/remote_sensing/event_camera.json`` is an example: the telescope with 10 µm pixels (2 m GSD), scanning at the orbit's ground speed (as Falcon Neuro did from the ISS), thresholds of 0.25.
   At Railroad Valley the largest log contrast at 2 m is 0.24: with these thresholds, the camera sees only its noise events; with thresholds of 0.03, the facility's edges stream events along the scan (10 ms of 512² pixels, 354 steps, in 25 s).

``preview.py IN.exr OUT [--bands --crop --bin --zoom --range | --percentiles --per-band --linear --quality]``
   Writes a viewable image (JPEG or PNG, by ``OUT``'s extension) of an EXR's channels: a render, ``toa.py``'s, ``detector.py``'s output, or ``materials.exr``.
   Three channels (``--bands R,G,B``; default ``red,green,blue`` when the EXR has them) make a colour image, one a grey one.
   ``--crop X,Y,W,H`` takes a window of the EXR (pixels), ``--bin N`` averages N × N pixels of it, ``--zoom N`` repeats each pixel N × N times.
   Values map linearly to black and white from ``--range LO,HI`` (one range, or one per channel: ``LO1,HI1,LO2,HI2,LO3,HI3``), or by default from the ``--percentiles P0,P1`` (0.5,99.5) of all the chosen channels together, which keeps their balance (``--per-band``: each channel's own), then are sRGB-encoded (unless ``--linear``). It prints the range used.
   The :doc:`gallery`'s images (but the events') are its previews.

``telescope.py {two-mirror,hubble} [--kind --diameter --focal-length --primary-focal-length --back-focus --off-axis --field-bias --fold --secondary-radius --coating --pixel-pitch --pixels --altitude --view-zenith --view-azimuth --samples --elements-only]``
   Writes a ``telescope`` sensor (:doc:`scene_elements`) of a design: its mirrors, their clear apertures and placement, and the focal plane's frame.
   ``two-mirror`` is a Cassegrain telescope from its first-order layout (default the example instrument's: 0.4 m aperture, 2.5 m focal length, a 0.6 m primary focal length, 0.15 m back focus): the secondary's magnification *m* = *f*/*f*\ :sub:`1`, its distance *f*\ :sub:`1` (1 − *k*) from the primary, *k* = (1 + *b*/*f*\ :sub:`1`)/(1 + *m*), its radius 2 *k f*\ :sub:`1` *m*/(*m* − 1) (Schroeder, *Astronomical Optics*), with ``--kind cassegrain`` (a paraboloid and a hyperboloid) or ``ritchey-chretien`` (aplanatic conics; the default).
   The secondary's aperture and the primary's hole pass the film's field.
   ``--off-axis Y`` takes an unobscured off-axis segment of the design, its primary's aperture centred Y from the axis; ``--field-bias`` puts the film's centre off the axis; ``--fold`` adds a flat mirror at 45° in the converging beam.
   ``hubble`` is the Hubble Space Telescope (Laikin, as in Optiland's sample).
   ``--coating`` coats every mirror: ``ideal`` (the default), ``aluminium``, ``silver`` or ``gold`` (bare, from wave_tracer's data), or ``protected-aluminium`` (under 25 nm of magnesium fluoride, as Hubble's).
   It prints the sensor, placed by its ``--altitude`` and view angles (strings, so that a scene's defines such as ``$vz`` pass through), or with ``--elements-only`` only the optics, to paste into a sensor.
   As a module, ``two_mirror()`` and ``hubble()`` return designs and ``sensor_xml()`` writes them (e.g. for ``build_scene.py``).

``telescope_reference.py``
   Prints, as JSON, Optiland's (an independent ray tracer's) values that the Rust bindings' tests check the ``telescope`` sensor against: the designs' paraxial focal lengths and entrance pupils, and a vignetting design's relative throughput at the film's corner (the entrance pupil's area whose rays pass, times the solid angle per focal-plane area of the image's height h(θ): A(θ) cos θ sin θ/(h h'), by the conservation of étendue).
   Each design is modelled as the sensor sees it, with its obstacles (the secondary's shadow, the primary's hole) as apertures on the planes the rays cross.

With the default scene (1920² pixels at 1 m GSD), 1024 samples per pixel render in 25 s on an NVIDIA GB10 (ray tracing); loading the 2 M-triangle terrain takes ~18 s.
Under the 2022-07-19 sun without an atmosphere, the mean band radiances of the playa (reflectance 0.35) match a hand calculation, ρ E cos θ / π, to 0.3%.
With the atmosphere, a uniform ground of reflectance 0.35 renders 6S's own decomposition of the radiance that comes straight from the ground (Tg T↓ T↑\ :sub:`direct` ρ / (1 − ρ S) · E₀ cos θ / π) to 0.01–0.16% (blue to near infrared), and with ``toa.py``, 6S's top-of-atmosphere radiance likewise: 199.6, 172.5, 143.2 and 95.7 W/m²/sr/µm in the four bands.
With the materials of ``materials.py``, the mean radiance renders the same decomposition for the materials' mean spectrum to 0.02–0.18%, and with ``toa.py``, 6S's top-of-atmosphere radiance for that spectrum to 0.08–0.19% (161.0, 186.4, 196.3 and 145.2 W/m²/sr/µm; 1024 samples per pixel in 35 s).

The ``telescope`` sensor, checked by the Rust bindings' tests (``tests/telescope.rs``) against Optiland and first-order optics:

* Focal lengths (the image centroid's scale, from the lines of sight about the film's centre): the Ritchey-Chrétien's 2.50004 m and Hubble's 57.6031 m against Optiland's paraxial 2.5 m and 57.6001 m; the classical Cassegrain's 2.50424 m against 2.50425 m, its paraxial focal length lengthened by its coma, f (1 + ⟨U²⟩/2).
* The film centre's throughput, π (sin²U − sin²u) of the outer and inner marginal rays: to 0.01% with Abbe's sine condition for the aplanatic designs, sin U = h/f, and with a paraboloid's, h = 2f tan(U/2), for the Cassegrain; entrance pupils and obscurations to 10⁻⁵.
* Vignetting by an undersized secondary, 0.51° off axis: a relative throughput of 0.944546 against Optiland's 0.944555 (5.31% of the pupil's rays blocked, and the distortion's 0.24%).
* Radiometry: a uniformly lit plane renders as with a ``satellite_frame`` camera, to 0.2% over the film and 2% in its corners (Monte Carlo noise), tracing rays or beams.
* Coatings, against Fresnel's equations and Airy's summation for thin films (Born & Wolf), written independently in the tests: normal reflectances to 10⁻⁵ (2·10⁻⁴ for aluminium from wave_tracer's data, interpolated differently), including an absentee half-wave overcoat; two aluminium mirrors dim a uniform plane to 0.84798 of ideal ones (R² = 0.84845, at normal incidence).
* Polarization, averaged over the beam converging on the film: a 45° aluminium fold polarizes an unpolarized scene to Q/I = 0.02900 along its s, the film's x (0.02900 expected); under a quarter-wave overcoat, to −0.00566 (−0.00569: p now reflects more); a periscope's second fold, its plane of incidence turned 45°, turns it circular, |V|/I = 0.006027 (0.006027) bare and 0.005816 (0.005824) overcoated, with the same sign under 0.1 nm of overcoat as bare.
* Geometry: a marker images where a pinhole camera of the focal length would, to 0.15 pixels, at nadir and 30–45° off it, through the centred Ritchey-Chrétien, its off-axis segment (whose pupil, 0.2 m off the axis, sees the ground from there) and the folded one (mirrored); the off-axis segment with its field 0.3° off the axis points it at the target.

``detector.py``, checked against independent calculations:

* MTF: a slanted edge (5°) rendered at 1.25 µm pixels, imaged onto 5 µm pixels, measures the model's system MTF (telescope × jitter × pixel) to 0.006 up to the Nyquist frequency (0.218 against 0.222 there, green). The renders' own MTF, box × Gaussian (σ ¼ pixel), measures to 0.003 to 1 cycle per pixel, with ray and with beam tracing.
* Radiometry and noise: a uniform radiance gives the electrons of a numerical integration of its photons to 10⁻⁸, and digital numbers whose mean and standard deviation match the model's (signal, dark current, PRNU, DSNU, read and quantization noise) to 0.02 DN and 0.15%.
* Railroad Valley at the example instrument: 551, 1072, 998 and 1238 electrons (blue, green, red, near infrared) and an SNR of 23–35 at the mean radiance; 26 s for 4096² render pixels, after ``toa.py``'s 22 s. With the TDI camera, 1.2–2.8 × 10⁴ electrons and an SNR of 110–166, in 52 s.
* Time, against closed forms (the slanted edge, along track or across it; the residual is the static case's): a frame camera's smear of one pixel (× sinc *f*), a synchronous 8-stage TDI camera (× sinc *f* along track, and 8 times a line's electrons), one whose line period is 5% long (× sinc((1 + ε) *f*) sin(π *f N* ε) / (*N* sin(π *f* ε))), and a staring camera's sinusoidal jitter of 0.5 pixels over many periods (× J₀(2π *a f*)), each to 0.0054 up to the Nyquist frequency. The random jitter's RMS and autocorrelation match its model's to 0.3% and 0.003.

``event_camera.py``: an exponential radiance (*L* ∝ e\ :sup:`ax`, whose blurred logarithm rises at exactly *a v* in a pixel moving at *v*) gives every pixel an ON event every 12.504 ms (12.5 expected), half as many (10) with a refractory period of twice their spacing, and with a 10 Hz low-pass filter, events within 20 µs of the analytic lag's times; noise events count as their rate gives (32 774 for 32 768).
