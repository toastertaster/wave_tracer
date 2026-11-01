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

``fetch_site.py OUT_DIR [--lat --lon --size --date --aeronet-radius]``
   Fetches, for a square area in the conterminous US (default: 2048 m around the RadCalNet site at Railroad Valley Playa), the USGS 3DEP 1 m elevation (``dem.tif``), the NAIP orthophoto nearest ``--date`` (``naip.tif``, default 2022-07-19), the TSIS-1 solar spectrum (``tsis1_hsrs_1nm.csv``), and AERONET's Level 2.0 daily averages of ``--date`` from the nearest site within ``--aeronet-radius`` (``aeronet.csv``, default 50 km; Railroad Valley's is 0.8 km away), with ``site.json`` describing them.
   Both rasters share the area's bounds in its UTM zone (NAD83); only the area is read from the sources (cloud-optimized GeoTIFFs).
   See :doc:`data` for the sources and their terms.

``build_scene.py SITE_DIR [options]``
   Writes ``terrain.ply`` (the elevation as a mesh, every ``--mesh-spacing`` DEM pixels: 2 m, ~2 M triangles for 2048 m) and ``scene.xml``:

   * the scene frame of the ``satellite_frame`` sensor (x east, y up, z south, origin on the ground at the area's centre);
   * a ``satellite_frame`` camera (``--altitude``, ``--view-zenith``, ``--view-azimuth``, ``--gsd``, ``--pixel-pitch``, ``--pixels``) recording band-averaged radiance in W/m²/sr/µm in four placeholder bands (blue 450–520, green 520–600, red 630–690, near infrared 760–900 nm);
   * the sun at its position at ``--time`` (default 2022-07-19T18:19:19Z, Sentinel-2B's pass), with the TSIS-1 spectrum scaled to that day's Earth–Sun distance;
   * with ``atmosphere.json`` (unless ``--no-atmosphere``), per wavelength: the sun attenuated by the direct transmittance and the gases' downward transmittance; the sky, ``sky.ply``, a black hemisphere of radius ``--sky-radius`` (100 km) of uniform radiance that gives a horizontal surface 6S's diffuse irradiance, including the light reflected back and forth between the ground (of the terrain's mean reflectance spectrum) and the atmosphere, with a hole around the sun; and the camera's ``band_radiance`` transmittance, the direct upward transmittance and the rest of the gases' two-way transmittance.
     The render is then the top-of-atmosphere radiance that comes straight from the ground; ``scene.json`` gives ``toa.py`` what it needs to add the rest. The ``satellite_frame`` camera's rays start inside the sky.
     Without an atmosphere, radiances are of the surface, under the top-of-atmosphere sun.

   * the terrain's reflectance: with ``materials.json`` (from ``materials.py``), a ``spectral_mix`` of the materials' spectra weighted by their abundances (``materials.exr``).
     Without it, or with ``--reflectance R``, a placeholder, ``reflectance.png``: spectrally flat, scaled from NAIP's brightness so that its median is R (default 0.35).


   The scene's defines ``spp`` and ``ray_tracing`` (beams by default) can be overridden with ``-D``.

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

``toa.py SCENE_JSON RENDER.exr OUT.exr [--components]``
   Adds to a render of a scene with an atmosphere, per band, the path radiance (sunlight scattered towards the camera by the atmosphere) and the adjacency effect: the render averaged with 6S's environment function (a satellite's, corrected for the view zenith; it reaches kilometres, so beyond the image the ground is taken to be as bright as the image's mean) times the ratio of diffuse to direct upward transmittance. ``--components`` also writes them separately.
   The image is taken as a grid on the ground at the nadir sampling distance: off nadir, an approximation.

With the default scene (1920² pixels at 1 m GSD), 1024 samples per pixel render in 25 s on an NVIDIA GB10 (ray tracing); loading the 2 M-triangle terrain takes ~18 s.
Under the 2022-07-19 sun without an atmosphere, the mean band radiances of the playa (reflectance 0.35) match a hand calculation, ρ E cos θ / π, to 0.3%.
With the atmosphere, a uniform ground of reflectance 0.35 renders 6S's own decomposition of the radiance that comes straight from the ground (Tg T↓ T↑\ :sub:`direct` ρ / (1 − ρ S) · E₀ cos θ / π) to 0.01–0.16% (blue to near infrared), and with ``toa.py``, 6S's top-of-atmosphere radiance likewise: 199.6, 172.5, 143.2 and 95.7 W/m²/sr/µm in the four bands.
With the materials of ``materials.py``, the mean radiance renders the same decomposition for the materials' mean spectrum to 0.02–0.18%, and with ``toa.py``, 6S's top-of-atmosphere radiance for that spectrum to 0.08–0.19% (161.0, 186.4, 196.3 and 145.2 W/m²/sr/µm; 1024 samples per pixel in 35 s).
