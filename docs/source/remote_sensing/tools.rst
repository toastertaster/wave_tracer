Tools
=======================================

``tools/remote_sensing/`` builds scenes of real sites. The scripts declare their Python dependencies inline: run them with `uv <https://docs.astral.sh/uv/>`_.

.. code-block:: sh

   uv run tools/remote_sensing/fetch_site.py sites/railroad_valley          # ~30 MB, ~20 s
   uv run tools/remote_sensing/build_scene.py sites/railroad_valley         # ~3 s
   ./build/wave_tracer render --gpu sites/railroad_valley/scene.xml -o out

``fetch_site.py OUT_DIR [--lat --lon --size --date]``
   Fetches, for a square area in the conterminous US (default: 2048 m around the RadCalNet site at Railroad Valley Playa), the USGS 3DEP 1 m elevation (``dem.tif``), the NAIP orthophoto nearest ``--date`` (``naip.tif``, default 2022-07-19), and the TSIS-1 solar spectrum (``tsis1_hsrs_1nm.csv``), with ``site.json`` describing them.
   Both rasters share the area's bounds in its UTM zone (NAD83); only the area is read from the sources (cloud-optimized GeoTIFFs).
   See :doc:`data` for the sources and their terms.

``build_scene.py SITE_DIR [options]``
   Writes ``terrain.ply`` (the elevation as a mesh, every ``--mesh-spacing`` DEM pixels: 2 m, ~2 M triangles for 2048 m), ``reflectance.png``, and ``scene.xml``:

   * the scene frame of the ``satellite_frame`` sensor (x east, y up, z south, origin on the ground at the area's centre);
   * a ``satellite_frame`` camera (``--altitude``, ``--view-zenith``, ``--view-azimuth``, ``--gsd``, ``--pixel-pitch``, ``--pixels``) recording band-averaged radiance in W/m²/sr/µm in four placeholder bands (blue 450–520, green 520–600, red 630–690, near infrared 760–900 nm);
   * the sun at its position at ``--time`` (default 2022-07-19T18:19:19Z, Sentinel-2B's pass), with the TSIS-1 spectrum scaled to that day's Earth–Sun distance;
   * no atmosphere yet: radiances are of the surface, under the top-of-atmosphere sun.

   Reflectance is a placeholder until spectral material maps: spectrally flat, scaled from NAIP's brightness so that its median is ``--reflectance`` (default 0.35). NAIP is not radiometrically calibrated, and its RGB, as a reflectance, would leave the near infrared dark.

   The scene's defines ``spp`` and ``ray_tracing`` (beams by default) can be overridden with ``-D``.

With the default scene (1920² pixels at 1 m GSD), 1024 samples per pixel render in 25 s on an NVIDIA GB10 (ray tracing); loading the 2 M-triangle terrain takes ~18 s.
Under the 2022-07-19 sun, the mean band radiances of the playa (reflectance 0.35) match a hand calculation, ρ E cos θ / π, to 0.3%.
