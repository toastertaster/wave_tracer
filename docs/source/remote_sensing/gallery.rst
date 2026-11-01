Gallery
=======================================

The Railroad Valley Playa scene (:doc:`tools`) on 2022-07-19, at Sentinel-2B's pass (the sun at a zenith of 26.25°), seen at nadir from 500 km by a ``satellite_frame`` camera of 2.5 m focal length, in four bands (blue 450–520, green 520–600, red 630–690, near infrared 760–900 nm), and imaged by the example instruments.
Rendered on the GPU (an NVIDIA GB10, CUDA + OptiX build, ray tracing) at ``f565eed``; times as the command line reports them (``Elapsed``, without loading the scene's 2.16 M triangles, ~40 s).
The images (but the events') are previews of the EXR outputs, made with ``preview.py``: values mapped linearly from the range given (by default from percentiles of the values), then sRGB-encoded.

.. list-table::
   :header-rows: 1
   :widths: 30 32 14 12

   * - Scene
     - Film
     - Samples
     - Time
   * - ``scene.xml``
     - 1920², 1 m GSD, 5 µm pixels
     - 8192
     - 4 min 58 s
   * - ``noatm.xml`` (``--no-atmosphere``)
     - 1920², 1 m GSD, 5 µm pixels
     - 8192
     - 3 min 21 s
   * - ``scene_k4.xml``
     - 4096², 0.25 m GSD, 1.25 µm pixels
     - 1024
     - 2 min 32 s

The scenes were built and rendered with the commands of :doc:`tools` (``build_scene.py sites/railroad_valley --name noatm --no-atmosphere`` for ``noatm.xml``), with ``-D spp=8192`` (``-D spp=1024`` for ``scene_k4.xml``), then, from the repository's root, for example:

.. code-block:: sh

   P="uv run tools/remote_sensing/preview.py"
   $P out/toa.exr toa_rgb.jpg --bin 3 --per-band --percentiles 0.1,99.9
   $P out/toa.exr toa_cir.jpg --bands nir,red,green --bin 3 --per-band --percentiles 0.1,99.9
   $P out/toa.exr toa_facility.jpg --crop 640,720,400,400 --per-band
   $P out_noatm/camera.exr surface_blue.jpg --bands blue --crop 640,720,400,400 --range 142,166
   $P out/toa.exr toa_blue.jpg --bands blue --crop 640,720,400,400 --range 142,166
   $P sites/railroad_valley/materials.exr materials.jpg --bands B,G,B --crop 1173,1307,666,666 --bin 2 --range 0,0.3
   $P out_k4/dn.exr detector_frame.jpg --crop 256,352,256,256 --zoom 2 --per-band


Top-of-atmosphere radiance
--------------------------

``toa.py``'s output: the render plus the path radiance and the adjacency effect.
Its mean radiances are 161.0, 186.4, 196.3 and 145.2 W/m²/sr/µm (blue, green, red, near infrared), of which the render is 114.1, 156.4, 178.5 and 137.8, the path radiance 31.8, 16.2, 7.3 and 2.3, and the adjacency effect 15.1, 13.8, 10.5 and 5.1.
The playa fills almost all of the 1920 m: the facility near its centre (roads, structures) is the most contrast the scene has, so each band is stretched over its own 0.1–99.9th percentiles.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: True colour (red, green, blue)
      :img-top: ../_static/remote_sensing/toa_rgb.jpg

      1920 m, binned 3 × 3.

   .. grid-item-card:: Colour infrared (near infrared, red, green)
      :img-top: ../_static/remote_sensing/toa_cir.jpg

      1920 m, binned 3 × 3.

.. figure:: ../_static/remote_sensing/toa_facility.jpg
   :width: 60%
   :align: center

   The facility, 400 m at full resolution (1 m), each band stretched over its 0.5–99.5th percentiles.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Surface radiance, blue
      :img-top: ../_static/remote_sensing/surface_blue.jpg

      ``noatm.xml``: the ground under the top-of-atmosphere sun, without an atmosphere. 142–166 W/m²/sr/µm.

   .. grid-item-card:: Top-of-atmosphere radiance, blue
      :img-top: ../_static/remote_sensing/toa_blue.jpg

      ``scene.xml`` and ``toa.py``, on the same scale: the atmosphere attenuates the ground's contrast, the path radiance offsets it, and the adjacency effect blurs it.


Materials
---------

.. figure:: ../_static/remote_sensing/materials_facility.jpg
   :width: 60%
   :align: center

   ``materials.py``'s abundances over the facility (``materials.exr``, on NAIP's 0.6 m grid, binned 2 × 2; 0–0.3): concrete green, asphalt magenta.
   The playa (99.1% of the site's mean abundance) is not shown.


Detectors
---------

``scene_k4.xml``'s top-of-atmosphere radiance (4 × 4 render pixels per detector pixel) imaged by ``detector.py`` onto 5 µm pixels (1 m GSD), 256 m of the facility, zoomed 2 ×.
Each band is stretched over its own percentiles.

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Radiance
      :img-top: ../_static/remote_sensing/detector_ideal.jpg

      ``toa.py``'s output, binned to the detector's pixels: no blur, no noise.

   .. grid-item-card:: Frame camera
      :img-top: ../_static/remote_sensing/detector_frame.jpg

      ``instrument.json``: 0.4 m telescope, 0.1 ms exposure; 551–1238 electrons, SNR 23–35.

   .. grid-item-card:: TDI camera
      :img-top: ../_static/remote_sensing/detector_tdi.jpg

      ``instrument_tdi.json``: 16 stages, 141.6 µs lines, 100 Hz jitter; 1.2–2.8 × 10⁴ electrons, SNR 110–166.

.. figure:: ../_static/remote_sensing/events.png
   :width: 60%
   :align: center

   ``event_camera.py``: 10 ms of an event camera of 10 µm pixels (2 m GSD) scanning at the orbit's ground speed, with thresholds of 0.03 (``event_camera.json``'s are 0.25, which the scene's contrast at 2 m never reaches).
   10 830 events (5061 ON, 5769 OFF), 354 time steps, in 45 s; their counts per pixel over 512 m around the facility, ON red, OFF blue (square root of the counts, zoomed 2 ×).
