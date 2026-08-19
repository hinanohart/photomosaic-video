# photomosaic-video

Still image: 2.3 s · 10 s of video: ~1 min · CPU only (Ryzen 7 7735HS). Built mostly by AI. Check the results yourself.

<img src="https://raw.githubusercontent.com/hinanohart/photomosaic-video/main/docs/demo.gif" width="100%" alt="A panda clip rebuilt frame by frame from a folder of dog photos">

<img src="https://raw.githubusercontent.com/hinanohart/photomosaic-video/main/docs/still_zoom.jpg" width="100%" alt="A still image rebuilt the same way, with one cell zoomed in to the photograph that fills it">

```bash
pip install photomosaic-video
photomosaic-video --demo                    # try it, no downloads
photomosaic-video clip.mp4 -t ./dog         # a video
photomosaic-video portrait.jpg -t ./dog     # a still image
```

Python 3.10+. Video also needs ffmpeg.

Photo packs and the full README: <https://github.com/hinanohart/photomosaic-video>

Thank you to [flutie8211](https://pixabay.com/users/flutie8211-17475707/) (Pixabay) for the panda clip.

Code: MIT. The 919 bundled photos: CC BY 2.0, credited in `photomosaic_video/ATTRIBUTION.md`.
