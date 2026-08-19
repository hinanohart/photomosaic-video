# How photomosaic-video works

* Each cell gets its own tile, chosen in one global step that minimises the total mismatch (§1, §3).
* "Close" means shape and texture first, colour second. Faces and edges get the best tiles (§3.3).
* Each tile is then recoloured toward its cell. That makes the mosaic read as one picture, and it is the main trade-off (§4).
* A video keeps the previous frame's tiles and changes only cells whose picture changed, so it does not boil (§5).
* The same input gives the same frames on every run (§8). When a mosaic looks wrong, `--diagnose` usually blames the photos (§7).

Where numbers come from: **(source)** a constant or comment in the code. **(log)** the project's records, mostly
from earlier builds and other photo folders: typical, not exact. **(benchmark)** one still-image test on the
August 2026 build: 12 targets, a 4,254-photo dog folder, grids of 486, 1,350 and 3,015 cells, one machine.

Pipeline: cut the frame into a grid (§3.1), load the photos (§3.2), price every cell/tile pair (§3.3), assign one
tile per cell and re-check it by shape (§3.4–3.5), keep copies of a photo apart (§3.6), recolour (§4), finish (§6).
A video runs this on the first frame and after each scene cut; other frames redo only what changed (§5).

## 1. Why the obvious method fails

**Pasting the closest-colour photo into each cell fails, because one photo wins everywhere it fits.** A clear sky
is thousands of cells asking the same question, so your three bluest photos repeat across a third of the frame.
That reads as a texture, not a picture. (Robert Silvers popularised the photomosaic in the 1990s [1].)

Using each tile **once** fixes that, but cells now compete for tiles: one global decision, the **assignment
problem**. Solving it exactly is most of photomosaic-video. *n* cells need *n* tiles, so each photo becomes
**four**: itself, its mirror, and ±10% brightness. The floor is *n*/4 photos; with fewer, the run stops and says so.

## 2. Colour: why Oklab

**Colours are compared in Oklab [2], where numeric distance tracks visible difference.** sRGB does not (black to
dark grey is a small number but a big step), and CIELAB is uneven in blue, where skies and shadows live. Oklab is
cheap, and is used in the cost (§3.3), the vibrance stage (as Oklch), the video swap (§5) and `--diagnose` (§7).

## 3. Choosing which photo goes where

### 3.1 The grid

**The frame is scaled to `--height` and cut into square cells; each refusal below is a clear error, not a crash.**

| setting | what happens |
|---|---|
| cell size | 20–120 px **(source)**. Under 20 a tile stops reading as a photo; over 120 the bundled 120 px photos would only be enlarged. |
| `--cells` (700) | The cell size must divide the height, so the count nearest the request is used. 700 on 16:9 gives 54 px cells, 35 × 20, 1890 × 1080 **(source)**. |
| `--height` (1080) | Raised automatically when `--cells` cannot fit at 20 px, so a big request gives a big image. Odd heights, or heights no size from 20 to 120 divides, are refused. |
| width | Trimmed to whole cells, so nothing is resized at the end. An odd width moves by one cell, since libx264 refuses odd sides (`--cells 3000` once gave 1917 **(log)**). |
| size limit | 64 megapixels, for the working frame and for a video source: about 160,000 cells at 20 px. A very wide or tall source can hit it at a small `--cells`. |

### 3.2 Reading the photo folder

**Each photo is centre-cropped to a square, shrunk to the cell size, made into four tiles, and cached.**

* Every subfolder is read (`.jpg .jpeg .png .bmp .webp`). The crop cuts off the long edges, which is why close-ups
  work best: the subject survives and is still recognisable at cell size.
* Decoding is the slowest step, so the cache is keyed on each photo's resolved path, size and modification time
  plus the cell size (any change means a rebuild). The 8 most recent sets live in `photomosaic-video/` under
  `$XDG_CACHE_HOME`, else `%LOCALAPPDATA%`, else `~/.cache`: delete it to clear. Writes are atomic.

### 3.3 What "close" means

**Shape and texture pick the tile; colour only breaks ties.** Each cell and tile is described by **191 numbers** in
CIELAB: colour of 5×5 blocks (75), colour histograms (36), edge-direction histograms (40) and texture histograms
(local binary patterns [3], 40). Colour alone cannot tell half-black, half-white from grey; the others can.

**Cost** of a pair: the squared distance between the two vectors, divided by the matrix maximum. For each cell's
**100 cheapest** tiles only, an Oklab distance is added at weight **0.20**. Colour cannot rescue a wrong shape.

**Saliency** sends good tiles to faces and edges. Each cell's costs are scaled by a weight averaging 1.0:
`0.2 + 1.5 × edge density (Canny) + 1.0 × Laplacian-of-Gaussian energy + 0.5 × HSV saturation`, then ×**1.6**
inside any face box from a Haar cascade [4]; with no face, a mild centre bias (1.0 in the middle, 0.7 at corners).

<details>
<summary>The 191 numbers in detail</summary>

| count | what it captures |
|---|---|
| 75 | average colour of each of 5×5 blocks (where colour sits in the cell) |
| 36 | 12-bin histogram per channel (which colours are present) |
| 40 | gradient-orientation histograms, 8 bins, 4 sub-blocks plus the whole cell (edge direction) |
| 40 | local binary pattern histograms, 10 bins, 4 sub-blocks (fine texture) |

</details>

### 3.4 Solving it on a thinned graph

**The assignment is solved exactly [5][6], but only over each cell's K cheapest tiles:**
`k = min(n_tiles, max(400, ceil(0.8 * n_cells)))`. A full matrix would hold 18 million entries for 576 cells and
32,000 tiles. If scipy finds no full matching, K doubles, then a dense solve. A tight pool needs more photos.

<details>
<summary>Why K scales with the cell count, and how close the thinned answer is</summary>

* Below about **0.5 × cells** no full matching exists and scipy raises.
* Just above that, scipy does **not** raise: it quietly returns a **worse** answer. A constant K was rejected for
  this reason (silently worse between about 0.55n and 0.68n **(log)**).
* 576 cells, 2,800 tiles (4.9 per cell): identical to the dense optimum at every K from 100 to 2,800 **(log)**.
* 2,484 cells: identical at 1.6 and 1.3 tiles per cell, then 0.007% worse at 1.09× and 0.02% at 1.03× **(log)**.
* scipy sparse matrices drop stored zeros, so a zero-cost edge would vanish. Every edge gets **+1.0**; every full
  matching has the same number of edges, so the winner is unchanged.

</details>

### 3.5 A second look, by shape

**Each cell then re-checks its 25 cheapest candidates by shape and swaps only for a clear win.** Cheap grayscale
cross-correlation ranks them, **SSIM** [7] scores the best **4**, and the challenger must win by a factor of
**1.002**, so ties do not flip on floating-point noise (in video, a flicker). A `used` set keeps one tile per cell.

### 3.6 Keeping the copies apart

**A photo may fill up to four cells (as its variants); when two of them touch, cells trade tiles until they don't.**
Reuse is deliberate: four variants of 2,000 photos match better than 8,000 distinct photos **(source)**, and
banning it costs **27% more colour error (log)**. But a mirror pair in touching cells looks like a bug.

Each offending cell swaps with the cell that breaks the pair most cheaply; a swap is a permutation, so one tile per
cell holds. Added colour error: 0.1% or less at 750 cells, 0.9% at 288 cells **(log)**. It is best effort (a cell
only looks among its own cheap candidates), yet touching duplicates were **0.000** at every grid size **(benchmark)**.

## 4. Making the chosen photo fit

**Each tile is recoloured toward its cell, every step partial: at full strength a photo becomes a colour swatch.**

| step | strength **(source)** |
|---|---|
| linear Monge–Kantorovich transfer (MKL) [9]: the full colour covariance in CIELAB, not just each channel's mean and spread [8] | 0.50 |
| histogram matching (the whole distribution), blended over it | 0.30 |
| the target cell blended over the tile | 5–14% by saliency; at least 30% for dark cells (mean BGR below 60) |

Dark cells get more of the target because dark photos run out: one run had 1,122 dark tiles for 2,273 dark cells,
and only 3.5% of the photos were dark **(log)**. A brightness penalty in the placement did nothing, since no penalty
creates missing stock. Showing the target is the workaround; darker photos are the fix (§7).

### What this costs

**The recolouring is why the mosaic reads as a picture, and it is the main trade-off (benchmark):**

| | result |
|---|---|
| tile colour | moved ΔE₀₀ about 10–11 on average from the original photo; a tile can turn green or red |
| photo survives | structure correlation 0.98; the right photo was identified in 4,851 of 4,851 cells |
| at 3 px per cell | closer to the target (SSIM) than all six other tools tested: every target at 486 and 1,350 cells, 9 of 12 at 3,015 |
| at 1 px per cell | plain nearest-average colour wins, ΔE₀₀ 2.6–2.8 against 4.9–5.5, because that number is exactly what it minimises |
| memory | peak 1.3 GB with 4,254 photos, mostly the cost matrix (cells × tiles) |

If the photos must keep their own colours, this is the wrong tool.

## 5. Video: holding still

**Each frame keeps the previous frame's tiles and changes only cells whose picture changed.** A fresh assignment
per frame reshuffles about **20%** of cells on sensor noise alone **(log)**, which reads as boiling. Cell means
decide which cells moved; only those get new descriptions.

| gate | trigger **(source)** | effect |
|---|---|---|
| repaint | cell mean moved more than **2.0** per channel | redo the colour transfer, **same tile** |
| re-evaluate | drift since the last full look passed **6.0** | the cell may swap tiles |
| safety valve | brightness moved more than **8.0** since the tile was chosen | swap allowed; catches slow fades |

**Swaps are rare on purpose.** Of 7,617 real swaps, **59% improved the picture by less than 1 dE** [10] **(log)**.
So a swap must cut the cell's cost **below 0.90** of the current tile's, and the search is greedy: an exact
re-solve raised churn **+4% to +218%** for under **0.2 dE** **(log)**.

The swap uses the raw feature distance plus an Oklab term on *every* tile, with no saliency weight. **Saliency is
frozen at the keyframe** until the next cut, because per-frame weights make the picture breathe. So a false face
hit (2 of 15 frames in one faceless clip **(log)**) lasts the whole scene.

**Scene cuts** reset everything. ffmpeg's `scdet` (`threshold=0`) scores every frame 0–100, and **10 or more** is a
cut, ffmpeg's default; real cuts score 10.7 and up, pans peak near 7.6 **(source)**. If this pre-pass fails, the
run stops, because "no scores" looks like "no cuts". The scale matters: see "Missed scene cuts" in §9.

Time per frame depends on the clip and the pool: a clip where everything moves took 1.7× as long per frame as live
footage, and 8,000 photos 1.27× as long as 600 **(log)**.

**Files: ffmpeg does all the container work; the code sees only raw frames, and frames in equal frames out.** If only the
audio copy fails, the video is kept as `name_video_only.mp4` and the run ends with an error.

<details>
<summary>Frame rate, formats, output and audio</summary>

| | what happens |
|---|---|
| frame rate | fixed once by the `fps` filter at decode, then frames pass through untouched; the scene pre-pass uses the same `fps` first |
| banned | `-shortest`, `mpdecimate`, `tblend`, `minterpolate`: anything that adds, drops or blends frames (`-shortest` once dropped 2 of 180 **(log)**) |
| pixels | `bgr24` requested directly (`rgb24` risks a red/blue swap) |
| output | x264, CRF 17, `yuv420p`, preset medium **(source)**; written as `name.part.mp4`, renamed when done |
| audio | stream-copied from the source |
| encoder | own thread, 32-frame queue; deepening it from 6 to 32 cut about 13% **(log)** |
| frame count | the container's count is an estimate; the decoder decides, and the run notes any difference |
| ffmpeg | the system copy, else `imageio-ffmpeg` if installed; still images need none |

</details>

## 6. The finishing pass

**A fixed chain runs last, one frame behind on its own thread so it overlaps the next render:** gamma and shadow
lift, high-frequency lift, vibrance, saturation, contrast, colour harmony, unsharp.

* **Per-pixel stages** (vibrance, saturation, contrast, harmony) use a pixel's colour and frame statistics held
  between keyframes (or saturation pulses). Video bakes them into a **128³ lookup table** per keyframe. A still
  colours directly: 128³ is 2.10 million points, a 1080p frame 2.07 million pixels, so that is cheaper and exact.
* **Spatial stages** (high-frequency lift, unsharp) read neighbours and run every frame.

**Dirty band.** When few cells were repainted, only rows they can reach are redone: the blurs reach 7 and 2 px, so
**9 rows** around the change. The rest is copied byte for byte. The band keeps the full width (OpenCV's blurs vary
with row length); over **70%** of the frame, the whole frame is redone.

## 7. How good could this folder get?

**`--diagnose` computes the best colour match any arrangement of your photos could reach; usually the folder, not
the algorithm, is the limit.** Cells' Oklab means are the *demand*, every tile is *supply with capacity one*, and
the assignment is solved on colour alone, a partial optimal transport problem [11].

| part | meaning | fix |
|---|---|---|
| colour gap | nearest-neighbour distance, ignoring capacity | the folder lacks this colour: add *different* photos |
| stock gap | the rest | the colour exists but ran out: add *more* like the ones that work |

It also splits the frame into four brightness bands and flags any with too little supply. In one run the bound was
**12.52** against an actual **13.52 (log)**. The bound falls steadily from 1.0× to 15.2× tiles per cell with no
knee **(log)**: more photos always help, by less each time.

## 8. Same input, same frames

**The same input gives the same frames on every run, so there is no thread option** (also on 1, 4 and 12 cores **(log)**).

* On import, OpenCV is set to one thread, and BLAS and OpenMP default to one via environment variables (left alone
  if you set them): BLAS is not bit-reproducible across thread counts. Face detection lifts the pin for its call.
* The `.mp4` repeated byte for byte on one machine **(log)**, but libx264 sizes its threads to the cores, so another
  machine may encode the same frames into different bytes.

<details>
<summary>The tests that hold this</summary>

* `tests/golden.py`: placement and the finished frame bit for bit (plus an SSIM floor of 0.999), a coarse palette
  check, and an 8-frame video run twice. Its clip is committed, because libx264 output depends on the core count.
* `tests/panel.py`: five motions (pan, sub-pixel shift, light drift, sensor noise, fade) through a plain full-frame
  path and the real engine, which must agree bit for bit. It caught a data race that six clean runs missed.

</details>

## 9. What did not work, and what broke

Ideas rejected and silent failures fixed, so nobody repeats them. Those explained above are not repeated.

<details>
<summary>Rejected or removed (7)</summary>

| tried | result |
|---|---|
| Scene cuts from our own churn statistics | Fires on pans. `scdet` looks at the source instead. |
| Reusing a tile up to 3× in the background | Not adopted: a tile would fill several cells, breaking the use-once rule. With no cap, 94% of the background was one tile **(log)**. |
| `--calm` option for steadier video | Removed: swaps fell only 2.4% on pans and 7.4% on drift; on fades it did nothing or added swaps **(log)**. |
| CLAHE | Removed: it hurt quality, mostly in dark backgrounds; dropping it also cut the dirty-band reach from ~200 px to 9 **(log)**. |
| 256³ colour table | Rejected: 13× the build time at every cut; 128³ is already within 0.8 levels on average **(log)**. |
| Per-cell `cv2.Sobel` | Rejected: made threaded video 23% slower **(log)**. |
| One shared scratch buffer in the finishing pass | A data race, since the pass runs on its own thread. Buffers are thread-local. |

</details>

<details>
<summary>Silent failures found and fixed (8)</summary>

| bug | cause and fix |
|---|---|
| Video one frame behind its audio **(log)** | The rate was converted twice, by the `fps` filter and again on the decoder's output. Frame 0 was doubled and every later frame ran 42 ms late. That output now passes frames through. |
| Empty file reported as "wrote out.mp4" **(source)** | Some containers (most webm) store the frame rate as `0/0`, and `fps=0` decoded nothing without an error. Now it falls back to `r_frame_rate`, else stops with a message. |
| Missed scene cuts **(source)** | A threshold of 35, taken from a 0–1 scale and read as 0–100, missed 46 of 47 cuts. The code keeps it as `SCENE_SCORE = 0.10` and multiplies by 100 where it compares. |
| Tiles taken from other cells when tiles = cells, video only **(log)** | With no free tile, `argmin` over an all-infinite row returned 0 instead of failing. A free-tile check now runs first. |
| OpenCV 5.0 would have crashed every new install **(log)** | 5.0 removed `CascadeClassifier`, which face detection needs. Caught before release; the dependency is `opencv-python>=4.6,<5`. |
| Still image "wrote" with exit 0, nothing on disk **(log)** | The result of `cv2.imwrite` was ignored. It is checked now. |
| Raw traceback with 6 photos or fewer **(log)** | The shape re-check asked for 25 candidates from a smaller pool. It is clamped now. |
| Photo credits missing from the wheel **(log)** | The bundled photos are CC BY; their `ATTRIBUTION.md` now ships inside the package. |

</details>

## References

Each entry was checked against a primary or author-controlled source.

1. Robert Silvers. *Photomosaics*. Henry Holt and Company, 1997. Also US Patent 6,137,498, 2000: [patents.google.com](https://patents.google.com/patent/US6137498A/en)
2. Björn Ottosson. A perceptual color space for image processing. 2020: [bottosson.github.io](https://bottosson.github.io/posts/oklab/)
3. Ojala, Pietikäinen, Mäenpää. Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. *IEEE TPAMI* 24(7), 2002. [doi:10.1109/TPAMI.2002.1017623](https://doi.org/10.1109/TPAMI.2002.1017623)
4. Viola, Jones. Rapid Object Detection using a Boosted Cascade of Simple Features. *CVPR* 2001. [doi:10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)
5. Kuhn. The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly* 2, 1955. [doi:10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109)
6. Jonker, Volgenant. A shortest augmenting path algorithm for dense and sparse linear assignment problems. *Computing* 38, 1987. [doi:10.1007/BF02278710](https://doi.org/10.1007/BF02278710)
7. Wang, Bovik, Sheikh, Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE TIP* 13(4), 2004. [doi:10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)
8. Reinhard, Ashikhmin, Gooch, Shirley. Color Transfer between Images. *IEEE CG&A* 21(5), 2001. [doi:10.1109/38.946629](https://doi.org/10.1109/38.946629)
9. Pitié, Kokaram. The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer. *CVMP* 2007. [doi:10.1049/cp:20070055](https://doi.org/10.1049/cp:20070055)
10. Sharma, Wu, Dalal. The CIEDE2000 Color-Difference Formula. *Color Research & Application* 30(1), 2005. [doi:10.1002/col.20070](https://doi.org/10.1002/col.20070)
11. Peyré, Cuturi. Computational Optimal Transport. *Foundations and Trends in Machine Learning* 11(5–6), 2019. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)

<details>
<summary>BibTeX</summary>

```bibtex
% [1] the form itself
@book{silvers1997photomosaics,
    title     = {Photomosaics},
    author    = {Robert Silvers},
    year      = {1997},
    publisher = {Henry Holt and Company},
}
@misc{silvers2000patent,
    title  = {Digital composition of a mosaic image},
    author = {Robert S. Silvers},
    year   = {2000},
    note   = {US Patent 6,137,498; filed 1997, granted 24 Oct 2000},
    url    = {https://patents.google.com/patent/US6137498A/en},
}

% [2] the colour space (S2)
@misc{ottosson2020oklab,
    title  = {A perceptual color space for image processing},
    author = {Bj\"orn Ottosson},
    year   = {2020},
    url    = {https://bottosson.github.io/posts/oklab/},
}

% [3] the texture half of the cell descriptor (S3.3)
@article{ojala2002lbp,
    title   = {Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns},
    author  = {Timo Ojala and Matti Pietik\"ainen and Topi M\"aenp\"a\"a},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume  = {24},
    number  = {7},
    pages   = {971--987},
    year    = {2002},
    doi     = {10.1109/TPAMI.2002.1017623},
}

% [4] face detection behind the saliency weight (S3.3)
@inproceedings{viola2001rapid,
    title     = {Rapid Object Detection using a Boosted Cascade of Simple Features},
    author    = {Paul Viola and Michael Jones},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    volume    = {1},
    pages     = {511--518},
    year      = {2001},
    doi       = {10.1109/CVPR.2001.990517},
}

% [5] the assignment problem (S3.4)
@article{kuhn1955hungarian,
    title   = {The Hungarian method for the assignment problem},
    author  = {Harold W. Kuhn},
    journal = {Naval Research Logistics Quarterly},
    volume  = {2},
    number  = {1--2},
    pages   = {83--97},
    year    = {1955},
    doi     = {10.1002/nav.3800020109},
}

% [6] how it is solved in practice (S3.4)
@article{jonker1987shortest,
    title   = {A shortest augmenting path algorithm for dense and sparse linear assignment problems},
    author  = {Roy Jonker and Anton Volgenant},
    journal = {Computing},
    volume  = {38},
    pages   = {325--340},
    year    = {1987},
    doi     = {10.1007/BF02278710},
}

% [7] the structure score in the second look (S3.5)
@article{wang2004ssim,
    title   = {Image Quality Assessment: From Error Visibility to Structural Similarity},
    author  = {Zhou Wang and Alan C. Bovik and Hamid R. Sheikh and Eero P. Simoncelli},
    journal = {IEEE Transactions on Image Processing},
    volume  = {13},
    number  = {4},
    pages   = {600--612},
    year    = {2004},
    doi     = {10.1109/TIP.2003.819861},
}

% [8] the colour transfer photomosaic-video does not use (S4)
@article{reinhard2001color,
    title   = {Color Transfer between Images},
    author  = {Erik Reinhard and Michael Ashikhmin and Bruce Gooch and Peter Shirley},
    journal = {IEEE Computer Graphics and Applications},
    volume  = {21},
    number  = {5},
    pages   = {34--41},
    year    = {2001},
    doi     = {10.1109/38.946629},
}

% [9] the colour transfer it does use (S4)
@inproceedings{pitie2007linear,
    title     = {The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer},
    author    = {Fran\c{c}ois Piti\'e and Anil Kokaram},
    booktitle = {4th European Conference on Visual Media Production (CVMP)},
    publisher = {IET},
    year      = {2007},
    doi       = {10.1049/cp:20070055},
    note      = {The word "linear" does appear twice in the published title.},
}

% [10] the colour difference the video gates are measured in (S5)
@article{sharma2005ciede2000,
    title   = {The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations},
    author  = {Gaurav Sharma and Wencheng Wu and Edul N. Dalal},
    journal = {Color Research \& Application},
    volume  = {30},
    number  = {1},
    pages   = {21--30},
    year    = {2005},
    doi     = {10.1002/col.20070},
}

% [11] the framing of the pool diagnostic (S7)
@article{peyre2019ot,
    title   = {Computational Optimal Transport},
    author  = {Gabriel Peyr\'e and Marco Cuturi},
    journal = {Foundations and Trends in Machine Learning},
    volume  = {11},
    number  = {5--6},
    pages   = {355--607},
    year    = {2019},
    eprint  = {1803.00567},
    archivePrefix = {arXiv},
    url     = {https://arxiv.org/abs/1803.00567},
}
```

</details>
