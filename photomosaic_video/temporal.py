"""Frame-to-frame consistency: the part that makes this a video tool."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mosaic import OKLAB_COST_WEIGHT, Pool, cell_features, cell_oklab, sq_dists

NDArray = np.ndarray

PAINT_THRESHOLD = 2.0  # mean |delta| per BGR channel to repaint a cell
BAND_T2 = 6.0  # cumulative delta before a cell may swap tiles
SWAP_GAIN = 0.90  # a swap must cut the cell's cost to this fraction
SAFETY_TV = 8.0  # luminance drift that always unlocks a swap


def cell_means(cells_bgr: NDArray) -> tuple[NDArray, NDArray]:
    """Every cell's mean colour and mean luminance, from one pass."""
    total = cells_bgr.sum(axis=1, dtype=np.int64).sum(axis=1)
    n = cells_bgr[0].size // 3
    return total / n, total.sum(axis=1) / (3 * n)


@dataclass
class TemporalState:
    """Everything carried from one frame to the next."""

    is_keyframe: bool = True
    assignment: NDArray | None = None
    saliency: NDArray | None = None
    canvas: NDArray | None = None
    ref_paint: NDArray | None = None  # cell colour when last painted
    ref_full: NDArray | None = None  # cell colour when last re-evaluated
    assign_luma: NDArray | None = None  # cell luminance when tile was chosen

    # -- keyframe ---------------------------------------------------------
    def reset(self, cells_bgr: NDArray, assignment: NDArray, saliency: NDArray) -> None:
        """Scene cut: drop all history. Carrying it across a cut is wrong."""
        self.assignment = assignment
        self.saliency = saliency
        self.ref_paint, self.assign_luma = cell_means(cells_bgr)
        self.ref_full = self.ref_paint.copy()
        self.is_keyframe = False

    # -- intermediate frame ----------------------------------------------
    def update(
        self, cells_bgr: NDArray, cells_lab: NDArray, pool: Pool
    ) -> tuple[NDArray, NDArray]:
        """Return (assignment, indices to repaint) for this frame."""
        now, luma = cell_means(cells_bgr)
        d_paint = np.abs(now - self.ref_paint).mean(axis=1)
        d_full = np.abs(now - self.ref_full).mean(axis=1)
        forced = np.abs(luma - self.assign_luma) > SAFETY_TV

        repaint = np.flatnonzero(d_paint > PAINT_THRESHOLD)
        may_swap = np.flatnonzero((d_full > BAND_T2) | forced)
        swapped = self._try_swaps(may_swap, cells_bgr, cells_lab, pool)

        self.ref_paint[repaint] = now[repaint]
        self.ref_full[may_swap] = now[may_swap]
        if swapped.size:
            self.assign_luma[swapped] = luma[swapped]
        # Paste = cells that moved OR whose tile changed. Repainting only the
        # cells that moved leaves swapped cells showing their old tile.
        return self.assignment, np.union1d(repaint, swapped)

    def _try_swaps(
        self,
        candidates: NDArray,
        cells_bgr: NDArray,
        cells_lab: NDArray,
        pool: Pool,
    ) -> NDArray:
        """Greedy 1:1 swap against unused tiles. Deliberately not optimal:
        the exact re-solve is worse on video (+4-218% churn) for a fidelity
        gain under 0.2 dE, so the greedy lag is doing useful smoothing."""
        if candidates.size == 0:
            return np.empty(0, np.int64)
        feats = cell_features(cells_lab[candidates])
        oklab = cell_oklab(cells_bgr[candidates])
        used = np.zeros(len(pool), bool)
        used[self.assignment] = True
        # (cells, tiles) cost. float64: float32 re-association can flip a tile.
        d = sq_dists(feats.astype(np.float64), pool.features64, pool.feat_sq)
        colour = np.sqrt(sq_dists(oklab, pool.oklab, pool.oklab_sq))
        dist = d + colour * OKLAB_COST_WEIGHT * d.mean()
        # A pool exactly the size of the grid leaves nothing spare, and an
        # all-infinite row makes argmin answer 0 rather than refuse -- taking
        # that answer would hand tile 0 to a second cell and break the 1:1.
        # Checked once: a swap frees one tile and claims another, so the number
        # in use never changes and this cannot become false inside the loop.
        if used.all():
            return np.empty(0, np.int64)
        wish = (dist + np.where(used, np.inf, 0.0)).argmin(axis=1)
        changed = []
        for k, i in enumerate(candidates):
            current = int(self.assignment[i])
            best = int(wish[k])
            if used[best]:
                best = int(np.argmin(np.where(used, np.inf, dist[k])))
            gain = dist[k, best] / (dist[k, current] + 1e-12)
            if gain >= SWAP_GAIN:
                continue
            used[current], used[best] = False, True
            self.assignment[i] = best
            changed.append(i)
        return np.asarray(changed, np.int64)
