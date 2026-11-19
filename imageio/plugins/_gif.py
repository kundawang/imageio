# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Delta-frame (sub-rectangle) encoding for animated GIFs.

The regular GIF writer hands every frame to Pillow as a full canvas image.
Pillow may or may not compress the unchanged areas between consecutive
frames, and it drops frames that are identical to their predecessor
(folding their delay into the previous frame).

The writer in this module instead compares each *composited* input frame
with the previous one and stores only the bounding box of the pixels that
actually change.  The first frame always covers the whole canvas.  Every
input frame is kept, including identical ones (they become a tiny
keep-alive tile), so per-frame delays, the loop count and transparency are
all preserved.

Each tile is quantized with its own local color table.  This costs a few
hundred bytes per frame, but guarantees correct colors even when a frame
introduces colors that were absent from earlier frames, which a single
shared (global) palette cannot represent once more than 256 colors occur.

Disposal is chosen per frame:

* method 1 ("do not dispose") is the default.  A tile only paints the
  changed pixels; the unchanged ones are stored as transparent, so the
  previous canvas shows through.
* method 2 ("restore to background") is used when pixels that were opaque
  become transparent.  The affected region is cleared first, then fully
  repainted, which is the only GIF mechanism that can erase content.

The composited result at every playback instant is therefore identical to
the full-frame input.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import GifImagePlugin, Image


def _as_rgba(frame: Image.Image) -> np.ndarray:
    """Return a frame as an HxWx4 uint8 array for pixel comparison."""

    if frame.mode == "RGBA":
        return np.asarray(frame)
    # Comparing palette indices is meaningless once the palette changes,
    # so always compare the composited RGBA pixels.
    return np.asarray(frame.convert("RGBA"))


def _changed_mask(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Boolean mask of visible per-pixel differences.

    A difference that is fully transparent in both frames carries no
    visible information and is ignored.
    """

    changed = np.any(previous != current, axis=-1)
    invisible = (previous[..., 3] == 0) & (current[..., 3] == 0)
    return changed & ~invisible


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the True pixels, or ``None`` if the mask is empty."""

    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1


def _normalize_duration(duration: Any, n_frames: int) -> List[float]:
    """Expand the user's ``duration`` argument into one value per frame."""

    if duration is None:
        # Pillow's own default for GIF
        return [100.0] * n_frames
    if isinstance(duration, (list, tuple, np.ndarray)):
        values = [float(d) for d in duration]
        if len(values) < n_frames:
            # mirror Pillow: repeat the last known duration
            values.extend([values[-1]] * (n_frames - len(values)))
        return values[:n_frames]
    return [float(duration)] * n_frames


def _plan_frames(
    rgba_frames: Sequence[np.ndarray],
) -> Tuple[List[Optional[Tuple[int, int, int, int]]], List[int], List[bool]]:
    """Compute each frame's update box, disposal and erase flag.

    Returns a list of boxes (``None`` marks a frame identical to its
    predecessor), the disposal method that applies *after* each frame, and
    a per-frame flag telling whether opaque pixels become transparent.
    """

    n_frames = len(rgba_frames)
    boxes: List[Optional[Tuple[int, int, int, int]]] = []
    erases: List[bool] = []

    height, width = rgba_frames[0].shape[:2]
    boxes.append((0, 0, width, height))
    erases.append(False)

    for index in range(1, n_frames):
        previous = rgba_frames[index - 1]
        current = rgba_frames[index]
        mask = _changed_mask(previous, current)
        box = _mask_bbox(mask)
        boxes.append(box)
        if box is None:
            erases.append(False)
        else:
            becomes_transparent = (
                (previous[..., 3] == 255) & (current[..., 3] == 0) & mask
            ).any()
            erases.append(bool(becomes_transparent))

    disposal = [1] * n_frames
    for index in range(1, n_frames):
        if erases[index] and boxes[index] is not None:
            # Clear the previous frame's region before painting this one.
            disposal[index - 1] = 2
            # Make sure the current tile fully covers the cleared region,
            # otherwise previously visible pixels would stay erased.
            previous_box = boxes[index - 1]
            if previous_box is not None:
                current_box = boxes[index]
                boxes[index] = (
                    min(current_box[0], previous_box[0]),
                    min(current_box[1], previous_box[1]),
                    max(current_box[2], previous_box[2]),
                    max(current_box[3], previous_box[3]),
                )

    return boxes, disposal, erases


def write_gif_with_subrectangles(
    file_handle: Any,
    frames: Sequence[Image.Image],
    *,
    duration: Any = None,
    loop: Optional[int] = None,
    optimize: bool = True,
    **ignored_kwargs: Any,
) -> None:
    """Write an animated GIF using minimal per-frame update regions.

    Parameters
    ----------
    file_handle
        A writable, seekable binary file object that receives the GIF.
    frames
        A sequence of full-canvas Pillow images of identical size.
    duration
        Per-frame delay in milliseconds; a scalar or a sequence with one
        value per frame.
    loop
        Number of times the animation should loop.  ``0`` (the default
        when ``None``) means loop forever.
    optimize
        Forwarded to Pillow's palette optimization for every tile.
    ignored_kwargs
        Other GIF save arguments.  ``disposal`` is intentionally managed
        here and any user value is ignored.
    """

    if not frames:
        raise ValueError("Cannot write a GIF without any frames.")

    canvas_size = frames[0].size
    for index, frame in enumerate(frames[1:], start=1):
        if frame.size != canvas_size:
            raise ValueError(
                "`subrectangles=True` requires all frames to share the same "
                f"size, but frame 0 has size {canvas_size} and frame "
                f"{index} has size {frame.size}."
            )

    durations = _normalize_duration(duration, len(frames))
    if loop is None:
        loop = 0

    rgba_frames = [_as_rgba(frame) for frame in frames]
    boxes, disposal, erases = _plan_frames(rgba_frames)

    for index in range(len(frames)):
        current = rgba_frames[index]
        box = boxes[index]

        if index == 0:
            tile_image = Image.fromarray(current, "RGBA")
        elif box is None:
            # No visible change: keep the frame and its delay with a
            # transparent 1x1 keep-alive tile instead of dropping it.
            tile_image = Image.fromarray(np.zeros((1, 1, 4), np.uint8), "RGBA")
            box = (0, 0, 1, 1)
        else:
            x0, y0, x1, y1 = box
            tile = current[y0:y1, x0:x1].copy()
            if not erases[index] and disposal[index - 1] != 2:
                # Store unchanged pixels as transparent so only the changed
                # pixels are painted on top of the retained canvas.
                previous_tile = rgba_frames[index - 1][y0:y1, x0:x1]
                unchanged = ~np.any(previous_tile != tile, axis=-1)
                tile[unchanged] = (0, 0, 0, 0)
            tile_image = Image.fromarray(tile, "RGBA")

        tile = GifImagePlugin._normalize_mode(tile_image.copy())
        params = {
            "duration": durations[index],
            "loop": loop,
            "optimize": optimize,
            "disposal": disposal[index],
        }
        if index > 0:
            # Each tile carries its own palette, so new colors introduced
            # by a frame are represented correctly without corrupting
            # earlier frames.
            params["include_color_table"] = True
        # ``_normalize_mode`` assigns a palette index for transparent pixels;
        # propagate it so the graphics control extension marks them.
        if "transparency" in tile.info:
            params["transparency"] = tile.info["transparency"]

        tile.encoderinfo = params
        if index == 0:
            for chunk in GifImagePlugin._get_global_header(tile, params):
                file_handle.write(chunk)
        GifImagePlugin._write_frame_data(file_handle, tile, box[:2], params)

    file_handle.write(b";")
