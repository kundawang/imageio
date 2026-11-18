# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read/Write QOI images.

Backend: built-in pure-Python encoder/decoder implementing
`QOI v1.0 <https://qoiformat.org/qoi-spec.pdf>`_.

QOI (Quite OK Image format) is a fast, lossless raster format that stores
8-bit RGB or RGBA images. The format compresses images with a small set of
byte-aligned opcodes (index, diff, luma, run and full RGB/RGBA chunks) and is
streamable in principle; this implementation buffers the whole image.

This plugin exposes the QOI v1.0 header fields through ``metadata()`` and
``properties()``. In particular, the ``colorspace`` byte (``0`` for sRGB with
linear alpha, ``1`` for all channels linear) is read back verbatim.

Parameters
----------
colorspace : int
    Only used while writing. Value of the QOI header's colorspace byte.
    ``0`` (default) means sRGB with linear alpha, ``1`` means all channels
    are linear.

Methods
-------
.. note::
    Check the respective function for a list of supported kwargs and detailed
    documentation.

.. autosummary::
    :toctree:

    QoiPlugin.read
    QoiPlugin.iter
    QoiPlugin.write
    QoiPlugin.properties
    QoiPlugin.metadata

"""

from io import BytesIO
from typing import Any, Dict, Iterator, Optional, cast

import numpy as np

from ..core.request import URI_BYTES, InitializationError, IOMode, Request
from ..core.v3_plugin_api import ImageProperties, PluginV3
from ..typing import ArrayLike

# QOI v1.0 constants, see https://qoiformat.org/qoi-spec.pdf
QOI_MAGIC = b"qoif"
QOI_HEADER_SIZE = 14
QOI_END_MARKER = b"\x00" * 7 + b"\x01"

# The reference implementation refuses images with 400 million pixels or more
# (width * height). We keep the same limit so that the pixel count cannot
# overflow the header's uint32 range and decoding stays bounded.
QOI_PIXELS_MAX = 400_000_000

QOI_OP_INDEX = 0x00  # 00xxxxxx
QOI_OP_DIFF = 0x40  # 01xxxxxx
QOI_OP_LUMA = 0x80  # 10xxxxxx
QOI_OP_RUN = 0xC0  # 11xxxxxx
QOI_OP_RGB = 0xFE  # 11111110
QOI_OP_RGBA = 0xFF  # 11111111
QOI_MASK_2 = 0xC0

# Run lengths 63 and 64 are reserved by the RGB/RGBA tags, so a QOI_OP_RUN
# chunk may only carry values 0..61 (stored run length minus one).
QOI_RUN_MAX = 62


class QoiError(ValueError):
    """Raised when QOI data cannot be decoded or an image cannot be encoded."""


def _hash_index(r: int, g: int, b: int, a: int) -> int:
    return (r * 3 + g * 5 + b * 7 + a * 11) & 63


def _to_int8(value: int) -> int:
    """Truncate a Python int to the range of a C ``signed char``.

    The reference encoder computes channel differences in ``signed char``
    arithmetic, so values wrap modulo 256 into the range -128..127 before the
    diff/luma eligibility checks. Replicating this keeps our output
    byte-identical to the official encoder for large channel jumps such as
    1 -> 255, whose difference is -2 after truncation rather than 254.
    """

    return ((value + 128) & 0xFF) - 128


def _read_header(data: bytes) -> Dict[str, int]:
    """Parse and validate the 14-byte QOI header."""

    if len(data) < QOI_HEADER_SIZE + len(QOI_END_MARKER):
        raise QoiError(
            "QOI data is too short: it must contain a 14-byte header, at "
            "least one chunk and an 8-byte end marker."
        )

    magic = data[0:4]
    if magic != QOI_MAGIC:
        raise InitializationError(
            f"bad QOI magic {magic!r}; expected {QOI_MAGIC!r} ('qoif')."
        )

    width = int.from_bytes(data[4:8], "big")
    height = int.from_bytes(data[8:12], "big")
    channels = data[12]
    colorspace = data[13]

    if width == 0 or height == 0:
        raise QoiError(
            f"QOI image dimensions must be non-zero (got width={width}, "
            f"height={height})."
        )
    if channels not in (3, 4):
        raise QoiError(
            f"invalid QOI channels byte {channels}; expected 3 (RGB) or 4 " "(RGBA)."
        )
    if colorspace not in (0, 1):
        raise QoiError(
            f"invalid QOI colorspace byte {colorspace}; expected 0 (sRGB "
            "with linear alpha) or 1 (all channels linear)."
        )
    # Matches the reference implementation's overflow guard.
    if height >= QOI_PIXELS_MAX // width:
        raise QoiError(
            f"QOI image too large: {width}x{height} pixels exceeds the limit "
            f"of {QOI_PIXELS_MAX} pixels."
        )

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "colorspace": colorspace,
    }


def _decode_qoi(data: bytes) -> Dict[str, Any]:
    """Decode a QOI v1.0 byte string into pixels and header metadata."""

    header = _read_header(data)
    width = header["width"]
    height = header["height"]
    channels = header["channels"]

    n_pixels = width * height
    # chunks live between the header and the 8-byte end marker
    chunks_len = len(data) - len(QOI_END_MARKER)
    if data[chunks_len:] != QOI_END_MARKER:
        raise QoiError(
            "invalid QOI end marker: the stream must end with seven 0x00 "
            "bytes followed by a single 0x01 byte."
        )

    pixels = bytearray(n_pixels * channels)
    index = [(0, 0, 0, 0)] * 64

    r = g = b = 0
    a = 255
    run = 0
    pos = QOI_HEADER_SIZE
    out_pos = 0

    for _ in range(n_pixels):
        if run > 0:
            run -= 1
        else:
            if pos >= chunks_len:
                raise QoiError(
                    "truncated QOI stream: pixel data ended before all "
                    f"{n_pixels} pixels were decoded."
                )

            b1 = data[pos]
            pos += 1

            if b1 == QOI_OP_RGB:
                if pos + 3 > chunks_len:
                    raise QoiError(
                        "truncated QOI_OP_RGB chunk: expected 3 color bytes."
                    )
                r, g, b = data[pos], data[pos + 1], data[pos + 2]
                pos += 3
            elif b1 == QOI_OP_RGBA:
                if pos + 4 > chunks_len:
                    raise QoiError(
                        "truncated QOI_OP_RGBA chunk: expected 4 color bytes."
                    )
                r, g, b, a = data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
                pos += 4
            elif (b1 & QOI_MASK_2) == QOI_OP_INDEX:
                r, g, b, a = index[b1]
            elif (b1 & QOI_MASK_2) == QOI_OP_DIFF:
                r = (r + ((b1 >> 4) & 0x03) - 2) & 0xFF
                g = (g + ((b1 >> 2) & 0x03) - 2) & 0xFF
                b = (b + (b1 & 0x03) - 2) & 0xFF
            elif (b1 & QOI_MASK_2) == QOI_OP_LUMA:
                if pos >= chunks_len:
                    raise QoiError(
                        "truncated QOI_OP_LUMA chunk: expected a second byte."
                    )
                b2 = data[pos]
                pos += 1
                vg = (b1 & 0x3F) - 32
                r = (r + vg - 8 + ((b2 >> 4) & 0x0F)) & 0xFF
                g = (g + vg) & 0xFF
                b = (b + vg - 8 + (b2 & 0x0F)) & 0xFF
            else:  # QOI_OP_RUN
                # Tags 0xFE/0xFF are caught by the RGB/RGBA checks above, so
                # the reachable stored values are 0..61 (run length 1..62).
                # The reference decoder applies no further validation here.
                run = b1 & 0x3F

            index[_hash_index(r, g, b, a)] = (r, g, b, a)

        pixels[out_pos] = r
        pixels[out_pos + 1] = g
        pixels[out_pos + 2] = b
        if channels == 4:
            pixels[out_pos + 3] = a
        out_pos += channels

    if pos != chunks_len:
        raise QoiError(
            f"unexpected trailing data: {chunks_len - pos} byte(s) remain "
            "after the last pixel."
        )

    shape = (height, width, channels)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(shape).copy()
    header["image"] = image
    return header


def _encode_qoi(image: np.ndarray, *, colorspace: int = 0) -> bytes:
    """Encode an HxWx3 or HxWx4 uint8 image as a QOI v1.0 byte string."""

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise QoiError(
            "QOI can only encode 2D RGB/RGBA images with shape (height, "
            f"width, 3|4); got shape {image.shape}."
        )
    if image.dtype != np.uint8:
        raise QoiError(f"QOI can only encode uint8 images; got dtype {image.dtype}.")
    if colorspace not in (0, 1):
        raise QoiError(
            f"invalid QOI colorspace {colorspace}; expected 0 (sRGB with "
            "linear alpha) or 1 (all channels linear)."
        )

    height, width, channels = image.shape
    n_pixels = width * height
    if n_pixels >= QOI_PIXELS_MAX:
        raise QoiError(
            f"QOI image too large: {width}x{height} pixels exceeds the limit "
            f"of {QOI_PIXELS_MAX} pixels."
        )

    # Worst case: 5 bytes per pixel plus header and end marker.
    out = bytearray(n_pixels * (channels + 1) + QOI_HEADER_SIZE + 8)
    out[0:4] = QOI_MAGIC
    out[4:8] = width.to_bytes(4, "big")
    out[8:12] = height.to_bytes(4, "big")
    out[12] = channels
    out[13] = colorspace
    pos = QOI_HEADER_SIZE

    index = [(0, 0, 0, 0)] * 64
    prev_r = prev_g = prev_b = 0
    prev_a = 255
    run = 0

    flat = np.ascontiguousarray(image).reshape(-1)
    # The view keeps us from allocating a (potentially huge) bytes copy.
    data = memoryview(flat)
    byte_pos = 0

    for pixel_index in range(n_pixels):
        r, g, b = data[byte_pos], data[byte_pos + 1], data[byte_pos + 2]
        a = data[byte_pos + 3] if channels == 4 else 255
        byte_pos += channels

        if r == prev_r and g == prev_g and b == prev_b and a == prev_a:
            run += 1
            if run == QOI_RUN_MAX or pixel_index == n_pixels - 1:
                out[pos] = QOI_OP_RUN | (run - 1)
                pos += 1
                run = 0
        else:
            if run > 0:
                out[pos] = QOI_OP_RUN | (run - 1)
                pos += 1
                run = 0

            index_pos = _hash_index(r, g, b, a)
            if index[index_pos] == (r, g, b, a):
                out[pos] = QOI_OP_INDEX | index_pos
                pos += 1
            else:
                index[index_pos] = (r, g, b, a)

                if a == prev_a:
                    # Mirror the reference encoder's signed char arithmetic;
                    # see _to_int8 for details.
                    vr = _to_int8(r - prev_r)
                    vg = _to_int8(g - prev_g)
                    vb = _to_int8(b - prev_b)
                    vg_r = _to_int8(vr - vg)
                    vg_b = _to_int8(vb - vg)

                    if -3 < vr < 2 and -3 < vg < 2 and -3 < vb < 2:
                        out[pos] = (
                            QOI_OP_DIFF | (vr + 2) << 4 | (vg + 2) << 2 | (vb + 2)
                        )
                        pos += 1
                    elif -9 < vg_r < 8 and -33 < vg < 32 and -9 < vg_b < 8:
                        out[pos] = QOI_OP_LUMA | (vg + 32)
                        out[pos + 1] = ((vg_r + 8) << 4) | (vg_b + 8)
                        pos += 2
                    else:
                        out[pos] = QOI_OP_RGB
                        out[pos + 1] = r
                        out[pos + 2] = g
                        out[pos + 3] = b
                        pos += 4
                else:
                    out[pos] = QOI_OP_RGBA
                    out[pos + 1] = r
                    out[pos + 2] = g
                    out[pos + 3] = b
                    out[pos + 4] = a
                    pos += 5

        prev_r, prev_g, prev_b, prev_a = r, g, b, a

    out[pos : pos + 8] = QOI_END_MARKER
    pos += 8
    return bytes(out[:pos])


class QoiPlugin(PluginV3):
    """Support for the QOI (Quite OK Image) v1.0 format.

    QOI stores a single 8-bit RGB or RGBA image per file. Reading returns an
    ``(height, width, 3|4)`` uint8 array. Writing accepts the same kind of
    array.

    Parameters
    ----------
    request : iio.Request
        A request object that represents the user's intent.
    colorspace : int
        Write-only. Value of the QOI colorspace header byte: ``0`` (default)
        for sRGB with linear alpha or ``1`` for all-linear channels.

    """

    def __init__(self, request: Request, *, colorspace: int = 0) -> None:
        super().__init__(request)
        self._colorspace = colorspace
        self._file = self.request.get_file()
        self._header: Optional[Dict[str, int]] = None

        if request.mode.io_mode == IOMode.read:
            # Read the whole resource once; QOI is not decoded incrementally.
            self._data = self._file.read()
            if not isinstance(self._data, bytes):
                self._data = bytes(self._data)

            if len(self._data) < QOI_HEADER_SIZE:
                raise InitializationError(
                    "the resource is too short to contain a QOI header."
                )
            try:
                self._header = _read_header(self._data)
            except InitializationError:
                # bad magic: this plugin does not recognize the resource, so
                # plugin auto-detection should try the next backend.
                raise
            except QoiError:
                # Recognizable QOI data with an invalid header; keep the
                # detailed error for read()/properties()/metadata().
                self._header = None
        else:
            self._data = None
            self._header = None

    # ---------------------
    # Standard V3 Interface
    # ---------------------

    def read(self, *, index: int = 0) -> np.ndarray:
        """Read the image.

        Parameters
        ----------
        index : int
            QOI files hold exactly one image. Only ``0`` (or ``...``/``None``,
            which are treated as ``0``) are accepted.

        Returns
        -------
        ndimage : np.ndarray
            The decoded image as an ``(height, width, channels)`` uint8 array.

        """

        if index not in (0, None, Ellipsis):
            raise IndexError(
                f"Index `{index}` is out of bounds for a QOI file, which "
                "contains exactly one image."
            )

        decoded = _decode_qoi(self._data)
        return decoded["image"]

    def iter(self) -> Iterator[np.ndarray]:
        """Yield the single image stored in the file.

        Yields
        ------
        ndimage : np.ndarray
            A decoded ndimage.

        """

        yield self.read(index=0)

    def write(
        self, ndimage: ArrayLike, *, colorspace: Optional[int] = None
    ) -> Optional[bytes]:
        """Save an RGB/RGBA uint8 image as QOI.

        Parameters
        ----------
        ndimage : ArrayLike
            The image to encode. Must be an ``(height, width, 3|4)`` uint8
            array; QOI does not support batches.
        colorspace : int
            Value of the QOI colorspace header byte: ``0`` (default) for sRGB
            with linear alpha or ``1`` for all-linear channels. Overrides the
            value given to the constructor.

        Returns
        -------
        encoded_image : bytes or None
            If the ImageResource is ``"<bytes>"``, the encoded byte string is
            returned; otherwise ``None``.

        """

        if colorspace is None:
            colorspace = self._colorspace

        image = np.asarray(ndimage)
        encoded = _encode_qoi(image, colorspace=colorspace)
        self._file.write(encoded)

        if self._request._uri_type == URI_BYTES:
            file = cast(BytesIO, self._request.get_file())
            return file.getvalue()
        return None

    def properties(self, index: int = 0) -> ImageProperties:
        """Standardized ndimage metadata.

        Parameters
        ----------
        index : int
            Accepted for API compatibility; QOI files hold one image.

        Returns
        -------
        properties : ImageProperties
            A dataclass with the image's shape and dtype.

        """

        if index not in (0, None, Ellipsis):
            raise IndexError(
                f"Index `{index}` is out of bounds for a QOI file, which "
                "contains exactly one image."
            )

        if self._header is None:
            # Re-run the header parser to surface the descriptive error.
            self._header = _read_header(self._data)
        header = self._header

        return ImageProperties(
            shape=(header["height"], header["width"], header["channels"]),
            dtype=np.dtype(np.uint8),
            n_images=1,
            is_batch=False,
        )

    def metadata(self, index: int = 0, exclude_applied: bool = True) -> Dict[str, Any]:
        """Format-specific QOI metadata.

        Parameters
        ----------
        index : int
            Accepted for API compatibility; QOI files hold one image.
        exclude_applied : bool
            Ignored. The header fields describe the encoded stream rather than
            transformations applied to the returned ndimage.

        Returns
        -------
        metadata : dict
            A dictionary containing the raw QOI v1.0 header fields:
            ``width``, ``height``, ``channels`` (3 for RGB, 4 for RGBA) and
            ``colorspace`` (0 for sRGB/linear alpha, 1 for all-linear).

        """

        if index not in (0, None, Ellipsis):
            raise IndexError(
                f"Index `{index}` is out of bounds for a QOI file, which "
                "contains exactly one image."
            )
        if self._header is None:
            self._header = _read_header(self._data)

        return dict(self._header)

    def close(self) -> None:
        super().close()
