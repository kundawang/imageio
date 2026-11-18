# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Tests for the built-in QOI (Quite OK Image) v1.0 plugin."""

import io
from itertools import product

import numpy as np
import pytest

import imageio.v3 as iio
from imageio.config import known_extensions, known_plugins
from imageio.core.request import InitializationError
from imageio.plugins import qoi
from imageio.plugins.qoi import QoiError, QoiPlugin

HEADER_SIZE = 14
END_MARKER = b"\x00" * 7 + b"\x01"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_header(width, height, channels=3, colorspace=0):
    return (
        b"qoif"
        + int(width).to_bytes(4, "big")
        + int(height).to_bytes(4, "big")
        + bytes((channels, colorspace))
    )


@pytest.fixture
def rgb_image():
    return np.array(
        [
            [[0, 0, 0], [255, 255, 255], [10, 128, 240]],
            [[1, 2, 3], [1, 2, 3], [200, 100, 50]],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def rgba_image():
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=(5, 7, 4), dtype=np.uint8)
    # force a couple of fully transparent and opaque pixels
    image[0, 0] = (0, 0, 0, 0)
    image[1, 1] = (255, 255, 255, 255)
    return image


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_plugin_registration():
    assert "qoi" in known_plugins
    assert known_plugins["qoi"].plugin_class is QoiPlugin
    assert ".qoi" in known_extensions
    assert "qoi" in known_extensions[".qoi"][0].priority


def test_imopen_selects_qoi(tmp_path):
    path = tmp_path / "image.qoi"
    iio.imwrite(path, np.zeros((2, 2, 3), np.uint8))
    with iio.imopen(path, "r") as plugin:
        assert isinstance(plugin, QoiPlugin)


# ---------------------------------------------------------------------------
# Roundtrips through the public imageio.v3 API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [3, 4])
def test_file_roundtrip(tmp_path, channels):
    image = np.random.default_rng(channels).integers(
        0, 256, size=(9, 11, channels), dtype=np.uint8
    )
    path = tmp_path / "roundtrip.qoi"

    iio.imwrite(path, image)
    decoded = iio.imread(path)

    assert decoded.dtype == np.uint8
    assert decoded.shape == image.shape
    np.testing.assert_array_equal(decoded, image)


def test_bytes_roundtrip(rgba_image):
    encoded = iio.imwrite("<bytes>", rgba_image, extension=".qoi")
    assert isinstance(encoded, bytes)
    decoded = iio.imread(encoded, extension=".qoi")
    np.testing.assert_array_equal(decoded, rgba_image)


def test_filelike_roundtrip(rgb_image):
    buffer = io.BytesIO()
    iio.imwrite(buffer, rgb_image, plugin="qoi")
    buffer.seek(0)
    decoded = iio.imread(buffer, plugin="qoi")
    np.testing.assert_array_equal(decoded, rgb_image)


def test_explicit_format_hint(tmp_path, rgb_image):
    # no .qoi suffix; the extension hint must route the request to us
    path = tmp_path / "blob_without_extension"
    iio.imwrite(path, rgb_image, extension=".qoi")
    decoded = iio.imread(path, extension=".qoi")
    np.testing.assert_array_equal(decoded, rgb_image)

    # the deprecated format_hint kwarg keeps working
    encoded = iio.imwrite("<bytes>", rgb_image, format_hint=".qoi")
    decoded = iio.imread(encoded, format_hint=".qoi")
    np.testing.assert_array_equal(decoded, rgb_image)


def test_iter_yields_single_image(tmp_path, rgb_image):
    path = tmp_path / "iter.qoi"
    iio.imwrite(path, rgb_image)
    images = list(iio.imiter(path))
    assert len(images) == 1
    np.testing.assert_array_equal(images[0], rgb_image)


def test_properties_and_metadata(tmp_path, rgba_image):
    path = tmp_path / "meta.qoi"
    iio.imwrite(path, rgba_image, colorspace=1)

    properties = iio.improps(path)
    assert properties.shape == rgba_image.shape
    assert properties.dtype == np.dtype(np.uint8)
    assert properties.n_images == 1
    assert properties.is_batch is False

    metadata = iio.immeta(path)
    assert metadata == {
        "width": rgba_image.shape[1],
        "height": rgba_image.shape[0],
        "channels": 4,
        "colorspace": 1,
    }


@pytest.mark.parametrize("colorspace", [0, 1])
def test_colorspace_roundtrip(colorspace, rgb_image):
    encoded = iio.imwrite("<bytes>", rgb_image, plugin="qoi", colorspace=colorspace)
    assert encoded[13] == colorspace
    assert iio.immeta(encoded, extension=".qoi")["colorspace"] == colorspace
    np.testing.assert_array_equal(iio.imread(encoded, extension=".qoi"), rgb_image)


def test_non_contiguous_input(tmp_path):
    base = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    sliced = base[:, ::2, :]
    assert not sliced.flags["C_CONTIGUOUS"]

    path = tmp_path / "strided.qoi"
    iio.imwrite(path, sliced)
    np.testing.assert_array_equal(iio.imread(path), np.ascontiguousarray(sliced))


# ---------------------------------------------------------------------------
# Encoder opcode coverage
# ---------------------------------------------------------------------------


def test_all_opcodes_are_emitted():
    # Construct an image that forces INDEX, DIFF, LUMA, RUN, RGB and RGBA
    # chunks and verify the resulting byte stream contains every tag.
    image = np.zeros((4, 8, 4), dtype=np.uint8)

    # RUN: long stretch of identical pixels
    image[0, :6] = (10, 20, 30, 255)
    # DIFF: small per-channel change from the previous pixel
    image[0, 6] = (11, 19, 31, 255)
    # LUMA: larger green-dominated change outside the DIFF range
    image[0, 7] = (15, 35, 38, 255)
    # RGB: arbitrary change with unchanged alpha
    image[1, 0] = (200, 100, 50, 255)
    # RGBA: alpha change forces a full RGBA chunk
    image[1, 1] = (200, 100, 50, 128)
    # INDEX: revisit a color previously stored in the hash array
    image[1, 2] = (10, 20, 30, 255)
    # second RUN segment exercising run flushing on a changed pixel
    image[2, :7] = (7, 7, 7, 255)
    image[2, 7] = (8, 8, 8, 255)
    image[3, :] = (8, 8, 8, 255)

    encoded = qoi._encode_qoi(image)
    tags = encoded[HEADER_SIZE:-8]

    assert qoi.QOI_OP_RGB in tags
    assert qoi.QOI_OP_RGBA in tags
    assert any(b & 0xC0 == qoi.QOI_OP_INDEX for b in tags)
    assert any(b & 0xC0 == qoi.QOI_OP_DIFF for b in tags)
    assert any(b & 0xC0 == qoi.QOI_OP_LUMA for b in tags)
    assert any(
        b & 0xC0 == qoi.QOI_OP_RUN and b not in (qoi.QOI_OP_RGB, qoi.QOI_OP_RGBA)
        for b in tags
    )

    # every emitted chunk must decode back to the original pixels
    np.testing.assert_array_equal(qoi._decode_qoi(encoded)["image"], image)


def test_run_lengths_are_legal_and_split_at_62():
    # 200 identical pixels force runs that must be split at the 62-pixel cap
    image = np.zeros((1, 200, 3), dtype=np.uint8)
    image[:] = (42, 42, 42)
    encoded = qoi._encode_qoi(image)

    run_bytes = [
        b
        for b in encoded[HEADER_SIZE:-8]
        if b & 0xC0 == qoi.QOI_OP_RUN and b not in (qoi.QOI_OP_RGB, qoi.QOI_OP_RGBA)
    ]
    assert run_bytes  # runs were actually used
    for b in run_bytes:
        # stored run length minus one must stay below the RGB/RGBA tag range
        assert (b & 0x3F) <= 61
    # the first pixel is emitted as a full RGB chunk; the rest are runs
    assert sum((b & 0x3F) + 1 for b in run_bytes) == 199
    np.testing.assert_array_equal(qoi._decode_qoi(encoded)["image"], image)


def test_signed_char_wraparound_matches_reference_semantics():
    # A 1 -> 255 jump has a raw difference of 254, but the reference encoder
    # computes it in signed char arithmetic (-2) and may pick QOI_OP_DIFF.
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    image[0, 0] = (0, 0, 0)
    image[0, 1] = (1, 1, 1)
    image[0, 2] = (255, 255, 255)
    encoded = qoi._encode_qoi(image)
    # the last pixel must be encoded as a DIFF rather than a full RGB chunk
    assert encoded[-9] & 0xC0 == qoi.QOI_OP_DIFF
    np.testing.assert_array_equal(qoi._decode_qoi(encoded)["image"], image)


@pytest.mark.parametrize("channels", [3, 4])
def test_random_images_roundtrip(channels):
    rng = np.random.default_rng(2024 + channels)
    for shape in [(1, 1), (2, 3), (17, 13), (33, 65)]:
        image = rng.integers(0, 256, size=(*shape, channels), dtype=np.uint8)
        encoded = qoi._encode_qoi(image)
        decoded = qoi._decode_qoi(encoded)["image"]
        np.testing.assert_array_equal(decoded, image)


def test_long_run_images_roundtrip():
    for channels, space in product((3, 4), (0, 1)):
        image = np.zeros((40, 40, channels), dtype=np.uint8)
        image[:] = (10, 128, 240, 70)[:channels]
        # a 65-pixel stripe crosses the 62-pixel run boundary
        image[0, :65] = (1, 2, 3, 4)[:channels]
        encoded = qoi._encode_qoi(image, colorspace=space)
        decoded = qoi._decode_qoi(encoded)
        assert decoded["colorspace"] == space
        np.testing.assert_array_equal(decoded["image"], image)


# ---------------------------------------------------------------------------
# Hand-crafted byte streams: exact decoder behavior
# ---------------------------------------------------------------------------


def test_decode_known_byte_stream():
    # 2x1 RGB: QOI_OP_RGB (1,2,3) then a DIFF (+1,-1,0) -> (2,1,3)
    stream = (
        make_header(2, 1, channels=3)
        + b"\xfe\x01\x02\x03"
        + bytes((0x40 | (3 << 4) | (1 << 2) | 2,))
        + END_MARKER
    )
    decoded = qoi._decode_qoi(stream)
    expected = np.array([[[1, 2, 3], [2, 1, 3]]], dtype=np.uint8)
    np.testing.assert_array_equal(decoded["image"], expected)
    assert decoded["width"] == 2 and decoded["height"] == 1
    assert decoded["channels"] == 3 and decoded["colorspace"] == 0


def test_decode_luma_chunk():
    # 1x2 RGBA: first pixel RGB(0,10,0) alpha 255, second via LUMA with
    # vg = 8, vg_r = 1, vg_b = -1 -> (9, 18, 7, 255)
    stream = (
        make_header(1, 2, channels=4)
        + b"\xfe\x00\x0a\x00"
        + bytes((0x80 | (8 + 32), ((1 + 8) << 4) | (-1 + 8)))
        + END_MARKER
    )
    decoded = qoi._decode_qoi(stream)["image"]
    np.testing.assert_array_equal(
        decoded, np.array([[[0, 10, 0, 255]], [[9, 18, 7, 255]]], np.uint8)
    )


def test_decode_index_chunk():
    # Two identical pixels: the second is served from the color index array
    index_pos = qoi._hash_index(5, 6, 7, 255)
    stream = (
        make_header(2, 1, channels=3)
        + b"\xfe\x05\x06\x07"
        + bytes((qoi.QOI_OP_INDEX | index_pos,))
        + END_MARKER
    )
    decoded = qoi._decode_qoi(stream)["image"]
    np.testing.assert_array_equal(decoded, np.array([[[5, 6, 7], [5, 6, 7]]], np.uint8))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_bad_magic_raises_initialization_error():
    data = b"XXXX" + b"\x00" * 10 + END_MARKER
    with pytest.raises(InitializationError):
        qoi._read_header(data)
    # auto-detection surfaces this as an OSError and tries other backends
    with pytest.raises(OSError):
        iio.imread(b"definitely not qoi data here", extension=".qoi")


def test_too_short_data():
    with pytest.raises(QoiError):
        qoi._decode_qoi(b"qoif" + b"\x00" * 5)


def test_oversized_dimensions():
    data = make_header(200_000_000, 200_000_000) + b"\xc0" + END_MARKER
    with pytest.raises(QoiError, match="too large"):
        qoi._decode_qoi(data)


def test_zero_dimensions():
    with pytest.raises(QoiError, match="non-zero"):
        qoi._decode_qoi(make_header(0, 4) + END_MARKER)
    with pytest.raises(QoiError, match="non-zero"):
        qoi._decode_qoi(make_header(4, 0) + END_MARKER)


def test_invalid_header_bytes():
    with pytest.raises(QoiError, match="channels"):
        qoi._decode_qoi(make_header(1, 1, channels=2) + END_MARKER)
    with pytest.raises(QoiError, match="colorspace"):
        qoi._decode_qoi(make_header(1, 1, colorspace=9) + END_MARKER)


def test_truncated_opcodes():
    header = make_header(2, 1)

    truncated_rgb = header + b"\xfe\x01\x02" + END_MARKER
    with pytest.raises(QoiError, match="QOI_OP_RGB"):
        qoi._decode_qoi(truncated_rgb)

    truncated_rgba = header + b"\xff\x01\x02\x03" + END_MARKER
    with pytest.raises(QoiError, match="QOI_OP_RGBA"):
        qoi._decode_qoi(truncated_rgba)

    truncated_luma = header + b"\xfe\x01\x01\x01" + b"\xa8" + END_MARKER
    with pytest.raises(QoiError, match="QOI_OP_LUMA"):
        qoi._decode_qoi(truncated_luma)


def test_stream_ends_before_all_pixels():
    # header promises 4 pixels, but only one run of length one is present
    stream = make_header(2, 2) + b"\xc0" + END_MARKER
    with pytest.raises(QoiError, match="truncated"):
        qoi._decode_qoi(stream)


def test_invalid_end_marker():
    good = qoi._encode_qoi(np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(QoiError, match="end marker"):
        qoi._decode_qoi(good[:-1] + b"\x00")


def test_extra_trailing_data():
    good = qoi._encode_qoi(np.zeros((2, 2, 3), np.uint8))
    # insert an extra byte before the end marker: the marker stays intact,
    # but one chunk byte remains after the last pixel was decoded
    with_trailing = good[:-8] + b"\x00" + END_MARKER
    with pytest.raises(QoiError, match="trailing data"):
        qoi._decode_qoi(with_trailing)


def test_encoder_rejects_bad_inputs():
    with pytest.raises(QoiError, match="shape"):
        qoi._encode_qoi(np.zeros((4, 4), np.uint8))
    with pytest.raises(QoiError, match="shape"):
        qoi._encode_qoi(np.zeros((4, 4, 2), np.uint8))
    with pytest.raises(QoiError, match="dtype"):
        qoi._encode_qoi(np.zeros((4, 4, 3), np.uint16))
    with pytest.raises(QoiError, match="colorspace"):
        qoi._encode_qoi(np.zeros((4, 4, 3), np.uint8), colorspace=2)


def test_read_index_out_of_bounds(rgb_image):
    encoded = iio.imwrite("<bytes>", rgb_image, plugin="qoi")
    with iio.imopen(encoded, "r", extension=".qoi") as plugin:
        with pytest.raises(IndexError):
            plugin.read(index=1)
        with pytest.raises(IndexError):
            plugin.properties(index=1)
        with pytest.raises(IndexError):
            plugin.metadata(index=1)
        # the single-image selectors are accepted
        np.testing.assert_array_equal(plugin.read(index=0), rgb_image)
        np.testing.assert_array_equal(plugin.read(index=...), rgb_image)


def test_short_header_is_initialization_error():
    from imageio.core import Request

    request = Request(b"qoif\x00\x00", "r", extension=".qoi")
    with pytest.raises(InitializationError):
        QoiPlugin(request)
    # through imopen the same failure becomes an OSError and lets
    # auto-detection continue with other backends
    with pytest.raises(OSError):
        iio.imopen(b"qoif\x00\x00", "r", extension=".qoi")


def test_invalid_header_surfaces_lazily():
    # correct magic but an invalid channels byte: construction succeeds so
    # that auto-detection recognizes QOI, and read()/properties() raise later
    data = make_header(1, 1, channels=2) + END_MARKER
    with iio.imopen(data, "r", extension=".qoi") as plugin:
        with pytest.raises(QoiError, match="channels"):
            plugin.read()
    with iio.imopen(data, "r", extension=".qoi") as plugin:
        with pytest.raises(QoiError, match="channels"):
            plugin.properties()
    with iio.imopen(data, "r", extension=".qoi") as plugin:
        with pytest.raises(QoiError, match="channels"):
            plugin.metadata()


def test_encoder_oversize_guard():
    # avoid allocating 400 million pixels via a zero-stride placeholder array
    placeholder = np.zeros(1, dtype=np.uint8)
    huge = np.lib.stride_tricks.as_strided(
        placeholder, shape=(200_000, 200_000, 3), strides=(0, 0, 0)
    )
    with pytest.raises(QoiError, match="too large"):
        qoi._encode_qoi(huge)


def test_read_from_filelike_returning_bytearray(rgb_image):
    class ByteArrayFile:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return bytearray(self._payload)

        def close(self):
            pass

    encoded = qoi._encode_qoi(rgb_image)
    with iio.imopen(ByteArrayFile(encoded), "r", extension=".qoi") as plugin:
        np.testing.assert_array_equal(plugin.read(), rgb_image)


# ---------------------------------------------------------------------------
# Compatibility with the official QOI test images.
# Download https://qoiformat.org/qoi_test_images.zip, extract it and point
# IMAGEIO_QOI_TEST_IMAGES at the directory containing the paired .qoi/.png
# files to enable these checks (Pillow is used to read the reference PNGs).
# ---------------------------------------------------------------------------


def _official_images():
    import os
    from pathlib import Path

    directory = os.getenv("IMAGEIO_QOI_TEST_IMAGES")
    if not directory:
        return []
    return sorted(Path(directory).glob("*.qoi"))


@pytest.mark.skipif(
    not _official_images(),
    reason="Set IMAGEIO_QOI_TEST_IMAGES to the extracted official qoi_test_images",
)
@pytest.mark.parametrize("qoi_path", _official_images())
def test_official_samples(qoi_path):
    from PIL import Image

    data = qoi_path.read_bytes()
    decoded = qoi._decode_qoi(data)
    image = decoded["image"]

    # the matching PNG is the ground-truth pixel data
    reference = np.asarray(Image.open(qoi_path.with_suffix(".png")))
    if reference.ndim == 3 and reference.shape[2] == 3 and image.shape[2] == 4:
        # edgecase.png is RGB while the QOI file stores an opaque alpha
        reference = np.dstack([reference, np.full(reference.shape[:2], 255, np.uint8)])
    np.testing.assert_array_equal(image, reference)

    # re-encoding the decoded pixels must produce a valid stream that
    # round-trips exactly
    re_encoded = qoi._encode_qoi(image, colorspace=decoded["colorspace"])
    np.testing.assert_array_equal(qoi._decode_qoi(re_encoded)["image"], image)
