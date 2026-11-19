"""Tests for imageio's pillow plugin"""

import io
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from imageio.core.request import InitializationError, Request, SeekableFileObject
from imageio.core.v3_plugin_api import PluginV3
from imageio.plugins.pillow import PillowPlugin
from PIL import Image, ImageSequence, ImageOps, __version__  # type: ignore

PILLOW_VERSION = tuple(int(x) for x in __version__.split("."))


@pytest.mark.parametrize(
    "im_npy,im_out,im_comp",
    [
        ("chelsea.npy", "iio.png", "pil.png"),
        ("chelsea.npy", "iio.jpg", "pil.jpg"),
        ("chelsea.npy", "iio.jpeg", "pil.jpg"),
        ("chelsea.npy", "iio.bmp", "pil.bmp"),
    ],
)
def test_write_single_frame(test_images, tmp_path, im_npy, im_out, im_comp):
    # the base image as numpy array
    im = np.load(test_images / im_npy)
    # written with imageio
    iio_file = tmp_path / im_out
    iio.imwrite(iio_file, im, plugin="pillow")

    # written with pillow directly
    pil_file = tmp_path / im_comp
    Image.fromarray(im).save(pil_file)

    # file exists
    assert os.path.exists(iio_file)

    # imageio content matches pillow content
    assert iio_file.read_bytes() == pil_file.read_bytes()


@pytest.mark.parametrize(
    "im_npy,im_out,im_comp",
    [
        # Note: There might be a problem with reading/writing frames
        # Tracking Issue: https://github.com/python-pillow/Pillow/issues/5307
        ("newtonscradle_rgb.npy", "iio.gif", "pil.gif"),
        # ("newtonscradle_rgba.npy", "iio.gif", "pil.gif"),
    ],
)
@pytest.mark.needs_internet
def test_write_multiframe(test_images, tmp_path, im_npy, im_out, im_comp):
    # the base image as numpy array
    im = np.load(test_images / im_npy)
    # written with imageio
    iio_file = tmp_path / im_out
    iio.imwrite(iio_file, im, plugin="pillow")

    # written with pillow directly
    pil_file = tmp_path / im_comp
    pil_images = [Image.fromarray(frame) for frame in im]
    pil_images[0].save(pil_file, save_all=True, append_images=pil_images[1:])

    # file exists
    assert os.path.exists(iio_file)

    # imageio content matches pillow content
    assert iio_file.read_bytes() == pil_file.read_bytes()


def test_png_compression(test_images, tmp_path):
    # Note: Note sure if we should test this or pillow

    im = np.load(test_images / "chelsea.npy")

    iio.imwrite(tmp_path / "1.png", im, plugin="pillow", compress_level=0)
    iio.imwrite(tmp_path / "2.png", im, plugin="pillow", compress_level=9)

    size_1 = os.stat(tmp_path / "1.png").st_size
    size_2 = os.stat(tmp_path / "2.png").st_size
    assert size_2 < size_1


def test_png_quantization(test_images, tmp_path):
    # Note: Note sure if we should test this or pillow

    im = np.load(test_images / "chelsea.npy")

    iio.imwrite(tmp_path / "1.png", im, plugin="pillow", bits=8)
    iio.imwrite(tmp_path / "2.png", im, plugin="pillow", bits=2)

    size_1 = os.stat(tmp_path / "1.png").st_size
    size_2 = os.stat(tmp_path / "2.png").st_size
    assert size_2 < size_1


def test_png_16bit(test_images, tmp_path):
    im = np.load(test_images / "chelsea.npy")[..., 0]
    iio.imwrite(
        tmp_path / "1.png",
        2 * im.astype(np.uint16),
        plugin="pillow",
        mode="I",
    )
    iio.imwrite(tmp_path / "2.png", im, plugin="pillow", mode="L")

    size_1 = os.stat(tmp_path / "1.png").st_size
    size_2 = os.stat(tmp_path / "2.png").st_size
    assert size_2 < size_1

    im2 = iio.imread(tmp_path / "2.png", plugin="pillow")
    assert im2.dtype == np.uint8

    if PILLOW_VERSION >= (10, 0, 0):
        im3 = iio.imread(tmp_path / "1.png", plugin="pillow")
        assert im3.dtype == np.uint16
    else:
        # legacy behavior for pillow < v10.0.0
        with pytest.warns(UserWarning):
            im3 = iio.imread(tmp_path / "1.png", plugin="pillow")

        assert im3.dtype == np.int32


@pytest.mark.needs_internet
def test_png_remote():
    # issue #202

    url = "https://raw.githubusercontent.com/imageio/test_images/main/chelsea.png"
    im = iio.imread(url, plugin="pillow")
    assert im.shape == (300, 451, 3)


def test_png_transparent_pixel(test_images):
    # see issue #245
    im = iio.imread(
        test_images / "imageio_issue246.png",
        plugin="pillow",
        mode="RGBA",
    )
    assert im.shape == (24, 30, 4)


def test_png_gamma_correction(test_images: Path):
    # opens the file twice, but touches more parts of the API
    im1 = iio.imread(test_images / "kodim03.png", plugin="pillow")
    im1_meta = iio.immeta(
        test_images / "kodim03.png", plugin="pillow", exclude_applied=False
    )

    im2 = iio.imread(
        test_images / "kodim03.png",
        plugin="pillow",
        apply_gamma=True,
    )

    # Test result depending of application of gamma
    assert im1_meta["gamma"] < 1
    assert im1.mean() < im2.mean()

    assert im1.shape == (512, 768, 3)
    assert im1.dtype == "uint8"
    assert im2.shape == (512, 768, 3)
    assert im2.dtype == "uint8"


def test_jpg_compression(test_images, tmp_path):
    # Note: Note sure if we should test this or pillow

    im = np.load(test_images / "chelsea.npy")

    iio.imwrite(tmp_path / "1.jpg", im, plugin="pillow", quality=90)
    iio.imwrite(tmp_path / "2.jpg", im, plugin="pillow", quality=10)

    size_1 = os.stat(tmp_path / "1.jpg").st_size
    size_2 = os.stat(tmp_path / "2.jpg").st_size
    assert size_2 < size_1


def test_exif_orientation(test_images, tmp_path):
    from PIL.Image import Exif  # type: ignore

    im = np.load(test_images / "chelsea.npy")

    # original image is has landscape format
    assert im.shape[0] < im.shape[1]

    im_flipped = np.rot90(im, 1)
    exif_tag = Exif()
    exif_tag[274] = 6  # Set Orientation to 6

    iio.imwrite(
        tmp_path / "chelsea_tagged.png",
        im_flipped,
        plugin="pillow",
        exif=exif_tag,
    )

    with iio.imopen(
        tmp_path / "chelsea_tagged.png",
        "r",
        plugin="pillow",
    ) as f:
        im_reloaded = f.read()
        im_meta = f.get_meta()

    # ensure raw image is now portrait
    assert im_reloaded.shape[0] > im_reloaded.shape[1]
    # ensure that the Exif tag is set in the file
    assert "Orientation" in im_meta and im_meta["Orientation"] == 6

    im_reloaded = iio.imread(
        tmp_path / "chelsea_tagged.png", plugin="pillow", rotate=True
    )

    assert np.array_equal(im, im_reloaded)

    # apply the transform with Pillow's exif_transpose
    with Image.open(tmp_path / "chelsea_tagged.png") as f:
        im_pillow = np.asarray(ImageOps.exif_transpose(f))

    assert np.array_equal(im, im_pillow)


@pytest.mark.parametrize("orientation", [0, 9])
def test_exif_invalid_orientation(test_images, tmp_path, orientation):
    from PIL.Image import Exif  # type: ignore

    im = np.load(test_images / "chelsea.npy")

    exif_tag = Exif()
    exif_tag[274] = orientation  # Set Orientation to 0, which is invalid

    iio.imwrite(
        tmp_path / "chelsea_tagged.png",
        im,
        plugin="pillow",
        exif=exif_tag,
    )

    with iio.imopen(
        tmp_path / "chelsea_tagged.png",
        "r",
        plugin="pillow",
    ) as f:
        im_reloaded = f.read()
        im_meta = f.get_meta()

    # ensure that the Exif tag is set in the file
    assert "Orientation" in im_meta and im_meta["Orientation"] == orientation

    im_reloaded = iio.imread(
        tmp_path / "chelsea_tagged.png", plugin="pillow", rotate=True
    )

    assert np.array_equal(im, im_reloaded)

    # apply the transform with Pillow's exif_transpose
    with Image.open(tmp_path / "chelsea_tagged.png") as f:
        im_pillow = np.asarray(ImageOps.exif_transpose(f))

    assert np.array_equal(im, im_pillow)


def test_gif_rgb_vs_rgba(test_images):
    # Note: I don't understand the point of this test
    im_rgb = iio.imread(
        test_images / "newtonscradle.gif",
        plugin="pillow",
        mode="RGB",
    )
    im_rgba = iio.imread(
        test_images / "newtonscradle.gif",
        plugin="pillow",
        mode="RGBA",
    )

    assert np.allclose(im_rgb, im_rgba[..., :3])


def test_gif_gray(test_images, tmp_path):
    # Note: There was no assert here; we test that it doesn't crash?
    im = iio.imread(
        test_images / "newtonscradle.gif",
        plugin="pillow",
        mode="L",
    )

    iio.imwrite(
        tmp_path / "test.gif",
        im[..., 0],
        plugin="pillow",
        duration=0.2,
        mode="L",
    )


def test_gif_fps_warning(test_images, tmp_path):
    im = iio.imread(
        test_images / "newtonscradle.gif",
        plugin="pillow",
        mode="L",
    )

    with pytest.warns(DeprecationWarning):
        iio.imwrite(
            tmp_path / "test.gif",
            im[..., 0],
            plugin="pillow",
            fps=60,
            mode="L",
        )


def test_gif_irregular_duration(test_images, tmp_path):
    im = iio.imread(
        test_images / "newtonscradle.gif",
        plugin="pillow",
        mode="RGBA",
    )
    duration = [0.5 if idx in [2, 5, 7] else 0.1 for idx in range(im.shape[0])]

    with iio.imopen(tmp_path / "test.gif", "w", plugin="pillow") as file:
        for frame, duration in zip(im, duration):
            file.write(frame, duration=duration)

    # how to assert duration here


def test_gif_palletsize(test_images, tmp_path):
    im = iio.imread(
        test_images / "newtonscradle.gif",
        plugin="pillow",
        mode="RGBA",
    )

    iio.imwrite(tmp_path / "test.gif", im, plugin="pillow", palletsize=100)
    # TODO: assert pallet size is 128


def test_gif_loop_and_duration(test_images, tmp_path):
    # Note: I think this test tests pillow kwargs, not imageio functionality
    # maybe we should drop it?

    im = iio.imread(
        test_images / "newtonscradle.gif",
        plugin="pillow",
        mode="RGBA",
    )

    with iio.imopen(tmp_path / "test.gif", "w", plugin="pillow") as file:
        for frame in im:
            file.write(frame, palettesize=100, duration=0.00005, loop=2)

    # This test had no assert; how to assert fps and loop count?


def test_gif_indexed_read(test_images):
    idx = 0
    numpy_im = np.load(test_images / "newtonscradle_rgb.npy")[idx, ...]

    with iio.imopen(test_images / "newtonscradle.gif", "r", plugin="pillow") as file:
        # exists to touch branch, would be better two write an explicit test
        meta = file.get_meta(index=idx)
        assert "version" in meta

        pillow_im = file.read(index=idx, mode="RGB")

    assert np.allclose(pillow_im, numpy_im)


def test_unknown_image(tmp_path):
    with open(tmp_path / "foo.unknown", "w") as file:
        file.write("This image, which is actually no image, has an unknown image type.")

    with pytest.raises(InitializationError):
        r = Request(tmp_path / "foo.unknown", "r")
        PillowPlugin(r)


def test_gif_transparent_pixel(test_images):
    # see issue #245
    im = iio.imread(
        test_images / "imageio_issue245.gif", plugin="pillow", mode="RGBA", index=0
    )
    assert im.shape == (24, 30, 4)


def test_gif_list_write(test_images, tmp_path):
    im = iio.imread(test_images / "imageio_issue245.gif", plugin="pillow")
    im_list = [x for x in im]
    iio.imwrite(tmp_path / "test.gif", im_list, plugin="pillow")
    im2 = iio.imread(test_images / "imageio_issue245.gif", plugin="pillow", index=0)

    assert im2.shape == (24, 30, 3)


@pytest.mark.needs_internet
def test_gif_first_p_frame():
    # Bugfix: https://github.com/imageio/imageio/issues/1030
    im = iio.imread(
        "https://raw.githubusercontent.com/imageio/test_images/refs/heads/main/newtonscradle.gif",
        plugin="pillow",
        index=None,
    )
    assert im.shape == (36, 150, 200, 3)


def _corner_blink_frames(seed=0, size=64, corner=6):
    """Three frames where only a small corner region changes."""

    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 255, (size, size, 4), dtype=np.uint8)
    canvas[..., 3] = 255
    frames = []
    for color in [(255, 0, 0, 255), (0, 0, 255, 255), (255, 0, 0, 255)]:
        frame = canvas.copy()
        frame[:corner, :corner] = color
        frames.append(frame)
    return np.stack(frames)


def _decode_gif(blob):
    """Return composited RGBA frames, per-frame duration, tile and disposal."""

    decoded, durations, tiles, disposals = [], [], [], []
    n_frames = Image.open(io.BytesIO(blob)).n_frames
    for index in range(n_frames):
        image = Image.open(io.BytesIO(blob))
        image.seek(index)
        tile = image.tile[0][1] if image.tile else (0, 0) + image.size
        disposals.append(image.disposal_method)
        durations.append(image.info.get("duration"))
        image.load()
        decoded.append(np.asarray(image.convert("RGBA")))
        tiles.append(tile)
    return np.stack(decoded), durations, tiles, disposals, n_frames


def _assert_visually_equal(readback, original, color_tol=60):
    # alpha must be exact; RGB is compared only on opaque input pixels
    alpha_diff = np.abs(
        readback[..., 3].astype(int) - original[..., 3].astype(int)
    ).max()
    opaque = original[..., 3] == 255
    color_diff = np.abs(readback[..., :3].astype(int) - original[..., :3].astype(int))[
        opaque
    ].max()
    assert alpha_diff == 0
    assert color_diff <= color_tol


def test_gif_subrectangles_corner_blink():
    # the headline case: a tiny icon blinks while the full canvas is noisy
    frames = _corner_blink_frames()
    durations = [100, 200, 300]

    full = iio.imwrite("<bytes>", frames, extension=".gif", duration=durations, loop=0)
    delta = iio.imwrite(
        "<bytes>",
        frames,
        extension=".gif",
        duration=durations,
        loop=0,
        subrectangles=True,
    )

    readback, got_durations, tiles, disposals, n_frames = _decode_gif(delta)

    # only the corner region is stored after the first frame, so the file
    # is markedly smaller than full frames
    assert len(delta) < len(full)
    # the first frame always covers the whole canvas
    assert n_frames == 3
    assert tiles[0] == (0, 0, 64, 64)
    assert tiles[1:] == [(0, 0, 6, 6), (0, 0, 6, 6)]
    # every playback instant matches the full-frame input, with exact delays
    for index in range(3):
        _assert_visually_equal(readback[index], frames[index])
        assert got_durations[index] == durations[index]
    # do-not-dispose keeps the composited canvas intact
    assert disposals == [1, 1, 1]


def test_gif_subrectangles_default_unchanged():
    frames = _corner_blink_frames()
    kwargs = dict(extension=".gif", duration=[100, 200, 300], loop=0)
    first = iio.imwrite("<bytes>", frames, **kwargs)
    second = iio.imwrite("<bytes>", frames, **kwargs)
    assert first == second


def test_gif_subrectangles_keeps_identical_frame():
    frames = _corner_blink_frames()
    sequence = np.stack([frames[0], frames[0].copy(), frames[1]])

    blob = iio.imwrite(
        "<bytes>",
        sequence,
        extension=".gif",
        duration=[100, 250, 300],
        subrectangles=True,
    )
    readback, durations, tiles, _, n_frames = _decode_gif(blob)

    # the identical frame is retained (not merged) with its own delay
    assert n_frames == 3
    assert durations == [100, 250, 300]
    assert tiles[1] == (0, 0, 1, 1)
    assert np.array_equal(readback[1], readback[0])


def test_gif_subrectangles_transparent_erase():
    base = np.zeros((64, 64, 4), np.uint8)
    base[12:52, 12:52] = (90, 140, 200, 255)
    appears = base.copy()
    appears[:6, :6] = (255, 255, 255, 255)
    disappears = base.copy()  # corner goes back to fully transparent

    blob = iio.imwrite(
        "<bytes>",
        np.stack([base, appears, disappears]),
        extension=".gif",
        duration=[100, 200, 300],
        subrectangles=True,
    )
    readback, _, _, disposals, _ = _decode_gif(blob)

    _assert_visually_equal(readback[0], base)
    _assert_visually_equal(readback[1], appears)
    _assert_visually_equal(readback[2], disappears)
    # erasing content requires restore-to-background on the preceding frame
    assert disposals[1] == 2
    assert readback[1][3, 3, 3] == 255
    assert readback[2][3, 3, 3] == 0


def test_gif_subrectangles_palette_change():
    # each frame introduces a brand-new corner color; per-tile local
    # palettes must represent them correctly
    frames = []
    for color in [(255, 0, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]:
        frame = np.zeros((64, 64, 4), np.uint8)
        frame[20:44, 20:44] = (0, 200, 0, 255)
        frame[:6, :6] = color
        frames.append(frame)

    blob = iio.imwrite(
        "<bytes>",
        np.stack(frames),
        extension=".gif",
        duration=[100, 200, 300],
        subrectangles=True,
    )
    readback, _, _, _, _ = _decode_gif(blob)
    for index, frame in enumerate(frames):
        _assert_visually_equal(readback[index], frame)


def test_gif_subrectangles_mismatched_sizes():
    with pytest.raises(ValueError, match="same width and height"):
        iio.imwrite(
            "<bytes>",
            [np.zeros((32, 32, 4), np.uint8), np.zeros((16, 16, 4), np.uint8)],
            extension=".gif",
            subrectangles=True,
        )


def test_gif_subrectangles_single_frame():
    frames = _corner_blink_frames()
    blob = iio.imwrite(
        "<bytes>",
        frames[0][None],
        extension=".gif",
        duration=150,
        subrectangles=True,
    )
    readback, durations, tiles, _, n_frames = _decode_gif(blob)
    assert n_frames == 1
    assert tiles[0] == (0, 0, 64, 64)
    assert durations == [150]
    _assert_visually_equal(readback[0], frames[0])


def test_gif_subrectangles_large_sequence():
    frames = np.repeat(_corner_blink_frames(), 20, axis=0)  # 60 frames
    blob = iio.imwrite(
        "<bytes>", frames, extension=".gif", duration=50, subrectangles=True
    )
    readback, durations, _, _, n_frames = _decode_gif(blob)
    assert n_frames == 60
    assert all(duration == 50 for duration in durations)
    for index in range(n_frames):
        _assert_visually_equal(readback[index], frames[index])


def test_gif_subrectangles_loop_count():
    frames = _corner_blink_frames()
    blob = iio.imwrite(
        "<bytes>",
        frames,
        extension=".gif",
        duration=[100, 200, 300],
        loop=2,
        subrectangles=True,
    )
    index = blob.find(b"NETSCAPE2.0")
    assert blob[index + 13 : index + 15] == (2).to_bytes(2, "little")


def test_legacy_exif_orientation(test_images, tmp_path):
    from PIL.Image import Exif

    im = np.load(test_images / "chelsea.npy")

    # original image is has landscape format
    assert im.shape[0] < im.shape[1]

    im_flipped = np.rot90(im, 1)
    exif_tag = Exif()
    exif_tag[274] = 6  # Set Orientation to 6

    iio.imwrite(
        tmp_path / "chelsea_tagged.png",
        im_flipped,
        plugin="pillow",
        exif=exif_tag,
    )

    with iio.imopen(
        tmp_path / "chelsea_tagged.png",
        "r",
        legacy_mode=True,
        plugin="PNG-PIL",
    ) as f:
        im_reloaded = np.asarray(f.read()[0])
        im_meta = f.get_meta()

    # ensure raw image is now portrait
    assert im_reloaded.shape[0] > im_reloaded.shape[1]
    # ensure that the Exif tag is set in the file
    assert "exif" in im_meta

    im_reloaded = iio.imread(
        tmp_path / "chelsea_tagged.png", plugin="pillow", rotate=True
    )

    assert np.array_equal(im, im_reloaded)

    # apply the transform with Pillow's exif_transpose
    with Image.open(tmp_path / "chelsea_tagged.png") as f:
        im_pillow = np.asarray(ImageOps.exif_transpose(f))

    assert np.array_equal(im, im_pillow)


def test_incomatible_write_format(tmp_path):
    with pytest.raises(IOError):
        iio.imopen(tmp_path / "foo.mp3", "w", plugin="pillow", legacy_mode=False)


def test_write_to_bytes():
    image = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)

    # writing to bytes with pillow
    with io.BytesIO() as output:
        Image.fromarray(image).save(output, format="PNG")
        contents = output.getvalue()

    # writing to bytes with imageIO
    with io.BytesIO() as output:
        iio.imwrite(output, image, plugin="pillow", extension=".png")
        iio_contents = output.getvalue()

    assert iio_contents == contents


def test_write_to_bytes_rgba():
    image = (np.random.rand(600, 800, 4) * 255).astype(np.uint8)

    # writing to bytes with pillow
    with io.BytesIO() as output:
        Image.fromarray(image).save(output, format="PNG")
        contents = output.getvalue()

    # writing to bytes with imageIO
    with io.BytesIO() as output:
        iio.imwrite(output, image, plugin="pillow", extension=".png", mode="RGBA")
        iio_contents = output.getvalue()

    assert iio_contents == contents


def test_write_to_bytes_imwrite():
    image = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)

    # writing to bytes with pillow
    with io.BytesIO() as output:
        Image.fromarray(image).save(output, format="PNG")
        contents = output.getvalue()

    # write with ImageIO
    bytes_string = iio.imwrite("<bytes>", image, plugin="pillow", extension=".png")

    assert contents == bytes_string


def test_write_to_bytes_jpg():
    image = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)

    # writing to bytes with pillow
    with io.BytesIO() as output:
        Image.fromarray(image).save(output, format="JPEG")
        contents = output.getvalue()

    # write with ImageIO
    bytes_string = iio.imwrite("<bytes>", image, plugin="pillow", extension=".jpeg")

    assert contents == bytes_string


def test_write_jpg_to_bytes_io():
    # this is a regression test
    # see: https://github.com/imageio/imageio/issues/687

    image = np.zeros((200, 200), dtype=np.uint8)
    bytes_io = io.BytesIO()
    iio.imwrite(bytes_io, image, plugin="pillow", extension=".jpeg", mode="L")
    bytes_io.seek(0)

    image_from_file = iio.imread(bytes_io, plugin="pillow")
    assert np.allclose(image_from_file, image)


def test_initialization_failure(test_images):
    test_image = b"this is not an image and will break things."

    with pytest.raises(OSError):
        iio.imread(test_image, plugin="pillow")

    with pytest.raises(OSError):
        # pillow can not handle npy
        iio.imread(test_images / "chelsea_jpg.npy", plugin="pillow")


def test_boolean_reading(tmp_path):
    # Bugfix: https://github.com/imageio/imageio/issues/721
    expected = np.arange(256 * 256).reshape((256, 256)) % 2 == 0

    Image.fromarray(expected).save(tmp_path / "iio.png")

    actual = iio.imread(tmp_path / "iio.png")
    assert np.allclose(actual, expected)


def test_boolean_writing(tmp_path):
    # Bugfix: https://github.com/imageio/imageio/issues/721
    expected = np.arange(256 * 256).reshape((256, 256)) % 2 == 0

    iio.imwrite(tmp_path / "iio.png", expected)

    actual = np.asarray(Image.open(tmp_path / "iio.png"))
    # actual = iio.imread(tmp_path / "iio.png")
    assert np.allclose(actual, expected)


def test_quantized_gif(test_images, tmp_path):
    original = iio.imread(test_images / "newtonscradle.gif")

    iio.imwrite(tmp_path / "quantized.gif", original, plugin="pillow", bits=4)
    quantized = iio.imread(tmp_path / "quantized.gif")

    for original_frame, quantized_frame in zip(original, quantized):
        assert len(np.unique(quantized_frame)) <= len(np.unique(original_frame))


def test_properties(image_files: Path):
    file: PluginV3

    # test a flat image (RGB PNG)
    with iio.imopen(image_files / "chelsea.png", "r", plugin="pillow") as file:
        properties = file.properties()

    assert properties.shape == (300, 451, 3)
    assert properties.dtype == np.uint8
    assert properties.n_images is None

    # test a ndimage (GIF)
    properties = iio.improps(image_files / "newtonscradle.gif", plugin="pillow")
    assert properties.shape == (36, 150, 200, 3)
    assert properties.dtype == np.uint8
    assert properties.n_images == 36
    assert properties.is_batch is True

    # test a flat gray image
    properties = iio.improps(image_files / "text.png", plugin="pillow", index=0)

    assert properties.shape == (172, 448)
    assert properties.dtype == np.uint8
    assert properties.n_images is None

    # test an image format that doesn't have n_frames
    properties = iio.improps(image_files / "rommel.jpg", plugin="pillow")
    assert properties.n_images is None

    properties = iio.improps(image_files / "rommel.jpg", plugin="pillow", index=...)
    assert properties.n_images == 1


def test_metadata(test_images):
    meta = iio.immeta(test_images / "newtonscradle.gif")
    assert "version" in meta and meta["version"] == b"GIF89a"

    with iio.imopen(
        test_images / "newtonscradle.gif", "r", plugin="pillow"
    ) as image_file:
        image_file.read(index=5)
        meta = image_file.metadata(index=0)
        assert "version" in meta and meta["version"] == b"GIF89a"
        assert "palette" not in meta

    meta_all = iio.immeta(test_images / "newtonscradle.gif", exclude_applied=False)
    palette = meta_all["palette"]
    assert palette.shape == (102, 3)
    assert np.issubdtype(palette.dtype, np.number)


def test_apng_reading(tmp_path, test_images):
    # create a APNG
    img = iio.imread(test_images / "newtonscradle.gif")
    iio.imwrite(tmp_path / "test.apng", img)

    props = iio.improps(tmp_path / "test.apng")
    assert props.shape == (36, 150, 200, 3)
    assert props.is_batch is True

    # test single image read
    with Image.open(tmp_path / "test.apng") as im:
        im.seek(8)
        expected = np.asarray(im)
    actual = iio.imread(tmp_path / "test.apng", index=8)
    assert np.allclose(actual, expected)

    # test reading all frames
    all_frames = iio.imread(tmp_path / "test.apng")
    with Image.open(tmp_path / "test.apng") as im:
        for idx, frame in enumerate(ImageSequence.Iterator(im)):
            expected = np.asarray(frame)
            actual = all_frames[idx]
            assert np.allclose(actual, expected)


def test_apng_metadata(tmp_path, test_images):
    # create a APNG
    img = iio.imread(test_images / "newtonscradle.gif")
    iio.imwrite(tmp_path / "test.apng", img)

    metadata = iio.immeta(tmp_path / "test.apng")
    assert metadata["shape"] == (200, 150)
    assert metadata["loop"] == 0

    # set default index explicitly
    metadata2 = iio.immeta(tmp_path / "test.apng", index=...)
    assert metadata == metadata2


def test_write_format_warning(test_images):
    frames = iio.imread(test_images / "chelsea.png")
    bytes_image = iio.imwrite("<bytes>", frames, extension=".png", plugin="pillow")

    with pytest.warns(UserWarning):
        old_bytes = iio.imwrite("<bytes>", frames, plugin="pillow", format="PNG")

    assert bytes_image == old_bytes


def test_8bit_with_16bit_depth():
    rng = np.random.default_rng()

    img16 = rng.integers(2**0, 2**8, (128, 128), dtype=np.uint16)
    img16_bytes = iio.imwrite("<bytes>", img16, extension=".png", plugin="pillow")
    img16_read = iio.imread(img16_bytes)

    assert img16_read.dtype != np.uint8
    assert np.allclose(img16_read, img16)


def test_deprecated_kwargs(test_images):
    with pytest.raises(TypeError):
        iio.imread(test_images / "chelsea.png", plugin="pillow", as_gray=True)

    with pytest.warns(DeprecationWarning):
        img = iio.imread(test_images / "chelsea.png", plugin="pillow", pilmode="I")
        assert img.shape == (300, 451)

    with pytest.warns(DeprecationWarning):
        iio.imread(test_images / "chelsea.png", plugin="pillow", exifrotate=True)


def test_png_batch_fail():
    # this is a regression test for
    # https://github.com/imageio/imageio/issues/904

    img = np.ones((128, 128, 1), dtype=np.uint8)

    with pytest.raises(ValueError):
        iio.imwrite("<bytes>", img, extension=".png", plugin="pillow")


def test_incremental_gif_write(tmp_path):
    # this is a regression test for
    # https://github.com/imageio/imageio/issues/974

    rng = np.random.default_rng()

    with iio.imopen(tmp_path / "test.gif", "w", plugin="pillow") as file:
        for _ in range(10):
            img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
            file.write(img)

    final_gif = iio.imread(tmp_path / "test.gif")
    assert final_gif.shape == (10, 100, 100, 3)


def test_format_change_warning(tmp_path):
    img = np.full((100, 100, 3), 42, dtype=np.uint8)

    with iio.imopen(tmp_path / "test.gif", "w", plugin="pillow") as file:
        file.write(img, format="PNG")

        with pytest.warns(UserWarning):
            file.write(img, format="GIF")


def test_writable_output():
    rng = np.random.default_rng()
    img = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    buffer = iio.imwrite("<bytes>", img, extension=".png")

    frame = iio.imread(buffer, plugin="pillow")
    assert frame.flags["WRITEABLE"]

    frame = iio.imread(buffer, writeable_output=False, plugin="pillow")
    assert not frame.flags["WRITEABLE"]


@pytest.mark.needs_internet
def test_heif_remote():
    pytest.importorskip("pillow_heif")
    url = "http://github.com/tigranbs/test-heic-images/raw/master/image4.heic"
    im = iio.imread(url, plugin="pillow")
    assert im.shape == (476, 700, 4)


@pytest.mark.needs_internet
def test_avif_remote():
    pytest.importorskip("pillow_heif")

    url = "https://github.com/link-u/avif-sample-images/raw/master/fox.profile0.10bpc.yuv420.avif"
    im = iio.imread(url, plugin="pillow")
    assert im.shape == (800, 1204, 3)


@pytest.mark.needs_internet
def test_webp_remote():
    # this is a regression test for
    # https://github.com/imageio/imageio/issues/1065
    # the image in the issue is actually a webp image
    url = "https://github.com/python-pillow/Pillow/raw/main/Tests/images/hopper_orientation_2.webp"
    im = iio.imread(url, plugin="pillow")
    assert im.shape == (128, 128, 3)


def test_pillow_readline_on_seekable_stream():
    # this is a regression test for
    # https://github.com/imageio/imageio/issues/1007
    # Pillow's XPM (and IM) plugins call fp.readline(), which HTTP streams
    # expose via SeekableFileObject.
    xpm = (
        b"/* XPM */\n"
        b"static char * test_xpm[] = {\n"
        b'"1 1 1 1",\n'
        b'"a c #FF0000",\n'
        b'"a"\n'
        b"};\n"
    )
    img = iio.imread(SeekableFileObject(io.BytesIO(xpm)), plugin="pillow")
    assert img.shape == (1, 1, 3)
    assert np.array_equal(img[0, 0], [255, 0, 0])
