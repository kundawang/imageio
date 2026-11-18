# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Tests for the opt-in metadata preservation pipeline.

Covers ``imageio.plugins.pillow_metadata.MetadataPreserver`` and its
integration into the Pillow plugin's ``write`` via the ``preserve_metadata``
keyword. All assets are fixed: they are generated deterministically (fixed
seed / fixed bytes) inside ``tmp_path`` and carry a real ICC profile and
EXIF tags.
"""

import struct

import numpy as np
import pytest
from PIL import ExifTags, Image, ImageCms

import imageio.v3 as iio
from imageio.plugins.pillow_metadata import (
    DEFAULT_EXIF_ALLOWLIST,
    MetadataPreserver,
)


@pytest.fixture(scope="module")
def icc_profile():
    """A real (sRGB) ICC profile, fixed for every test run."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@pytest.fixture(scope="module")
def pixels():
    """Fixed test pixels (fixed seed -> identical asset on every run)."""
    rng = np.random.default_rng(20240117)
    return rng.integers(0, 256, size=(60, 40, 3), dtype=np.uint8)


def _source_exif(orientation=6):
    """EXIF block with orientation, dimensions, camera and privacy fields."""
    exif = Image.Exif()
    exif[0x0112] = orientation  # Orientation
    exif[0x010F] = "FixedMake"  # Make
    exif[0x0110] = "FixedModel"  # Model
    exif[0x013B] = "Jane Photographer"  # Artist (privacy)
    exif[0x9286] = "private note"  # UserComment (privacy)
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[0xA002] = 40  # ExifImageWidth
    exif_ifd[0xA003] = 60  # ExifImageHeight
    exif_ifd[0x920A] = 35.0  # FocalLength
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[0x0002] = (52.5, 30.0, 0.0)  # GPSLatitude (privacy)
    return exif.tobytes()


@pytest.fixture()
def jpeg_with_metadata(tmp_path, pixels, icc_profile):
    """Fixed JPEG asset carrying ICC + EXIF (incl. GPS/privacy fields)."""
    path = tmp_path / "fixed_source.jpg"
    Image.fromarray(pixels).save(
        path, format="JPEG", exif=_source_exif(), icc_profile=icc_profile
    )
    return path


@pytest.fixture()
def png_with_metadata(tmp_path, pixels, icc_profile):
    """Fixed PNG asset carrying ICC + EXIF."""
    path = tmp_path / "fixed_source.png"
    Image.fromarray(pixels).save(
        path, format="PNG", exif=_source_exif(), icc_profile=icc_profile
    )
    return path


@pytest.fixture()
def plain_jpeg(tmp_path, pixels):
    """Fixed JPEG asset without ICC and without EXIF."""
    path = tmp_path / "plain.jpg"
    Image.fromarray(pixels).save(path, format="JPEG")
    return path


@pytest.fixture()
def corrupt_exif_jpeg(tmp_path, pixels, icc_profile):
    """Fixed JPEG asset whose EXIF block is corrupted (ICC stays intact)."""
    path = tmp_path / "corrupt_exif.jpg"
    Image.fromarray(pixels).save(
        path, format="JPEG", exif=_source_exif(), icc_profile=icc_profile
    )
    data = bytearray(path.read_bytes())
    header = data.find(b"Exif\x00\x00")
    assert header != -1
    # destroy the TIFF magic number inside the EXIF block
    data[header + 8 : header + 10] = b"\xff\xff"
    path.write_bytes(bytes(data))
    return path


@pytest.fixture()
def duplicate_tag_jpeg(tmp_path, pixels):
    """Fixed JPEG asset whose IFD0 contains the Make tag twice."""
    entry1 = struct.pack(">HHI", 0x010F, 2, 2) + b"A\x00\x00\x00"  # Make="A"
    entry2 = struct.pack(">HHI", 0x010F, 2, 2) + b"B\x00\x00\x00"  # Make="B"
    ifd0 = struct.pack(">H", 2) + entry1 + entry2 + struct.pack(">I", 0)
    tiff = b"MM" + struct.pack(">H", 42) + struct.pack(">I", 8) + ifd0
    exif_bytes = b"Exif\x00\x00" + tiff

    path = tmp_path / "duplicate_tags.jpg"
    Image.fromarray(pixels).save(path, format="JPEG", exif=exif_bytes)
    return path


@pytest.fixture()
def multipage_tiff(tmp_path, pixels):
    """Fixed two-page TIFF asset with EXIF on the pages."""
    path = tmp_path / "multipage.tiff"
    first = Image.fromarray(pixels)
    second = Image.fromarray(np.flipud(pixels))
    exif = Image.Exif()
    exif[0x010F] = "FixedMake"
    first.save(
        path, format="TIFF", save_all=True, append_images=[second], exif=exif.tobytes()
    )
    return path


# ---------------------------------------------------------------------------
# requirement 1: explicit preservation of ICC + allowlisted EXIF, with a
# queryable discard report
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_icc_and_allowlisted_exif(
    tmp_path, jpeg_with_metadata, icc_profile
):
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)

    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    with Image.open(out) as result:
        assert result.info["icc_profile"] == icc_profile
        exif = result.getexif()
        assert exif[0x010F] == "FixedMake"  # allowlisted tag kept
        assert exif[0x0110] == "FixedModel"

    report = preserver.report
    assert report.icc_present and report.icc_preserved
    assert "Make" in report.kept_exif
    assert report.source_format == "JPEG"
    assert report.target_format == "PNG"


def test_discard_report_is_queryable(tmp_path, jpeg_with_metadata):
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)

    # GIF supports neither EXIF nor ICC -> everything must be reported
    out = tmp_path / "out.gif"
    iio.imwrite(out, image, preserve_metadata=preserver)

    report = preserver.report
    assert not report.ok
    assert not report.icc_preserved

    icc_drops = report.dropped_names(kind="icc")
    assert icc_drops == ["ICC profile"]
    assert report.by_reason("does not support ICC")

    exif_drops = report.by_reason("does not support EXIF")
    dropped_names = {entry.name for entry in exif_drops}
    assert {"Make", "Model", "Orientation"} <= dropped_names
    # every dropped entry carries kind and reason
    assert all(entry.kind in ("exif", "icc") for entry in report.dropped)
    assert all(entry.reason for entry in report.dropped)

    with Image.open(out) as result:
        assert "icc_profile" not in result.info
        assert not dict(result.getexif())


def test_allowlist_filters_nonlisted_tags(tmp_path, jpeg_with_metadata):
    preserver = MetadataPreserver.from_source(
        jpeg_with_metadata, exif_allowlist=["Make"]
    )
    image = iio.imread(jpeg_with_metadata)
    out = tmp_path / "out.jpg"
    iio.imwrite(out, image, preserve_metadata=preserver)

    with Image.open(out) as result:
        exif = result.getexif()
        assert exif[0x010F] == "FixedMake"
        assert 0x0110 not in exif  # Model was not allowlisted

    report = preserver.report
    assert report.kept_exif == ("Make",)
    assert "Model" in report.dropped_names()
    assert report.by_reason("not in EXIF allowlist")


def test_explicit_write_kwargs_override_preserver(
    tmp_path, jpeg_with_metadata, icc_profile
):
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)
    other_icc = b"explicitly-passed-icc"

    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver, icc_profile=other_icc)

    with Image.open(out) as result:
        assert result.info["icc_profile"] == other_icc != icc_profile


def test_write_kwargs_standalone(tmp_path, jpeg_with_metadata):
    """write_kwargs can be merged into a write call manually."""
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)

    out = tmp_path / "out.png"
    kwargs = preserver.write_kwargs(format="PNG", shape=image.shape[:2])
    assert set(kwargs) == {"icc_profile", "exif"}
    iio.imwrite(out, image, **kwargs)

    with Image.open(out) as result:
        assert result.getexif()[0x010F] == "FixedMake"


# ---------------------------------------------------------------------------
# requirement 2: applied orientation -> standard orientation on write,
# pixel-dimension fields updated; default write behavior unchanged
# ---------------------------------------------------------------------------


def test_applied_orientation_writes_standard_orientation_and_updates_dims(
    tmp_path, jpeg_with_metadata
):
    preserver = MetadataPreserver.from_source(
        jpeg_with_metadata, orientation_applied=True
    )
    rotated = iio.imread(jpeg_with_metadata, rotate=True)
    assert rotated.shape == (40, 60, 3)  # 60x40 rotated by 90 deg

    cropped = rotated[5:-5, 5:-5]  # edit the pixels (crop)
    out = tmp_path / "out.png"
    iio.imwrite(out, cropped, preserve_metadata=preserver)

    with Image.open(out) as result:
        exif = result.getexif()
        assert exif[0x0112] == 1  # standard orientation written back
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        assert exif_ifd[0xA002] == 50  # ExifImageWidth matches new pixels
        assert exif_ifd[0xA003] == 30  # ExifImageHeight matches new pixels

    report = preserver.report
    assert report.orientation_normalized
    assert "Orientation" in report.kept_exif

    # the edited pixels survive the roundtrip exactly (lossless PNG)
    assert np.array_equal(iio.imread(out), cropped)


def test_orientation_untouched_when_not_applied(tmp_path, jpeg_with_metadata):
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)  # no rotate -> pixels untouched

    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    with Image.open(out) as result:
        exif = result.getexif()
        assert exif[0x0112] == 6  # original orientation preserved
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        assert exif_ifd[0xA002] == 40  # dimensions untouched
        assert exif_ifd[0xA003] == 60

    assert not preserver.report.orientation_normalized


def test_default_write_behavior_unchanged(tmp_path, jpeg_with_metadata):
    """Without a preserver, writes behave exactly as before (no metadata)."""
    image = iio.imread(jpeg_with_metadata)
    out = tmp_path / "out.png"
    iio.imwrite(out, image)

    with Image.open(out) as result:
        assert "icc_profile" not in result.info
        assert 0x010F not in result.getexif()


def test_preserve_metadata_type_checked(tmp_path, jpeg_with_metadata):
    image = iio.imread(jpeg_with_metadata)
    with pytest.raises(TypeError, match="MetadataPreserver"):
        iio.imwrite(tmp_path / "out.png", image, preserve_metadata=object())


# ---------------------------------------------------------------------------
# requirement 3: corrupt EXIF, missing ICC, cross-format, duplicate tags,
# privacy exclusion, multi-frame input
# ---------------------------------------------------------------------------


def test_corrupt_exif_is_reported_and_icc_survives(
    tmp_path, corrupt_exif_jpeg, icc_profile
):
    preserver = MetadataPreserver.from_source(corrupt_exif_jpeg)
    assert preserver.report.exif_corrupt

    image = iio.imread(corrupt_exif_jpeg)
    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    report = preserver.report
    assert report.exif_corrupt
    assert report.icc_preserved  # corrupt EXIF does not affect the ICC profile
    assert report.kept_exif == ()
    assert report.by_reason("corrupt")
    assert "EXIF block" in report.dropped_names(kind="exif")
    assert any("corrupt" in note for note in report.notes)

    with Image.open(out) as result:
        assert result.info["icc_profile"] == icc_profile
        assert not dict(result.getexif())


def test_missing_icc_is_reported(tmp_path, plain_jpeg):
    preserver = MetadataPreserver.from_source(plain_jpeg)
    assert not preserver.report.icc_present

    image = iio.imread(plain_jpeg)
    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    report = preserver.report
    assert not report.icc_present
    assert not report.icc_preserved
    assert "icc_profile" not in report.dropped_names(kind="icc")
    with Image.open(out) as result:
        assert "icc_profile" not in result.info


def test_cross_format_jpeg_to_png(tmp_path, jpeg_with_metadata, icc_profile):
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)

    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    report = preserver.report
    assert report.source_format == "JPEG"
    assert report.target_format == "PNG"
    assert report.icc_preserved
    assert "Make" in report.kept_exif

    with Image.open(out) as result:
        assert result.info["icc_profile"] == icc_profile
        assert result.getexif()[0x010F] == "FixedMake"


def test_cross_format_png_to_jpeg(tmp_path, png_with_metadata, icc_profile):
    preserver = MetadataPreserver.from_source(png_with_metadata)
    image = iio.imread(png_with_metadata)

    out = tmp_path / "out.jpg"
    iio.imwrite(out, image, preserve_metadata=preserver)

    with Image.open(out) as result:
        assert result.info["icc_profile"] == icc_profile
        assert result.getexif()[0x010F] == "FixedMake"


def test_duplicate_tags_are_reported_and_deduplicated(tmp_path, duplicate_tag_jpeg):
    preserver = MetadataPreserver.from_source(duplicate_tag_jpeg)
    assert preserver.report.duplicates == ("Make",)

    image = iio.imread(duplicate_tag_jpeg)
    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    with Image.open(out) as result:
        # a single, deterministic value is written (first occurrence wins,
        # matching Pillow's parsing semantics)
        assert result.getexif()[0x010F] == "A"

    assert "Make" in preserver.report.kept_exif


def test_privacy_fields_excluded_by_default(tmp_path, jpeg_with_metadata):
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)

    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    report = preserver.report
    privacy_drops = report.by_reason("privacy")
    dropped_names = {entry.name for entry in privacy_drops}
    assert {"GPSLatitude", "UserComment", "Artist"} <= dropped_names

    with Image.open(out) as result:
        exif = result.getexif()
        assert 0x9286 not in exif  # UserComment
        assert 0x013B not in exif  # Artist
        assert not dict(exif.get_ifd(ExifTags.IFD.GPSInfo))  # no GPS


def test_privacy_fields_kept_when_exclusion_disabled(tmp_path, jpeg_with_metadata):
    preserver = MetadataPreserver.from_source(
        jpeg_with_metadata,
        exif_allowlist=[*DEFAULT_EXIF_ALLOWLIST, "UserComment", "GPSLatitude"],
        exclude_privacy=False,
    )
    image = iio.imread(jpeg_with_metadata)

    out = tmp_path / "out.png"
    iio.imwrite(out, image, preserve_metadata=preserver)

    with Image.open(out) as result:
        exif = result.getexif()
        assert exif[0x9286] == "private note"
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        assert tuple(gps[0x0002]) == (52.5, 30.0, 0.0)

    assert "UserComment" in preserver.report.kept_exif
    assert "GPSLatitude" in preserver.report.kept_exif


def test_multiframe_source_capture_is_explicit(tmp_path, multipage_tiff):
    preserver = MetadataPreserver.from_source(multipage_tiff)
    report = preserver.report
    assert report.source_frame == 0
    assert any("2 frames" in note for note in report.notes)

    # an explicit frame can be selected
    preserver = MetadataPreserver.from_source(multipage_tiff, frame=1)
    assert preserver.report.source_frame == 1

    # out-of-range frames are a clear error
    with pytest.raises(ValueError, match="out of range"):
        MetadataPreserver.from_source(multipage_tiff, frame=5)


def test_multiframe_write_notes_primary_frame(tmp_path, jpeg_with_metadata):
    preserver = MetadataPreserver.from_source(jpeg_with_metadata)
    image = iio.imread(jpeg_with_metadata)
    batch = np.stack([image, np.flipud(image)], axis=0)

    out = tmp_path / "out.png"  # written as APNG (2 frames)
    iio.imwrite(out, batch, preserve_metadata=preserver)

    report = preserver.report
    assert any("primary frame" in note for note in report.notes)

    props = iio.improps(out, index=...)
    assert props.n_images == 2
    with Image.open(out) as result:
        result.seek(0)
        assert result.getexif()[0x010F] == "FixedMake"


# ---------------------------------------------------------------------------
# requirement 4: fixed assets, pixel checks after editing, field report
# ---------------------------------------------------------------------------


def test_edited_pixels_and_field_report(tmp_path, jpeg_with_metadata):
    """Crop a rotated photo; pixels and updated fields must both be right."""
    preserver = MetadataPreserver.from_source(
        jpeg_with_metadata, orientation_applied=True
    )
    rotated = iio.imread(jpeg_with_metadata, rotate=True)
    edited = rotated[10:30, 5:45].copy()
    edited[0, 0] = [255, 0, 0]  # a deliberate pixel edit

    out = tmp_path / "edited.png"
    iio.imwrite(out, edited, preserve_metadata=preserver)

    # pixels: exactly the edited array
    assert np.array_equal(iio.imread(out), edited)

    # fields: standard orientation + dimensions of the edited image
    with Image.open(out) as result:
        exif = result.getexif()
        assert exif[0x0112] == 1
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        assert exif_ifd[0xA002] == edited.shape[1]
        assert exif_ifd[0xA003] == edited.shape[0]

    # report: full field-level account of the write
    report = preserver.report
    assert set(report.kept_exif) == {
        "Make",
        "Model",
        "FocalLength",
        "Orientation",
        "ExifImageWidth",
        "ExifImageHeight",
    }
    assert {entry.name for entry in report.dropped} == {
        "Artist",
        "UserComment",
        "GPSLatitude",
    }
