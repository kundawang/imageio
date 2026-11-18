# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Opt-in metadata preservation pipeline for the Pillow plugin.

This module implements an explicit, opt-in way to carry color (ICC) and
allowlisted EXIF metadata from a source image over to an image written by
:class:`imageio.plugins.pillow.PillowPlugin`. The default read/write behavior
of imageio is unchanged; nothing is preserved unless a ``MetadataPreserver``
is created and passed to the writer explicitly.

Typical usage::

    import imageio.v3 as iio
    from imageio.plugins.pillow_metadata import MetadataPreserver

    preserver = MetadataPreserver.from_source("photo.jpg", orientation_applied=True)
    img = iio.imread("photo.jpg", rotate=True)
    img = img[10:-10, 10:-10]  # edit pixels, e.g. crop

    iio.imwrite("photo_cropped.jpg", img, preserve_metadata=preserver)
    print(preserver.report)          # human-readable summary
    print(preserver.report.dropped)  # queryable discard report

Every entry that is not written out is recorded in
:attr:`MetadataPreserver.report` together with the reason it was discarded
(unsupported output format, not allowlisted, privacy exclusion, corrupt
source data, ...), so callers can inspect exactly what happened.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import ExifTags, Image

__all__ = [
    "MetadataPreserver",
    "MetadataReport",
    "DroppedEntry",
    "DEFAULT_EXIF_ALLOWLIST",
    "PRIVACY_TAGS",
]

_EXIF_HEADER = b"Exif\x00\x00"

# name -> tag id for the main/Exif IFDs (first id wins for ambiguous names)
_TAG_NAME_TO_ID: Dict[str, int] = {}
for _tag_id, _tag_name in ExifTags.TAGS.items():
    _TAG_NAME_TO_ID.setdefault(_tag_name, _tag_id)

_GPS_NAME_TO_ID: Dict[str, int] = {}
for _tag_id, _tag_name in ExifTags.GPSTAGS.items():
    _GPS_NAME_TO_ID.setdefault(_tag_name, _tag_id)

#: EXIF tags preserved by default (matched by tag name). Anything not in this
#: list is discarded and reported unless a custom allowlist is supplied.
DEFAULT_EXIF_ALLOWLIST: Tuple[str, ...] = (
    "Orientation",
    "ImageWidth",
    "ImageLength",
    "XResolution",
    "YResolution",
    "ResolutionUnit",
    "Software",
    "DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "Make",
    "Model",
    "LensMake",
    "LensModel",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "PhotographicSensitivity",
    "FocalLength",
    "FocalLengthIn35mmFilm",
    "ExposureBiasValue",
    "ExposureProgram",
    "MeteringMode",
    "Flash",
    "WhiteBalance",
    "ColorSpace",
    "ExifImageWidth",
    "ExifImageHeight",
    "ImageDescription",
    "Copyright",
)

#: Tags that may contain personal/sensitive information. They are dropped
#: when ``exclude_privacy=True`` (the default), even if allowlisted. Every
#: tag inside the GPS IFD is treated as privacy-sensitive as well.
PRIVACY_TAGS: Tuple[str, ...] = (
    "GPSInfo",
    "UserComment",
    "MakerNote",
    "Artist",
    "CameraOwnerName",
    "OwnerName",
    "BodySerialNumber",
    "CameraSerialNumber",
    "LensSerialNumber",
    "ImageUniqueID",
)

# Pillow writer formats known to embed EXIF / ICC metadata. Anything else
# (GIF, BMP, PPM, ...) silently discards such metadata, so the preserver
# reports the discard instead of letting it happen silently.
_FORMATS_WITH_EXIF = frozenset({"JPEG", "PNG", "TIFF", "WEBP", "MPO", "HEIF", "AVIF"})
_FORMATS_WITH_ICC = frozenset({"JPEG", "PNG", "TIFF", "WEBP", "HEIF", "AVIF"})

_FORMAT_ALIASES = {"JPG": "JPEG", "TIF": "TIFF"}

# pixel-dimension related fields, updated when the orientation was applied
# to the pixels (i.e. the image was physically rotated/cropped). The second
# element marks fields holding the height (rather than the width).
_DIMENSION_TAGS = (
    ("ImageWidth", False),
    ("ImageLength", True),
    ("ExifImageWidth", False),
    ("ExifImageHeight", True),
)


@dataclass(frozen=True)
class DroppedEntry:
    """A single metadata item that was not written to the output.

    Attributes
    ----------
    kind : str
        Either ``"exif"`` or ``"icc"``.
    name : str
        The tag/field name, e.g. ``"GPSLatitude"`` or ``"ICC profile"``.
    reason : str
        Why the entry was discarded, e.g. ``"GIF does not support EXIF"``.
    """

    kind: str
    name: str
    reason: str


@dataclass
class MetadataReport:
    """Queryable result of a metadata preservation pass.

    Attributes
    ----------
    source_format, target_format : str or None
        Pillow format names of the source and the output (e.g. ``"JPEG"``).
    source_frame : int
        Index of the source frame the metadata was captured from.
    icc_present : bool
        Whether the source image carried an ICC profile.
    icc_preserved : bool
        Whether the ICC profile was (or will be) written to the output.
    exif_present : bool
        Whether the source image carried EXIF metadata.
    exif_corrupt : bool
        Whether the source EXIF data was present but could not be parsed.
    kept_exif : tuple of str
        Names of EXIF fields that are written to the output.
    dropped : tuple of DroppedEntry
        Every metadata item that was discarded, with a reason.
    duplicates : tuple of str
        Names of tags that occurred multiple times within one IFD of the
        source. The first occurrence's value is used (Pillow semantics).
    orientation_normalized : bool
        Whether the output Orientation was reset to standard (1) because the
        orientation had already been applied to the pixels.
    notes : tuple of str
        Additional human-readable remarks (multi-frame input, ...).
    """

    source_format: Optional[str] = None
    target_format: Optional[str] = None
    source_frame: int = 0
    icc_present: bool = False
    icc_preserved: bool = False
    exif_present: bool = False
    exif_corrupt: bool = False
    kept_exif: Tuple[str, ...] = ()
    dropped: Tuple[DroppedEntry, ...] = ()
    duplicates: Tuple[str, ...] = ()
    orientation_normalized: bool = False
    notes: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True if nothing had to be discarded."""
        return not self.dropped

    def dropped_names(self, kind: str = None) -> List[str]:
        """Names of discarded entries, optionally filtered by ``kind``."""
        return [
            entry.name for entry in self.dropped if kind is None or entry.kind == kind
        ]

    def by_reason(self, reason: str) -> List[DroppedEntry]:
        """All discarded entries matching ``reason`` (substring match)."""
        return [entry for entry in self.dropped if reason in entry.reason]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            f"MetadataReport({self.source_format} -> {self.target_format}, "
            f"icc={'preserved' if self.icc_preserved else 'not preserved'}, "
            f"kept={len(self.kept_exif)}, dropped={len(self.dropped)})"
        ]
        for entry in self.dropped:
            lines.append(f"  dropped {entry.kind}:{entry.name} ({entry.reason})")
        return "\n".join(lines)


def _normalize_format(format: Optional[str]) -> Optional[str]:
    if format is None:
        return None
    fmt = str(format).upper()
    return _FORMAT_ALIASES.get(fmt, fmt)


def _iter_ifd_tag_ids(exif_bytes: bytes) -> Dict[str, List[int]]:
    """Walk a raw EXIF blob and collect tag ids per IFD.

    Returns a dict with keys ``"main"``, ``"exif"`` and ``"gps"`` mapping to
    the list of tag ids found in that IFD (in file order, duplicates kept).
    Raises ``ValueError`` if the blob does not look like well-formed EXIF
    data; this is how corrupt EXIF is detected.
    """

    data = bytes(exif_bytes)
    if data[:6] == _EXIF_HEADER:
        data = data[6:]

    if len(data) < 8:
        raise ValueError("EXIF data too short for a TIFF header")

    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError("EXIF data has an invalid byte-order marker")

    (magic,) = struct.unpack_from(endian + "H", data, 2)
    if magic != 42:
        raise ValueError("EXIF data has an invalid TIFF magic number")

    (ifd0_offset,) = struct.unpack_from(endian + "I", data, 4)

    result: Dict[str, List[int]] = {"main": [], "exif": [], "gps": []}
    visited = set()

    def walk(offset: int, bucket: str, depth: int) -> None:
        if depth > 4 or offset in visited:
            raise ValueError("EXIF IFD chain is cyclic or too deep")
        visited.add(offset)
        if offset + 2 > len(data):
            raise ValueError("EXIF IFD offset points outside the data")
        (count,) = struct.unpack_from(endian + "H", data, offset)
        entries_start = offset + 2
        entries_end = entries_start + count * 12
        if entries_end + 4 > len(data):
            raise ValueError("EXIF IFD entries run past the end of the data")

        for i in range(count):
            entry = data[entries_start + i * 12 : entries_start + (i + 1) * 12]
            (tag_id,) = struct.unpack_from(endian + "H", entry, 0)
            result[bucket].append(tag_id)
            if bucket == "main" and tag_id == 0x8769:  # ExifIFDPointer
                (sub_offset,) = struct.unpack_from(endian + "I", entry, 8)
                walk(sub_offset, "exif", depth + 1)
            elif bucket == "main" and tag_id == 0x8825:  # GPSInfoIFDPointer
                (sub_offset,) = struct.unpack_from(endian + "I", entry, 8)
                walk(sub_offset, "gps", depth + 1)

    walk(ifd0_offset, "main", 0)
    return result


def _open_source(source):
    """Return ``(image, owns_image)`` for the supported source types."""
    if isinstance(source, Image.Image):
        return source, False
    if isinstance(source, (str, Path)):
        return Image.open(source), True
    # bytes or file-like
    return Image.open(source), True


class MetadataPreserver:
    """Capture ICC/EXIF metadata from a source and re-apply it on write.

    Parameters
    ----------
    exif_allowlist : sequence of str, optional
        Names of EXIF tags that may be carried over to the output. If None
        (default), :data:`DEFAULT_EXIF_ALLOWLIST` is used. Tags are matched
        by their Pillow/ExifTags name, e.g. ``"Make"`` or ``"FocalLength"``.
    exclude_privacy : bool
        If True (default), privacy-sensitive fields (GPS coordinates, owner
        names, serial numbers, user comments, maker notes; see
        :data:`PRIVACY_TAGS`) are always discarded, even if allowlisted.
    orientation_applied : bool
        Set this to True if the source image was read with its EXIF
        orientation applied to the pixels (e.g. ``iio.imread(...,
        rotate=True)``). The output then gets the standard orientation (1)
        and pixel-dimension fields (ImageWidth/ImageLength,
        ExifImageWidth/ExifImageHeight) are updated to the new shape.

    Notes
    -----
    The pipeline is strictly opt-in: imageio's default write behavior is
    unchanged unless a preserver is passed to the writer explicitly, either
    via ``iio.imwrite(..., preserve_metadata=preserver)`` or by merging
    ``preserver.write_kwargs(...)`` into the write call manually.

    """

    def __init__(
        self,
        *,
        exif_allowlist: Optional[Sequence[str]] = None,
        exclude_privacy: bool = True,
        orientation_applied: bool = False,
    ) -> None:
        self.exif_allowlist = frozenset(
            exif_allowlist if exif_allowlist is not None else DEFAULT_EXIF_ALLOWLIST
        )
        self.exclude_privacy = bool(exclude_privacy)
        self.orientation_applied = bool(orientation_applied)

        self._source_format: Optional[str] = None
        self._source_frame: int = 0
        self._n_source_frames: int = 1
        self._icc: Optional[bytes] = None
        self._exif_corrupt: bool = False
        self._exif_present: bool = False
        # per-IFD captured tags: name -> value
        self._main_ifd: Dict[str, Any] = {}
        self._exif_ifd: Dict[str, Any] = {}
        self._gps_ifd: Dict[str, Any] = {}
        self._duplicates: Tuple[str, ...] = ()
        self._notes: List[str] = []

        self.report = MetadataReport()

    @classmethod
    def from_source(
        cls,
        source: Union[str, Path, bytes, Image.Image],
        *,
        frame: int = 0,
        **kwargs,
    ) -> "MetadataPreserver":
        """Create a preserver and capture metadata from ``source``.

        ``source`` may be a path, bytes, a file-like object, or an already
        open ``PIL.Image``. ``frame`` selects the frame for multi-frame
        sources (metadata is captured from a single frame; the report notes
        this when the source holds more than one frame).
        """
        preserver = cls(**kwargs)
        preserver.capture(source, frame=frame)
        return preserver

    def capture(
        self,
        source: Union[str, Path, bytes, Image.Image],
        *,
        frame: int = 0,
    ) -> "MetadataPreserver":
        """Capture ICC profile and EXIF tags from ``source``."""
        image, owned = _open_source(source)
        try:
            n_frames = getattr(image, "n_frames", 1)
            if not 0 <= frame < n_frames:
                raise ValueError(
                    f"Frame {frame} out of range: source has {n_frames} frame(s)."
                )
            if getattr(image, "is_animated", False) or n_frames > 1:
                image.seek(frame)

            self._source_format = image.format
            self._source_frame = frame
            self._n_source_frames = n_frames
            self._notes = []
            if n_frames > 1:
                self._notes.append(
                    f"Source has {n_frames} frames; metadata captured from "
                    f"frame {frame} only."
                )

            icc = image.info.get("icc_profile")
            self._icc = bytes(icc) if icc else None

            self._main_ifd = {}
            self._exif_ifd = {}
            self._gps_ifd = {}
            self._duplicates = ()
            self._exif_corrupt = False

            raw_exif = image.info.get("exif")
            if raw_exif:
                try:
                    tag_ids = _iter_ifd_tag_ids(raw_exif)
                except ValueError:
                    self._exif_corrupt = True
                    self._notes.append(
                        "Source EXIF data is corrupt and cannot be parsed; "
                        "all EXIF fields will be discarded."
                    )
                else:
                    self._duplicates = self._find_duplicates(tag_ids)

            if not self._exif_corrupt:
                try:
                    self._read_exif(image)
                except Exception:
                    # Pillow failed to parse EXIF that our walker accepted
                    self._exif_corrupt = True
                    self._main_ifd = {}
                    self._exif_ifd = {}
                    self._gps_ifd = {}
                    self._notes.append(
                        "Source EXIF data could not be parsed by Pillow; "
                        "all EXIF fields will be discarded."
                    )

            self._exif_present = bool(
                self._main_ifd or self._exif_ifd or self._gps_ifd or raw_exif
            )
        finally:
            if owned:
                image.close()

        self._refresh_report()
        return self

    @staticmethod
    def _find_duplicates(tag_ids: Dict[str, List[int]]) -> Tuple[str, ...]:
        duplicates = []
        for ids in tag_ids.values():
            seen = set()
            for tag_id in ids:
                if tag_id in seen:
                    name = ExifTags.TAGS.get(
                        tag_id, ExifTags.GPSTAGS.get(tag_id, f"0x{tag_id:04X}")
                    )
                    if name not in duplicates:
                        duplicates.append(name)
                seen.add(tag_id)
        return tuple(duplicates)

    def _read_exif(self, image: Image.Image) -> None:
        exif = image.getexif()
        if not exif:
            return

        self._main_ifd = {
            ExifTags.TAGS.get(tag_id, f"0x{tag_id:04X}"): value
            for tag_id, value in dict(exif).items()
            if tag_id not in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo)
        }

        try:
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        except Exception:
            exif_ifd = {}
        self._exif_ifd = {
            ExifTags.TAGS.get(tag_id, f"0x{tag_id:04X}"): value
            for tag_id, value in dict(exif_ifd).items()
        }

        try:
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except Exception:
            gps_ifd = {}
        self._gps_ifd = {
            ExifTags.GPSTAGS.get(tag_id, f"0x{tag_id:04X}"): value
            for tag_id, value in dict(gps_ifd).items()
        }

    def _refresh_report(self, **updates) -> None:
        current = self.report
        merged = dict(
            source_format=self._source_format,
            source_frame=self._source_frame,
            icc_present=self._icc is not None,
            exif_present=self._exif_present,
            exif_corrupt=self._exif_corrupt,
            duplicates=self._duplicates,
            notes=tuple(self._notes),
            target_format=current.target_format,
            icc_preserved=current.icc_preserved,
            kept_exif=current.kept_exif,
            dropped=current.dropped,
            orientation_normalized=current.orientation_normalized,
        )
        merged.update(updates)
        self.report = MetadataReport(**merged)

    def write_kwargs(
        self,
        *,
        format: Optional[str] = None,
        shape: Optional[Tuple[int, int]] = None,
        n_frames: int = 1,
    ) -> Dict[str, Any]:
        """Build the Pillow save kwargs that carry the preserved metadata.

        Parameters
        ----------
        format : str
            The output format (Pillow format name or common alias, e.g.
            ``"JPEG"``/``"jpg"``). Entries the format cannot embed are
            discarded and recorded in :attr:`report`.
        shape : (int, int), optional
            The ``(height, width)`` of the image being written. Needed to
            update pixel-dimension EXIF fields when ``orientation_applied``
            is True.
        n_frames : int
            Number of frames being written. If greater than 1, a note is
            added to the report that metadata is applied to the primary
            frame only.

        Returns
        -------
        kwargs : dict
            A dict with ``"icc_profile"`` and/or ``"exif"`` entries suitable
            for ``PIL.Image.save`` / ``imageio.imwrite``. Explicitly passed
            write kwargs should take precedence over these.

        """
        target_format = _normalize_format(format)
        kwargs: Dict[str, Any] = {}
        kept: List[str] = []
        dropped: List[DroppedEntry] = []
        notes = list(self._notes)
        orientation_normalized = False

        # --- ICC profile -------------------------------------------------
        if self._icc is None:
            pass  # nothing to preserve; icc_present=False in the report
        elif target_format in _FORMATS_WITH_ICC:
            kwargs["icc_profile"] = self._icc
        else:
            dropped.append(
                DroppedEntry(
                    "icc",
                    "ICC profile",
                    f"{target_format} does not support ICC profiles",
                )
            )

        # --- EXIF --------------------------------------------------------
        exif_bytes = None
        if self._exif_corrupt:
            dropped.append(
                DroppedEntry("exif", "EXIF block", "source EXIF data is corrupt")
            )
            for name in self._captured_tag_names():
                dropped.append(
                    DroppedEntry("exif", name, "source EXIF data is corrupt")
                )
        elif target_format not in _FORMATS_WITH_EXIF:
            for name in self._captured_tag_names():
                dropped.append(
                    DroppedEntry("exif", name, f"{target_format} does not support EXIF")
                )
        else:
            plan, kept, dropped_exif, plan_notes = self._plan_exif(shape)
            dropped.extend(dropped_exif)
            notes.extend(plan_notes)
            if plan:
                exif_bytes, unserializable = self._serialize_plan(plan)
                for name in unserializable:
                    kept.remove(name)
                    dropped.append(
                        DroppedEntry(
                            "exif", name, "value could not be serialized as EXIF"
                        )
                    )
            if exif_bytes is not None:
                kwargs["exif"] = exif_bytes
            orientation_normalized = self.orientation_applied

        if n_frames > 1:
            notes.append(
                f"Writing {n_frames} frames; preserved metadata is applied "
                "to the primary frame only."
            )

        self._refresh_report(
            target_format=target_format,
            icc_preserved="icc_profile" in kwargs,
            kept_exif=tuple(kept),
            dropped=tuple(dropped),
            orientation_normalized=orientation_normalized,
            notes=tuple(notes),
        )
        return kwargs

    def _captured_tag_names(self) -> List[str]:
        return list(self._main_ifd) + list(self._exif_ifd) + list(self._gps_ifd)

    def _plan_exif(self, shape: Optional[Tuple[int, int]]) -> Tuple[
        List[Tuple[str, int, str, Any]],
        List[str],
        List[DroppedEntry],
        List[str],
    ]:
        """Decide which EXIF tags to write.

        Returns ``(plan, kept, dropped, notes)`` where ``plan`` entries are
        ``(ifd, tag_id, name, value)`` with ``ifd`` in
        ``{"main", "exif", "gps"}``.
        """
        plan: List[Tuple[str, int, str, Any]] = []
        kept: List[str] = []
        dropped: List[DroppedEntry] = []
        notes: List[str] = []

        def consider(ifd: str, name: str, value: Any, name_to_id) -> None:
            if self.exclude_privacy and (name in PRIVACY_TAGS or ifd == "gps"):
                dropped.append(
                    DroppedEntry("exif", name, "excluded as privacy-sensitive")
                )
                return
            if name not in self.exif_allowlist:
                dropped.append(DroppedEntry("exif", name, "not in EXIF allowlist"))
                return
            tag_id = name_to_id.get(name)
            if tag_id is None:
                dropped.append(DroppedEntry("exif", name, "unknown EXIF tag"))
                return
            plan.append((ifd, tag_id, name, value))
            kept.append(name)

        for name, value in self._main_ifd.items():
            consider("main", name, value, _TAG_NAME_TO_ID)
        for name, value in self._exif_ifd.items():
            consider("exif", name, value, _TAG_NAME_TO_ID)
        for name, value in self._gps_ifd.items():
            consider("gps", name, value, _GPS_NAME_TO_ID)

        if self.orientation_applied:
            # pixels are already in standard orientation: write that back.
            # The forced standard orientation supersedes any earlier verdict
            # on a captured Orientation tag (kept or dropped).
            plan = [entry for entry in plan if entry[2] != "Orientation"]
            dropped = [entry for entry in dropped if entry.name != "Orientation"]
            if "Orientation" not in kept:
                kept.append("Orientation")
            plan.append(("main", 0x0112, "Orientation", 1))

            if shape is not None:
                height, width = shape
                kept_set = set(kept)
                for name, is_height in _DIMENSION_TAGS:
                    if name not in kept_set:
                        continue
                    new_value = height if is_height else width
                    plan = [
                        (
                            (entry[0], entry[1], entry[2], new_value)
                            if entry[2] == name
                            else entry
                        )
                        for entry in plan
                    ]
            else:
                notes.append(
                    "Orientation was applied but no output shape was given; "
                    "pixel-dimension EXIF fields were not updated."
                )

        return plan, kept, dropped, notes

    @staticmethod
    def _serialize_plan(
        plan: List[Tuple[str, int, str, Any]],
    ) -> Tuple[Optional[bytes], List[str]]:
        """Serialize the planned tags, dropping unserializable ones.

        Returns ``(exif_bytes, names_of_dropped_tags)``. ``exif_bytes`` is
        None if nothing serializable remains.
        """

        def build(entries) -> bytes:
            exif = Image.Exif()
            for ifd, tag_id, _name, value in entries:
                if ifd == "main":
                    exif[tag_id] = value
                elif ifd == "exif":
                    exif.get_ifd(ExifTags.IFD.Exif)[tag_id] = value
                else:
                    exif.get_ifd(ExifTags.IFD.GPSInfo)[tag_id] = value
            return exif.tobytes()

        try:
            return build(plan), []
        except Exception:
            pass

        # fall back to keeping only individually serializable tags
        kept_entries = []
        dropped_names = []
        for entry in plan:
            try:
                build(kept_entries + [entry])
                kept_entries.append(entry)
            except Exception:
                dropped_names.append(entry[2])

        if not kept_entries:
            return None, dropped_names
        return build(kept_entries), dropped_names
