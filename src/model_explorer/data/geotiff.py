from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


class GeoTiffDecodeUnavailable(RuntimeError):
    """Raised when no available decoder can read a GeoTIFF window."""


@dataclass(frozen=True)
class GeoTiffWindow:
    width: int
    height: int
    values: tuple[tuple[float, ...], ...]
    nodata_mask: tuple[tuple[bool, ...], ...]
    resolution_m: float
    projection: dict[str, Any]
    bounds: dict[str, int]
    reader_backend: str


def read_geotiff_window(
    path: str | Path,
    *,
    x: int = 0,
    y: int = 0,
    width: int | None = None,
    height: int | None = None,
    resolution_m: float,
    projection: dict[str, Any] | None = None,
    nodata_value: float | None = None,
    fallback: float = 0.0,
) -> GeoTiffWindow:
    """Read a bounded GeoTIFF window with the dependencies available in this repo.

    This adapter deliberately keeps geospatial metadata explicit in the manifest.
    Pillow can decode the pixel values in our smoke tests and many float TIFFs, but
    it is not a full GeoTIFF metadata reader. When richer CRS/window semantics are
    needed, Stage23.2A should route to the GeoTIFF dependency blocker instead of
    silently inventing projection data.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise GeoTiffDecodeUnavailable("Pillow is required for the basic GeoTIFF reader") from exc

    raster_path = Path(path)
    try:
        # Stage23.2A reads trusted, manifest-pinned lunar GeoTIFF products.
        # These source rasters are intentionally much larger than Pillow's
        # default decompression-bomb threshold, while the runner only consumes
        # bounded windows and records the product hash separately.
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(raster_path) as image:
            image_width, image_height = image.size
            window_width = image_width - int(x) if width is None else int(width)
            window_height = image_height - int(y) if height is None else int(height)
            if window_width <= 0 or window_height <= 0:
                raise ValueError("GeoTIFF window width and height must be positive")
            if int(x) < 0 or int(y) < 0 or int(x) + window_width > image_width or int(y) + window_height > image_height:
                raise ValueError("GeoTIFF window is outside image bounds")
            region = image.crop((int(x), int(y), int(x) + window_width, int(y) + window_height))
            data_reader = getattr(region, "get_flattened_data", region.getdata)
            raw_values = tuple(data_reader())
    except Exception as exc:
        raise GeoTiffDecodeUnavailable(f"failed to decode GeoTIFF product {raster_path}: {exc}") from exc

    rows: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    for row_index in range(window_height):
        start = row_index * window_width
        row_values = raw_values[start : start + window_width]
        numeric_row: list[float] = []
        mask_row: list[bool] = []
        for value in row_values:
            numeric = _finite_float(value, fallback=fallback)
            numeric_row.append(numeric)
            mask_row.append(_is_nodata(numeric, nodata_value))
        rows.append(tuple(numeric_row))
        masks.append(tuple(mask_row))

    return GeoTiffWindow(
        width=window_width,
        height=window_height,
        values=tuple(rows),
        nodata_mask=tuple(masks),
        resolution_m=float(resolution_m),
        projection=dict(projection or {}),
        bounds={"x": int(x), "y": int(y), "width": window_width, "height": window_height},
        reader_backend="pillow_basic_tiff",
    )


def _finite_float(value: Any, *, fallback: float) -> float:
    if isinstance(value, tuple):
        value = value[0] if value else fallback
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if isfinite(numeric) else float(fallback)


def _is_nodata(value: float, nodata_value: float | None) -> bool:
    if nodata_value is None:
        return False
    return value == float(nodata_value)
