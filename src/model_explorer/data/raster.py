from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


class RasterDecodeUnavailable(RuntimeError):
    """Raised when no optional raster decoder can read a requested product."""


@dataclass(frozen=True)
class RasterWindow:
    width: int
    height: int
    values: tuple[tuple[float, ...], ...]


def read_raster_window(
    path: str | Path,
    *,
    x: int = 0,
    y: int = 0,
    width: int | None = None,
    height: int | None = None,
    fallback: float = 0.0,
) -> RasterWindow:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RasterDecodeUnavailable(
            "Pillow is required to decode JP2 raster products; install a Pillow build with JPEG2000 support"
        ) from exc

    raster_path = Path(path)
    try:
        with Image.open(raster_path) as image:
            image_width, image_height = image.size
            window_width = image_width - int(x) if width is None else int(width)
            window_height = image_height - int(y) if height is None else int(height)
            if window_width <= 0 or window_height <= 0:
                raise ValueError("raster window width and height must be positive")
            if x < 0 or y < 0 or x + window_width > image_width or y + window_height > image_height:
                raise ValueError("raster window is outside image bounds")
            region = image.crop((int(x), int(y), int(x) + window_width, int(y) + window_height))
            data_reader = getattr(region, "get_flattened_data", region.getdata)
            raw_values = tuple(data_reader())
    except RasterDecodeUnavailable:
        raise
    except Exception as exc:
        raise RasterDecodeUnavailable(f"failed to decode raster product {raster_path}: {exc}") from exc

    rows: list[tuple[float, ...]] = []
    for row_index in range(window_height):
        start = row_index * window_width
        row_values = raw_values[start : start + window_width]
        rows.append(tuple(_finite_float(value, fallback=fallback) for value in row_values))
    return RasterWindow(width=window_width, height=window_height, values=tuple(rows))


def _finite_float(value, *, fallback: float) -> float:
    if isinstance(value, tuple):
        value = value[0] if value else fallback
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if isfinite(numeric) else float(fallback)
