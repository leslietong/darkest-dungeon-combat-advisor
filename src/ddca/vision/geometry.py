"""Resolution-independent normalized rectangles and pixel mapping."""

from __future__ import annotations

from dataclasses import dataclass

from ddca.vision.errors import InvalidCalibrationError


@dataclass(frozen=True)
class NormalizedRect:
    """Axis-aligned rectangle in unit coordinates of a captured window image.

    `x` and `y` are the top-left corner. Values are typically in ``[0, 1]``
    relative either to the full captured image or to a parent region.
    """

    x: float
    y: float
    width: float
    height: float

    def validate(self, *, context: str) -> None:
        """Reject empty or out-of-range rectangles with a repair hint."""

        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if value != value:  # NaN
                raise InvalidCalibrationError(
                    f"{context}: {name} is NaN. Use finite numbers in the YAML file."
                )
        if self.width <= 0 or self.height <= 0:
            raise InvalidCalibrationError(
                f"{context}: width and height must be positive "
                f"(got width={self.width}, height={self.height})."
            )
        if self.x < 0 or self.y < 0:
            raise InvalidCalibrationError(
                f"{context}: x and y must be >= 0 (got x={self.x}, y={self.y})."
            )
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise InvalidCalibrationError(
                f"{context}: rectangle extends outside 0-1 "
                f"(x={self.x}, y={self.y}, width={self.width}, height={self.height}). "
                "Reduce x+width and y+height so both are at most 1.0."
            )

    def to_pixel(self, image_width: int, image_height: int) -> PixelRect:
        """Map this rectangle onto an image of `image_width` x `image_height` pixels."""

        _validate_image_size(image_width, image_height)
        left = int(round(self.x * image_width))
        top = int(round(self.y * image_height))
        width = int(round(self.width * image_width))
        height = int(round(self.height * image_height))
        left, width = _clamp_span(left, width, image_width)
        top, height = _clamp_span(top, height, image_height)
        if width <= 0 or height <= 0:
            raise InvalidCalibrationError(
                "Normalized rectangle mapped to an empty pixel rectangle "
                f"on a {image_width}x{image_height} image. "
                "Increase the region size in the calibration YAML."
            )
        return PixelRect(left=left, top=top, width=width, height=height)

    def nest(self, local: NormalizedRect) -> NormalizedRect:
        """Interpret `local` as normalized within this rectangle."""

        return NormalizedRect(
            x=self.x + local.x * self.width,
            y=self.y + local.y * self.height,
            width=local.width * self.width,
            height=local.height * self.height,
        )

    def contains(self, inner: NormalizedRect, *, eps: float = 1e-6) -> bool:
        """Return True if `inner` lies inside this rectangle."""

        return (
            inner.x + eps >= self.x
            and inner.y + eps >= self.y
            and inner.x + inner.width <= self.x + self.width + eps
            and inner.y + inner.height <= self.y + self.height + eps
        )


@dataclass(frozen=True)
class PixelRect:
    """Inclusive-origin, exclusive-end rectangle in integer pixels."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Exclusive right edge."""

        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Exclusive bottom edge."""

        return self.top + self.height

    def validate(self, *, context: str) -> None:
        """Reject empty pixel rectangles."""

        if self.width <= 0 or self.height <= 0:
            raise InvalidCalibrationError(
                f"{context}: pixel width and height must be positive "
                f"(got {self.width}x{self.height})."
            )
        if self.left < 0 or self.top < 0:
            raise InvalidCalibrationError(
                f"{context}: pixel left and top must be >= 0 "
                f"(got left={self.left}, top={self.top})."
            )

    def to_normalized(self, image_width: int, image_height: int) -> NormalizedRect:
        """Convert pixel coordinates to unit coordinates of an image."""

        _validate_image_size(image_width, image_height)
        self.validate(context="pixel rectangle")
        if self.right > image_width or self.bottom > image_height:
            raise InvalidCalibrationError(
                f"Pixel rectangle ({self.left},{self.top}) {self.width}x{self.height} "
                f"does not fit in a {image_width}x{image_height} image. "
                "Measure the rectangle inside the captured PNG."
            )
        return NormalizedRect(
            x=self.left / image_width,
            y=self.top / image_height,
            width=self.width / image_width,
            height=self.height / image_height,
        )

    def numpy_slices(self) -> tuple[slice, slice]:
        """Return ``(row_slice, col_slice)`` for NumPy indexing."""

        return slice(self.top, self.bottom), slice(self.left, self.right)

    def contains(self, inner: PixelRect, *, tolerance: int = 1) -> bool:
        """Return True if `inner` lies inside this rectangle.

        `tolerance` allows 1px rounding error from normalized mapping.
        """

        return (
            inner.left >= self.left - tolerance
            and inner.top >= self.top - tolerance
            and inner.right <= self.right + tolerance
            and inner.bottom <= self.bottom + tolerance
        )


def _validate_image_size(image_width: int, image_height: int) -> None:
    if image_width <= 0 or image_height <= 0:
        raise InvalidCalibrationError(
            f"Image size must be positive (got {image_width}x{image_height})."
        )


def _clamp_span(origin: int, length: int, limit: int) -> tuple[int, int]:
    origin = max(0, origin)
    end = min(limit, origin + length)
    origin = min(origin, limit)
    return origin, max(0, end - origin)
