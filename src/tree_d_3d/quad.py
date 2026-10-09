"""The square cells of the quadtrees."""

from __future__ import annotations


class Quad:
    """Represents a quadrant in the quadtree."""

    xmid: float
    ymid: float
    half: float

    def __init__(self, x_mid: float, y_mid: float, half_size: float) -> None:
        """Initialize a quadrant.

        Args:
            x_mid, y_mid: Center coordinates
            half_size: Half-width/height of the quadrant
        """
        self.xmid = x_mid
        self.ymid = y_mid
        self.half = half_size

    def contains(self, x: float, y: float) -> bool:
        """Check if point is within this quadrant."""
        return (
            x >= self.xmid - self.half
            and x <= self.xmid + self.half
            and y >= self.ymid - self.half
            and y <= self.ymid + self.half
        )

    def northwest(self) -> Quad:
        """Create northwest child quadrant."""
        return Quad(
            self.xmid - self.half / 2.0,
            self.ymid + self.half / 2.0,
            self.half / 2.0,
        )

    def northeast(self) -> Quad:
        """Create northeast child quadrant."""
        return Quad(
            self.xmid + self.half / 2.0,
            self.ymid + self.half / 2.0,
            self.half / 2.0,
        )

    def southwest(self) -> Quad:
        """Create southwest child quadrant."""
        return Quad(
            self.xmid - self.half / 2.0,
            self.ymid - self.half / 2.0,
            self.half / 2.0,
        )

    def southeast(self) -> Quad:
        """Create southeast child quadrant."""
        return Quad(
            self.xmid + self.half / 2.0,
            self.ymid - self.half / 2.0,
            self.half / 2.0,
        )
