"""Dark Channel Prior (DCP) single-image dehazing.

Implementation of "Single Image Haze Removal Using Dark Channel Prior"
(He, Sun, Tang; CVPR 2009) with guided-filter transmission refinement
(He, Sun, Tang; "Guided Image Filtering", ECCV 2010 / PAMI 2013).

Convention used throughout this package: images are float32 arrays in [0, 1]
with shape (H, W, 3) in **BGR** channel order (as read by OpenCV).
"""

from .dark_channel import dark_channel
from .atmospheric_light import estimate_atmospheric_light
from .guided_filter import guided_filter
from .transmission import estimate_transmission, refine_transmission
from .dehaze import dehaze

__all__ = [
    "dark_channel",
    "estimate_atmospheric_light",
    "guided_filter",
    "estimate_transmission",
    "refine_transmission",
    "dehaze",
]
