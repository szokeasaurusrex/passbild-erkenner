from enum import Enum
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter, correlate, label
from PIL import Image
import numpy as np
import numpy.typing as npt


class FailReason(Enum):
    (
        DIMENSIONS,
        GRAYSCALE,
        SOLID_BACKGROUND,
        BRIGHT_BACKGROUND,
        BIG_HEAD,
        SMALL_HEAD,
    ) = range(6)


def reason_message(reason: FailReason) -> str:
    return {
        FailReason.DIMENSIONS: "Ein richtiges Passbild muss 45mm hoch und 35mm breit sein.",
        FailReason.GRAYSCALE: "Ein richtiges Passbild muss in Farbe sein.",
        FailReason.SOLID_BACKGROUND: "Ein richtiges Passbild muss einen einfarbigen Hintergrund haben.",
        FailReason.BRIGHT_BACKGROUND: "Ein richtiges Passbild soll einen hellen Hintergrund haben.",
        FailReason.BIG_HEAD: "Dein Kopf darf nicht mehr als 36 mm hoch sein.",
        FailReason.SMALL_HEAD: "Dein Kopf ist zu klein. Er soll ungefähr 2/3 des Bildes einnehmen.",
    }[reason]


def to_grayscale(im: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return im @ np.asarray([0.299, 0.587, 0.114])


def gaussian_derivative(
    col_values: npt.NDArray[np.float64], sigma: float
) -> npt.NDArray[np.float64]:
    row_values = col_values.T
    return (
        row_values
        / (sigma**3 * np.sqrt(2 * np.pi))
        * np.exp(-(row_values**2 + col_values**2) / (2 * sigma**2))
    )


def gaussian_derivative_mask(sigma: float) -> npt.NDArray[np.float64]:
    """
    Gaussian derivative mask for the rows (transpose for columns).
    """
    center_idx = int(np.ceil(2 * sigma))
    mask_size = 2 * center_idx + 1
    col_values = np.zeros(shape=(mask_size, mask_size))
    col_values[:] = np.linspace(-center_idx, center_idx, 2 * center_idx + 1)

    return gaussian_derivative(col_values, sigma)


def shrink_im(im: npt.NDArray[np.float64], max_height: int) -> npt.NDArray[np.float64]:
    shrinked_im = im.copy()
    while shrinked_im.shape[0] > max_height:
        shrinked_im = gaussian_filter(shrinked_im, sigma=1, axes=(0, 1))
        shrinked_im = shrinked_im[::2, ::2, ...]

    return shrinked_im


class PassbildVerifier:
    EDGE_THRESHOLD = 0.1

    def __init__(self, im: npt.NDArray[np.float64]):
        self.im = im

    def correct_dimensions(self) -> bool:
        correct_width = np.round(self.im.shape[0] / 45 * 35)

        return self.im.shape[1] == correct_width

    def is_color_image(self) -> bool:
        try:
            is_rgb = self.im.shape[2] == 3
        except IndexError:
            is_rgb = False

        return (
            is_rgb
            and not (self.im[..., 0] == self.im[..., 1]).all()
            and not (self.im[..., 1] == self.im[..., 2]).all()
        )

    def bright_background(self) -> bool:
        average_bg_graylevel = np.mean(self.shrinked_greyscale_im[self.background_mask])

        return average_bg_graylevel > 0.6

    def solid_background(self) -> bool:
        corner_color_diff = np.max(
            np.abs(self.shrinked_im[0, 0] - self.shrinked_im[0, -1])
        )

        return (
            corner_color_diff < 0.2
            and not ((self.edges > self.EDGE_THRESHOLD) & self.background_mask).any()
        )

    def head_big_enough(self) -> bool:
        return self.head_ratio > 0.6

    def head_small_enough(self) -> bool:
        return self.head_ratio < 0.8

    def verify(self) -> bool:
        checks = {
            self.correct_dimensions: FailReason.DIMENSIONS,
            self.is_color_image: FailReason.GRAYSCALE,
            self.solid_background: FailReason.SOLID_BACKGROUND,
            self.bright_background: FailReason.BRIGHT_BACKGROUND,
            self.head_big_enough: FailReason.SMALL_HEAD,
            self.head_small_enough: FailReason.BIG_HEAD,
        }

        for check_func, fail_reason in checks.items():
            if not check_func():
                return False, fail_reason

        return True, None

    @property
    def shrinked_im(self) -> npt.NDArray[np.float64]:
        try:
            return self._shrinked_im
        except AttributeError:
            self._shrinked_im = shrink_im(self.im, 128)
            return self._shrinked_im

    @property
    def shrinked_greyscale_im(self) -> npt.NDArray[np.float64]:
        return to_grayscale(self.shrinked_im)

    @property
    def edges(self) -> npt.NDArray[np.float64]:
        try:
            return self._edges
        except AttributeError:
            derivative_mask = gaussian_derivative_mask(1)
            self._edges = np.sqrt(
                correlate(self.shrinked_greyscale_im, derivative_mask) ** 2
                + correlate(self.shrinked_greyscale_im, derivative_mask.T) ** 2
            )
            return self._edges

    @property
    def background_mask(self) -> npt.NDArray[np.float64]:
        """
        Background mask for the shrinked image
        """
        try:
            return self._background_mask
        except AttributeError:
            background_region_width = int(self.edges.shape[1] * 0.1)
            background_region_height = int(self.edges.shape[0] * 0.6)

            self._background_mask = np.zeros(self.edges.shape, dtype=bool)
            self._background_mask[
                :background_region_height, :background_region_width
            ] = True
            self._background_mask[
                :background_region_height, -background_region_width:
            ] = True

            return self._background_mask

    @property
    def head_ratio(self) -> float:
        try:
            return self._head_ratio
        except AttributeError:
            non_edges = self.edges <= self.EDGE_THRESHOLD

            components, _ = label(non_edges)

            l_background = components == components[0, 0]
            r_background = components == components[0, -1]
            background = l_background | r_background

            head_top_row = np.argmin(np.min(background, axis=1))

            # Shadow down
            shadow = background.copy()
            for i, _ in enumerate(shadow[:-1]):
                shadow[i + 1] = shadow[i + 1] & shadow[i]
            shadow = shadow ^ background

            neckline = np.argmax(shadow.sum(axis=1))
            head_width = np.max(np.sum(1 - background[:neckline], axis=1))

            underestimate = (neckline - head_top_row) / background.shape[0]
            overestimate = head_width / (2 / 3 * background.shape[0])

            self._head_ratio = (underestimate + overestimate) / 2

            return self._head_ratio


def is_passbild(im: Image.Image) -> Tuple[bool, Optional[FailReason]]:
    return PassbildVerifier(np.asarray(im, dtype=np.float64) / 255).verify()
