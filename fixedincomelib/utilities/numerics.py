import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    # ------------------------------------------------------------------
    # Internal helpers
    #
    # The whole class is built on one observation: for a fixed x (or a fixed
    # integration window) both the interpolated value and its integral are
    # *linear* in the ordinate vector `values`.  So every public method is a
    # thin wrapper around a weight vector w with
    #
    #       f(x)                  = w(x)      . values
    #       \int_s^e f(u) du      = W(s, e)   . values
    #
    # and the two gradients are then simply w(x) and W(s, e) themselves.
    # Deriving the weights once keeps value and sensitivity exactly
    # consistent by construction -- they cannot drift apart.
    # ------------------------------------------------------------------

    def _bucket_index(self, x: float) -> int:
        """Index of the ordinate that governs `x`.

        Convention (left-continuous piecewise constant, FLAT wings):

            x <= axis1[0]                    -> values[0]      (left wing)
            axis1[i-1] < x <= axis1[i]       -> values[i]
            x >  axis1[-1]                   -> values[-1]     (right wing)

        `np.searchsorted(..., side='left')` returns the smallest i with
        axis1[i] >= x, which is exactly the bucket above; clipping to
        [0, n-1] folds the two flat wings into the same expression, so no
        special-casing is needed.
        """
        idx = int(np.searchsorted(self.axis1_, x, side='left'))
        return min(max(idx, 0), self.length_ - 1)

    def _support_bounds(self) -> tuple:
        """Half-open x-range (lo_i, hi_i] on which each ordinate is active.

        With axis1 = [1, 3, 5, 7] this gives

            i = 0 : (-inf, 1]      left wing merged with the first node
            i = 1 : (1, 3]
            i = 2 : (3, 5]
            i = 3 : (5, +inf)      last bucket merged with the right wing

        The supports are disjoint and cover the whole real line, so the
        integral of f over any window is just the sum of values[i] times the
        overlap length of [start, end] with support i.
        """
        lo = np.empty(self.length_, dtype=float)
        hi = np.empty(self.length_, dtype=float)

        lo[0] = -np.inf
        lo[1:] = self.axis1_[:-1]

        hi[:-1] = self.axis1_[:-1]
        hi[-1] = np.inf

        return lo, hi

    def _integration_weights(self, start_x: float, end_x: float) -> np.ndarray:
        r"""Weight vector W with \int_{start_x}^{end_x} f(u) du = W . values.

        W[i] is the length of the overlap between the integration window and
        the support of ordinate i.  Reversed windows are handled by the usual
        antisymmetry of the integral.
        """
        if start_x > end_x:
            return -self._integration_weights(end_x, start_x)

        lo, hi = self._support_bounds()

        # Overlap of [start_x, end_x] with (lo_i, hi_i]; the closed/half-open
        # distinction is irrelevant for a Lebesgue integral (single points
        # carry zero measure), so plain clipping is exact here.
        left = np.maximum(lo, start_x)
        right = np.minimum(hi, end_x)
        return np.maximum(right - left, 0.0)

    # ------------------------------------------------------------------
    # Interpolator1D interface
    # ------------------------------------------------------------------

    def interpolate(self, x: float) -> float:
        return float(self.values_[self._bucket_index(x)])

    def integrate(self, start_x: float, end_x: float) -> float:
        return float(np.dot(self._integration_weights(start_x, end_x), self.values_))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        """d f(x) / d values -- the indicator vector of the active bucket.

        f is piecewise constant in x but *linear* in the ordinates, so the
        gradient is exact (not an approximation): 1 in the governing bucket,
        0 elsewhere.
        """
        grad = np.zeros(self.length_, dtype=float)
        grad[self._bucket_index(x)] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        r"""d/d values of \int_{start_x}^{end_x} f(u) du -- the overlap lengths."""
        return self._integration_weights(start_x, end_x)


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')