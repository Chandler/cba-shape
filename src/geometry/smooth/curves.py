from jax import vmap, grad, jacfwd
from abc import abstractmethod
from typing import List
from jax import np as jnp
import math
import scipy
import numpy as np
from linear_algebra.la_util import normalize_vectors, normalize_vector


class Curve(object):
    """ Abstract base class for Plane and Space curves
    """

    @abstractmethod
    def t_range(self) -> List[float]:
        """ The domain of t, expressed as a floating point
        range, inclusive

        Returns
        -------
        [float, float]
        """
        pass

    def t_scale(self, step):
        """ The distance between two coordinate samples when
        a given step size is used.
        """
        t_start, t_end = self.t_range()
        return np.abs(np.subtract(t_start, t_end)) / step

    def t_linspace(self, step):
        t_start, t_end = self.t_range()
        return np.linspace(t_start, t_end, step)

    coordinates = t_linspace

    def discrete_integral(self, W, step):
        """ 
        W is a scalar field defined on the curve
            W : (t) -> S  (vectorized)
        
        This function integrates the field over the curve
        """
        values = W(self.t_linspace(step))
        return np.nansum(values * self.t_scale(step))


class PlaneCurve(Curve):
    def __init__(self):
        """ A paramterized curve in the plane
        """

        # The vectorized coordinate function for u
        self.x = vmap(self._x)

        # The vectorized coordinate function for v
        self.y = vmap(self._y)

        # The vectorized deriviative of x
        self.dx = vmap(grad(self._x))

        # The vectorized derivative of y
        self.dy = vmap(grad(self._y))

    @abstractmethod
    def _x(self, time: float) -> float:
        """ The coordinate function x
        """
        pass

    @abstractmethod
    def _y(self, time: float) -> float:
        """ The coordinate function y
        """
        pass

    def _f(self, t):
        """ The immersion of the curve into the plane

        see `self.f` for vectorized version

        Parameters
        ----------
        t: [float]

        Returns
        -------
        [float, float, float]
        """
        return [self._x(t), self._y(t)]

    def f(self, t):
        """ The immersion of the curve into the plane

        vectorized

        Parameters
        ----------
        t: nd.array, N, float32

        Returns
        -------
        nd.array, N x 3
        """
        return np.array([self.x(t), self.y(t)]).swapaxes(0, 1)


class SpaceCurve(Curve):
    """ A parameterized curve in 3D space
    """

    def __init__(self):
        # The vectorized coordinate functions
        self.X = vmap(self._X)  # (N x 2) -> N
        self.Y = vmap(self._Y)  # (N x 2) -> N
        self.Z = vmap(self._Z)  # (N x 2) -> N

        # _T,_N,_B are unvectorized functions that yield the unit tangent, normal
        # and bi-normal vectors of the curve at t.
        self._T = lambda t: normalize_vector(jnp.array(jacfwd(self._f)(t)))
        self._N = lambda t: normalize_vector(jnp.array(jacfwd(self._T)(t)))
        self._B = lambda t: normalize_vector(jnp.cross(self._T(t), self._N(t)))

        # Vectorized TNB
        self.T = vmap(self._T)
        self.N = vmap(self._N)
        self.B = vmap(self._B)

    @abstractmethod
    def _X(self, time: float) -> float:
        """ The coordinate function x
        """
        pass

    @abstractmethod
    def _Y(self, time: float) -> float:
        """ The coordinate function y
        """
        pass

    @abstractmethod
    def _Z(self, time: float) -> float:
        """ The coordinate function y
        """
        pass

    def _f(self, t):
        """ The immersion of the curve into space

        see `self.f` for vectorized version

        Parameters
        ----------
        t: [float]

        Returns
        -------
        [float, float, float]
        """
        return [self._X(t), self._Y(t), self._Z(t)]

    def f(self, t):
        """ The immersion of the curve into space

        vectorized

        Parameters
        ----------
        t: nd.array, N, float32

        Returns
        -------
        nd.array, N x 3
        """
        return np.array([self.X(t), self.Y(t), self.Z(t)]).swapaxes(0, 1)


class Equator(PlaneCurve):
    """ The equator of the unit sphere
    """

    def _x(self, t):
        return t

    def _y(self, t):
        return math.pi / 2.0

    def t_range(self):
        return [0, 2 * math.pi]


class Helix(SpaceCurve):
    """ https://en.wikipedia.org/wiki/Helix
    """

    def __init__(self, domain, R=1.0, h=1.0):
        """
        Parameters
        ----------
        domain: [float, float]
        R: float
            The Radius of the helix
        h: float
            The pitch of a helix is the height of one complete helix turn
            measured parallel to the axis of the helix.
        """
        self.domain = domain
        self.R = R
        self.h = h
        super().__init__()

    def t_range(self):
        return self.domain

    def _X(self, t):
        return self.h * t

    def _Y(self, t):
        return self.R * jnp.cos(t)

    def _Z(self, t):
        return self.R * jnp.sin(t)


class Tractrix(PlaneCurve):
    def __init__(self, a):
        self.a = a

    def t_range(self):
        return [0.001, 1.999]

    def _x(self, t):
        return self.a * jnp.sin(t)

    def _y(self, t):
        return self.a * (jnp.cos(t) + jnp.log(jnp.tan(t / 2.0)))


class TrefoilKnot(SpaceCurve):
    def t_range(self):
        return [0.001, math.pi * 2]

    def _X(self, t):
        return jnp.sin(t) + 2 * jnp.sin(2 * t)

    def _Y(self, t):
        return jnp.cos(t) - 2 * jnp.cos(2 * t)

    def _Z(self, t):
        return -1.0 * jnp.sin(3 * t)

class Line2D(PlaneCurve):
    def __init__(self, r0=[1,1], v=[0,1]):
        self.r0 = jnp.array(r0)
        self.v  = jnp.array(v)
        super().__init__()

    def t_range(self):
        return [0, 10.0]

    def line(self, t):
        return self.r0 + t*self.v

    def _x(self, t):
        return self.line(t)[0]

    def _y(self, t):
        return self.line(t)[1]

class Line3D(SpaceCurve):
    def __init__(self, r0=[1,1,1], v=[1,0.5,0]):
        self.r0 = jnp.array(r0)
        self.v  = jnp.array(v)
        super().__init__()

    def t_range(self):
        return [0, 1.0]

    def line(self, t):
        return self.r0 + t*self.v

    def _X(self, t):
        return self.line(t)[0]

    def _Y(self, t):
        return self.line(t)[1]

    def _Z(self, t):
        return self.line(t)[2]

