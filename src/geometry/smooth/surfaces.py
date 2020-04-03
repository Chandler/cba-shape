from jax import vmap, grad, hessian, jacfwd
from abc import abstractmethod
import math
import numpy as np
from jax import np as jnp


class Surface(object):
    """ A parametric surface in three dimensions
    """

    def __init__(self):
        # The vectorized coordinate functions
        self.X = vmap(self._X)  # (N x 2) -> N
        self.Y = vmap(self._Y)  # (N x 2) -> N
        self.Z = vmap(self._Z)  # (N x 2) -> N

        # The vectorized gradients of the individual coordinate
        # functions of f
        self.dX = vmap(grad(self._X))  # (N x 2) -> N x 2
        self.dY = vmap(grad(self._Y))  # (N x 2) -> N x 2
        self.dZ = vmap(grad(self._Z))  # (N x 2) -> N x 2

        # Second partial deriviatives
        # hesx, hesy, hesz where each is (N x 2 x 2)
        self.hessianXYZ = vmap(hessian(self._f))  # (N x 2) -> 3 x N x 2 x 2

    @abstractmethod
    def _X(self, uv):
        """ The coordinate function X
        """
        pass

    @abstractmethod
    def _Y(self, uv):
        """ The coordinate function Y
        """
        pass

    @abstractmethod
    def _Z(self, uv):
        """ The coordinate function Z
        """
        pass

    @abstractmethod
    def u_range(self):
        """ The domain of u, expressed as a floating point
        range, inclusive

        Returns
        -------
        [float, float]
        """
        pass

    @abstractmethod
    def v_range(self):
        """ The domain of u, expressed as a floating point
        range, inclusive

        Returns
        -------
        [float, float]
        """
        pass

    def sample_domain(self, step):
        """ Linearly sample the 2D domain of the surface
        """
        u_start, u_end = surface.u_range()
        v_start, v_end = surface.v_range()
        ulist = np.linspace(u_start, u_end, step)
        vlist = np.linspace(v_start, v_end, step)
        coords = []
        for u in ulist:
            for v in vlist:
                coords.append([u, v])
        return np.array(coords)

    def _f(self, uv):
        """ The immersion of the surface into space

        Also known as the parameterization

        see `self.f` for vectorized version

        Parameters
        ----------
        uv: [float, float]

        Returns
        -------
        [float, float, float]
        """
        return [self._X(uv), self._Y(uv), self._Z(uv)]

    def f(self, uv):
        """ The immersion of the surface into space

        Also known as the parameterization

        Vectorized function

        Parameters
        ----------
        uv: nd.array, N x 2, float32

        Returns
        -------
        nd.array, N x 3
        """
        return np.array([self.X(uv), self.Y(uv), self.Z(uv)]).swapaxes(0, 1)


class MongePatch(Surface):
    def _X(self, uv):
        return uv[0]

    def _Y(self, uv):
        return uv[1]


class Sphere(Surface):
    def _X(self, uv):
        return jnp.cos(uv[0]) * jnp.sin(uv[1])

    def _Y(self, uv):
        return jnp.sin(uv[0]) * jnp.sin(uv[1])

    def _Z(self, uv):
        return jnp.cos(uv[1])

    def u_range(self):
        return [0.0, 2 * math.pi]

    def v_range(self):
        return [0.0, math.pi]


class Ellipsoid(Surface):
    def __init__(self, a=3, b=2.2, c=1.9):
        self.a = a
        self.b = b
        self.c = c
        self.abc = (self.a, self.b, self.c)
        super().__init__()

    def _X(self, uv):
        w, B = uv
        a, b, c = self.abc
        top = a ** 2 - (b ** 2 * jnp.sin(B) ** 2) - (c ** 2 * jnp.cos(B) ** 2)
        return a * jnp.cos(w) * (jnp.sqrt(top) / jnp.sqrt(a ** 2 - c ** 2))

    def _Y(self, uv):
        w, B = uv
        a, b, c = self.abc
        return b * jnp.cos(B) * jnp.sin(w)

    def _Z(self, uv):
        w, B = uv
        a, b, c = self.abc
        top = (a ** 2 * jnp.sin(w) ** 2) + (b ** 2 * jnp.cos(w) ** 2) - c ** 2
        return c * jnp.sin(B) * (jnp.sqrt(top) / jnp.sqrt(a ** 2 - c ** 2))

    def u_range(self):
        return [-math.pi, math.pi]

    def v_range(self):
        return [-math.pi, math.pi]


class MonkeySaddle(MongePatch):
    def _Z(self, uv):
        u, v = uv
        return (u ** 3 - 3 * u * (v ** 2)) / 10

    def u_range(self):
        return [-math.pi, math.pi]

    def v_range(self):
        return [-math.pi, math.pi]
