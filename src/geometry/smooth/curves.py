from jax import vmap, grad, jacfwd
from abc import abstractmethod
from typing import List
from jax import np as jnp


class PlaneCurve(object):
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
    def t_range(self) -> List[float]:
        """ The domain of t, expressed as a floating point
        range, inclusive

        Returns
        -------
        [float, float]
        """
        pass

    @abstractmethod
    def _x(self, time: float) -> float:
        """ The coordinate function x
        """
        pass

    @abstractmethod
    def _v(self, time: float) -> float:
        """ The coordinate function y
        """
        pass

    def f(self, t):
        """ The immersion of the curve into the plane

        Vectorized function

        Parameters
        ----------
        t: nd.array, N, float32

        Returns
        -------
        nd.array, N x 2
        """
        return np.array([self.u(t), self.v(t)]).swapaxes(0, 1)


class SpaceCurve(object):
    """ A parameterized curve in 3D space
    """

    def __init__(self):
        # The first derivative of f
        self.df = jacfwd(self.f)

        # The second derivative of f
        self.ddf = jacfwd(jacfwd(self.f))

    @abstractmethod
    def t_range(self):
        """ The domain of t, expressed as a floating point
        range, inclusive

        Returns
        -------
        [float, float]
        """
        pass

    @abstractmethod
    def f(self, t):
        """ The immersion of the curve into space

        Vectorized function

        Parameters
        ----------
        t: nd.array, N, float32

        Returns
        -------
        nd.array, N x 3
        """
        pass

    def N(time):
        """ Unit normal
        """
        return vector_normalize(self.df(time))

    def T(time):
        """ Unit tangent
        """
        return vector_normalize(self.ddf(time))

    def B(self, time):
        """ Unit bi-normal
        """

    def frenet_frame(self, time):
        """ The orthonormal basis vectors of the frenet frame
        at time, vectorized

        Returns
        nd.array, N x 3
        """
        return jnp.array([self.T(time), self.N(time), self.B(time)]).swapaxes(0, 1)


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

    def f(self, t):
        return jnp.array(
            [self.h * t, self.R * jnp.cos(t), self.R * jnp.sin(t)], dtype=jnp.float32
        ).swapaxes(0, 1)
