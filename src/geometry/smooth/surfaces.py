from jax import vmap, grad, hessian, jacfwd
from abc import abstractmethod
import math
import numpy as np
from jax import np as jnp
from geometry.smooth.curves import PlaneCurve
from linear_algebra.la_util import normalize_vectors
import quadpy
import scipy


class Surface(object):
    """ A parametric surface in three dimensions.
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

        # The first derivative of f
        # This is equal to self.jacobian_matrix
        self.df = lambda uv: np.array(vmap(jacfwd(self._f))(uv)).swapaxes(0, 1)

        dX = grad(self._X)  # (N x 2) -> N x 2
        dY = grad(self._Y)  # (N x 2) -> N x 2
        dZ = grad(self._Z)  # (N x 2) -> N x 2

        # The Gauss Map, unvectorized
        def _N(uv):
            dfdu, dfdv = jnp.array(jacfwd(self._f)(uv)).swapaxes(0, 1)
            n = jnp.cross(dfdu, dfdv)
            return n / jnp.linalg.norm(n)

        # The vectorized guass map
        self.N = vmap(_N)

        # The vectorized differential of the Guass map
        # This is called the Weingarten map
        self.dN = vmap(jacfwd(_N))

    # ======================================================
    # All abstract methods required to implement this surface

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

    # ======================================================
    # The fundamental parameterizations of the surface, based
    # on the individual coordinate functions

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

    # ======================================================
    # Methods that deal with discretely sampling the
    # parameters space

    def u_scale(self, step):
        """ The distance between two u coordinate samples when
        a given step size is used.
        """
        u_start, u_end = self.u_range()
        return np.abs(np.subtract(u_start, u_end)) / step

    def v_scale(self, step):
        """ The distance between two v coordinate samples when
        a given step size is used.
        """
        v_start, v_end = self.v_range()
        return np.abs(np.subtract(v_start, v_end)) / step

    def u_linspace(self, step):
        u_start, u_end = self.u_range()
        return np.linspace(u_start, u_end, step)

    def v_linspace(self, step):
        v_start, v_end = self.v_range()
        return np.linspace(v_start, v_end, step)

    def coordinates(self, step):
        """ Sample the surface domain with a given
        step size and return a list of coordinates
        """
        U = self.u_linspace(step)
        V = self.v_linspace(step)
        # Some wild way to get every pair of U,V
        return np.flip(np.stack(np.meshgrid(U, V), -1).reshape(-1, 2), axis=1)

    def discrete_double_integral(self, W, step):
        """ 
        W is a scalar field defined on the surface
            W : (uv) -> S  (vectorized)
        
        This function integrates the field over the surface
        by sampling it on the surface then summing and scaling
        the result
        """
        # alternate: np.nansum(W(self.coordinates(step)) * self.u_scale(step) * self.v_scale(step))

        # construct 2-D integrand
        data = W(self.coordinates(step)).reshape((step, step))

        # do a 1-D integral over every row
        rows = np.zeros(step)
        for i in range(step):
            rows[i] = np.trapz(data[i, :], self.u_linspace(step))

        return np.trapz(rows, self.v_linspace(step))

    def shape_operator(self, uv):
        """ Compute the Shape Operator

        df * S = dN

        """
        # The differential of the function
        df = np.array(self.df(uv))

        # The differential of the Gauss map
        dN = np.array(self.dN(uv))

        # df•S = dN
        #
        # S = df^-1 • dN
        S = np.einsum("nij,njk->nik", np.linalg.pinv(df), dN)

        return S

    def principal_direction(self, uv):
        return np.linalg.eig(self.shape_operator(uv))[1]

    def principal_curvature(self, uv):
        return np.linalg.eig(self.shape_operator(uv))[0]

    def gaussian_curvature(self, uv):
        return np.linalg.det(self.shape_operator(uv))

    def total_gaussian_curvature(self, step):
        def k(uv):
            return self.gaussian_curvature(uv) * self.area_element()(uv)

        return self.discrete_double_integral(k, step)

    def normals(self, uv):
        """ Compute unnormalized surface normals
        as the cross product of the tangent vectors at each
        point on the surface.
        """
        dfdu, dfdv = self.gradients(uv)
        return np.cross(dfdu, dfdv)

    def unit_normals(self, uv):
        return normalize_vectors(self.normals(uv))

    def jacobian_matrix(self, uv):
        """  https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

        In vector calculus, the Jacobian matrix of a vector-valued function in
        several variables is the matrix of all its first-order partial derivatives. 
        When this matrix is square, that is, when the function takes the same number 
        of variables as input as the number of vector components of its output,

        Vectorized

        Returns
        -------
        nd.array, N x 3 x 2
            A list of 3x2 jacobian matrices
        """

        # Note: this is equal to self.df, but
        # this construction does a better job of reminding
        # you what's happening

        dX = np.array(self.dX(uv))  # dXdu, dXdv
        dY = np.array(self.dY(uv))  # dYdu, dYdv
        dZ = np.array(self.dZ(uv))  # dZdu, dZdv
        return np.array([dX, dY, dZ]).swapaxes(0, 1)

    def gradients(self, uv):
        """ The gradients of f with respect to u and v
        
        Each gradient is a vector field on the surface

        Together the u and v gradients span the tangent plane
        at each point on the surface

        vectorized

        """
        J = self.jacobian_matrix(uv)
        dfdu = J[:, :, 0]
        dfdv = J[:, :, 1]
        # (N x 3), (N x 3)
        return dfdu, dfdv

    def first_fundamental_form(self, uv):
        """
        https://en.wikipedia.org/wiki/First_fundamental_form#Example
        The coefficients of the first fundamental form may be
        found by taking the dot product of the partial derivatives.
        
        vectorized

        Returns
        -------
        [nd.array, nd.array, nd.array]
            E,F, and G for each coordinate uv
        """
        dfdu, dfdv = self.gradients(uv)
        E = np.einsum("ij,ij->i", dfdu, dfdu)
        F = np.einsum("ij,ij->i", dfdu, dfdv)
        G = np.einsum("ij,ij->i", dfdv, dfdv)
        return E, F, G

    def line_element(self, curve: PlaneCurve):
        """ Given a curve on this surface,
        produce the line element. A function
        that can be integrated to get arc length.

        A scalar field defined on a curve on the surface

        An integrand that can be integrated to get arc length

        Returns
        -------
        f(t: nd.array, N) -> nd.array, N
        """

        def f(t):
            uv = curve.f(t)
            E, F, G = self.first_fundamental_form(uv)
            dx = curve.dx(t)
            dy = curve.dy(t)
            return np.sqrt(E * dx * dx + 2 * F * dx * dy + G * dy * dy)

        return f

    def area_element(self):
        """ https://en.wikipedia.org/wiki/Volume_element#Area_element_of_a_surface

        A scalar field on the surface.

        An integrand that can be integrated to get surface area

        Returns
        -------
        f(t: nd.array, N) -> nd.array, N
        """

        def f(uv):
            E, F, G = self.first_fundamental_form(uv)
            return np.sqrt(E * G - F * F)

        return f

    # ======================================================
    # Properties of the surface
    #

    def surface_area(self, step):
        """
        Returns
        -------
        float
        """
        dA = self.area_element()
        return self.discrete_double_integral(dA, step)

    def arc_length(self, curve: PlaneCurve, step):
        """ Compute the length of a curve on this
        surface. The curve is defined in the parameter space
        and used to produce the line element, which is integrated
        over the curve.

        Returns
        -------
        float
        """
        ds = self.line_element(curve)
        return curve.discrete_integral(ds, step)


# ======================================================
# Specific surfaces
#


class MongePatch(Surface):
    def _X(self, uv):
        return uv[0]

    def _Y(self, uv):
        return uv[1]


class Sphere(Surface):
    def __init__(self, r=1):
        self.r = r
        super().__init__()

    def _X(self, uv):
        return self.r * jnp.cos(uv[0]) * jnp.sin(uv[1])

    def _Y(self, uv):
        return self.r * jnp.sin(uv[0]) * jnp.sin(uv[1])

    def _Z(self, uv):
        return self.r * jnp.cos(uv[1])

    def u_range(self):
        return [0.001, 2 * math.pi]

    def v_range(self):
        return [0.0001, math.pi]


class EllipsoidSpecial(Surface):
    """https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid#Triaxial_coordinate_systems
       https://math.stackexchange.com/questions/205915/parametrization-for-the-ellipsoids
    """

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
        return [0.001, math.pi]

    def v_range(self):
        return [-math.pi, math.pi]


class Ellipsoid(Surface):
    def __init__(self, a=3, b=2.2, c=1.9):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def _X(self, uv):
        u, v = uv
        return self.a * jnp.cos(u) * jnp.sin(v)

    def _Y(self, uv):
        u, v = uv
        return self.b * jnp.sin(u) * jnp.sin(v)

    def _Z(self, uv):
        u, v = uv
        return self.c * jnp.cos(v)

    def u_range(self):
        return [0.001, 2 * math.pi]

    def v_range(self):
        return [0.001, math.pi]


class MonkeySaddle(MongePatch):
    def _Z(self, uv):
        u, v = uv
        return (u ** 3 - 3 * u * (v ** 2)) / 10

    def u_range(self):
        return [-math.pi, math.pi]

    def v_range(self):
        return [-math.pi, math.pi]


class Torus(Surface):
    def __init__(self, a=8, b=3, c=7):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def _X(self, uv):
        return (self.a + (self.b * jnp.cos(uv[1]))) * jnp.cos(uv[0])

    def _Y(self, uv):
        return (self.a + (self.b * jnp.cos(uv[1]))) * jnp.sin(uv[0])

    def _Z(self, uv):
        return self.c * jnp.sin(uv[1])

    def u_range(self):
        return [0, math.pi * 2]

    def v_range(self):
        return [0, math.pi * 2]
