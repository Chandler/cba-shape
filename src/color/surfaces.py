from geometry.smooth.surfaces import Surface
from color.color_util import CIEXfitSimple, CIEYfitSimple, CIEZfitSimple, visual_range_nm
import numpy as np
from jax import np as jnp

class SpectralCone(Surface):
    def __init__(self, XYZ_to_outputspace=np.eye(3)):
        self.XYZ_to_outputspace = XYZ_to_outputspace
        super().__init__()

    def u_range(self):
        return [0.0, 1]   

    def v_range(self):
        return visual_range_nm

    def _XYZ(self, uv):
        """ The CMFs place a wavelength parameter into a space curve (the spectral locus)
        multiplying by intensity shifts to a position on a surface (the spectral cone)
        """
        intensity, wavelength = uv
        x = intensity * CIEXfitSimple(wavelength)
        y = intensity * CIEYfitSimple(wavelength)
        z = intensity * CIEZfitSimple(wavelength)
        return self.XYZ_to_outputspace(jnp.array([x,y,z]))

    def _X(self, uv):
        return self._XYZ(uv)[0]

    def _Y(self, uv):
        return self._XYZ(uv)[1]

    def _Z(self, uv):
        return self._XYZ(uv)[2]