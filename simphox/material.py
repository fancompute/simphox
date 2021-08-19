from .typing import Tuple, Dim2


class Material:
    def __init__(self, name: str, facecolor: Tuple[float, float, float] = None, eps: float = None):
        self.name = name
        self.eps = eps
        self.facecolor = facecolor

    def __str__(self):
        return self.name


class MaterialBlock:
    def __init__(self, dim: Dim2, material: Material):
        """Material block (substrate or waveguide)

        Args:
            dim: Dimension tuple of the form :code:`(x, y)` for the material block
            material: Material for the block
        """
        self.dim = dim
        self.material = material
        self.x = dim[0]
        self.y = dim[1]
        self.eps = self.material.eps


SILICON = Material('Silicon', (0.3, 0.3, 0.3), 3.4784 ** 2)
POLYSILICON = Material('Poly-Si', (0.5, 0.5, 0.5), 3.4784 ** 2)
OXIDE = Material('Oxide', (0.6, 0, 0), 1.4442 ** 2)
NITRIDE = Material('Nitride', (0, 0, 0.7), 1.996 ** 2)
LS_NITRIDE = Material('Low-Stress Nitride', (0, 0.4, 1))
LT_OXIDE = Material('Low-Temp Oxide', (0.8, 0.2, 0.2), 1.4442 ** 2)
ALUMINUM = Material('Aluminum', (0, 0.5, 0))
ALUMINA = Material('Alumina', (0.2, 0, 0.2), 1.75)
ETCH = Material('Etch', (0, 0, 0))


