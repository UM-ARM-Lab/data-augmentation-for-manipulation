from sympy import Matrix, sin, cos, symbols

import numpy as np

x, y, z, roll, pitch, yaw = symbols('x y z roll pitch yaw')
params = [x, y, z, roll, pitch, yaw]
translation = params[:3]
angles = params[3:]
sx = sin(roll)
sy = sin(pitch)
sz = sin(yaw)
cx = cos(roll)
cy = cos(pitch)
cz = cos(yaw)
m00 = cy * cz
m01 = (sx * sy * cz) - (cx * sz)
m02 = (cx * sy * cz) + (sx * sz)
m10 = cy * sz
m11 = (sx * sy * sz) + (cx * cz)
m12 = (cx * sy * sz) - (sx * cz)
m20 = -sy
m21 = sx * cy
m22 = cx * cy
matrix = Matrix([[m00, m01, m02, x],
                 [m10, m11, m12, y],
                 [m20, m21, m22, z],
                 [0, 0, 0, 1]])
params = Matrix(params)
print(matrix.shape, params.shape)
jacobians = []
for s in params:
    jacobians.append(matrix.diff(s))
print(jacobians)
from sympy.printing.pycode import pycode
pycode(matrix.diff(roll))

from sympy.abc import rho, phi
X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
Y = Matrix([rho, phi])
print(X.shape, Y.shape)
X.jacobian(Y)
