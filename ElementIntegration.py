# --------------------------------------------------------------------------
# Classes to form local finite element matrices for degree-one Lagrange
# basis functions on triangles.
#
#
#
# Katharine Long, Sep 2020
# For Math 5344
# --------------------------------------------------------------------------

import numpy as np

# --------------------------------------------------------------------------
#
# Class representation of a triangle, which computes the Jacobian and related
# quantities needed in element integrations.
#
class Triangle:
    # Construct with the coordinate pairs of the three vertices
    def __init__(self, A, B, C):
        # Copy points into numpy vectors so we can add/subtract them
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)

        # Now we're going to compute |det(J)| and J^{-T}. We'll need these
        # for element integrations.

        # Form the transpose of the Jacobian of the transformation from the
        # reference right triangle to this triangle
        self.Jt = np.array([self.B-self.A, self.C-self.A])
        # Compute det(J)
        self.detJ = self.Jt[0,0]*self.Jt[1,1]-self.Jt[0,1]*self.Jt[1,0]
        # The area of the physical triangle is |detJ| times the area of
        # the reference triangle
        self.area = 0.5*np.abs(self.detJ)
        # Go ahead and form the inverse of J^T. With a 2x2 matrix this is as
        # fast and accurate as an LU factorization followed by triangular solves
        self.JtInv = np.array([[self.Jt[1,1], -self.Jt[0,1]],
            [-self.Jt[1,0],self.Jt[0,0]]])/self.detJ

# --------------------------------------------------------------------------
#
# The LocalLaplacian class provides a localMat() function for computing
# the local matrix for the weak negative Laplacian operator
# \int_T grad(phi_i)*grad(phi_j)
# on a triangle with degree one Lagrange basis functions.
#
class LocalLaplacian:
    # The gradients of the three Lagrange basis functions on the reference
    # triangle are constant vectors and can be computed once and for all.
    # These are packed columnwise into the 2 by 3 matrix:
    # [ -1 1 0 ]
    # [ -1 0 1 ]
    # Implementation note: this variable has been defined outside of any
    # member functions of this class; it is a "class variable" and is shared
    # by all instances of this class. Since it's a property of the class rather
    # than a particular instance of this class, we access it as
    # LocalLaplacian.gradPhiRef rather than as self.gradPhiRef.
    gradPhiRef = np.array([[-1,1,0],[-1,0,1]])

    # Form the local matrix. Since the gradients are constant, no quadrature
    # is needed, but we do have to transform the gradients from reference to
    # physical coordinates. That's done as follows:
    # grad_phys(phi) = J^{-T} * grad_ref(phi).
    # This can be done on all three basis functions at once since we've packed
    # their gradients into columns of the matrix LocalLaplacian.gradPhiRef.
    def localMat(self, tri):
        # Transform the gradients
        gradPhi = np.matmul(tri.JtInv, LocalLaplacian.gradPhiRef)
        # Form the local matrix
        return tri.area * np.matmul(np.transpose(gradPhi), gradPhi)


# --------------------------------------------------------------------------
#
# The LocalMass class provides a localMat() function for computing
# the local mass matrix (aka Gram matrix)
# \int_T phi_i*phi_j
# on a triangle with degree one Lagrange basis functions. In transforming
# from reference to physical triangle, the integral is simply scaled by the
# Jacobian determinant. We can compute all the integrals once and for all
# offline, and re-use the matrix for all triangles.
#
class LocalMass:
    # Here's the local mass matrix, up to scaling by the Jacobian, computed
    # offline using Mathematica. Since it's the same for all triangles,
    # make it a class variable rather than an instance variable.
    M = np.array([[2,1,1],[1,2,1],[1,1,2]])/12.0

    # Compute the local mass matrix on a triangle.
    def localMat(self, tri):
        return tri.area * LocalMass.M

class LocalLoad:
    def __init__(self, func):
        self.func = func

    def localVec(self, tri):
        funcAB = self.func(0.5*(tri.A + tri.B))
        funcBC = self.func(0.5*(tri.B + tri.C))
        funcAC = self.func(0.5*(tri.A + tri.C))
        weight = tri.area/3.0

        return weight * np.array(
            [
                0.5*(funcAB + funcAC),
                0.5*(funcAB + funcBC),
                0.5*(funcAC + funcBC)
            ]
        )


if __name__=='__main__':

    T = Triangle((1,1), (2,1.5),(1.5,2))
    print('Jt=\n', T.Jt)
    print('det(J)=', T.detJ)

    print('Jt*inv(Jt)=\n', np.matmul(T.Jt, T.JtInv))

    lapl = LocalLaplacian()
    print('lapl=\n', lapl.localMat(T))

    mass = LocalMass()
    print('mass=\n', mass.localMat(T))

    def loadFunc(xy):
        return 1

    load = LocalLoad(loadFunc)
    print('load=\n', load.localVec(T))
