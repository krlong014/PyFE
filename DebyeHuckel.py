from ElementIntegration import *
from TriangleMeshReader import *
from LoadableMesh import *
import scipy.sparse.linalg as spla
import numpy.linalg as la
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite

# Function to produce the system matrix and load vector for the Debye-Huckel
# equation, discretized by finite elements with piecewise linear Lagrange
# basis functions on the specified mesh.
#
# The equation is
# \[
# -\nabla^2 u + \beta^2 u = f
# \]
# with homogeneous Neumann BCs on the entire boundary.

def DiscretizeDH(mesh, loadFunc, beta):

    # Matrix will be m by m, with m=len(mesh.verts)
    numNodes = len(mesh.verts)
    A = sp.dok_matrix((numNodes, numNodes))
    b = np.zeros(numNodes)

    betaSq = beta*beta

    # objects that provide functions for computing local matrices and vectors
    LLapl = LocalLaplacian()
    LMass = LocalMass()
    LLoad = LocalLoad(loadFunc)

    # Indicate whether debugging output is desired
    debug = False

    # Main loop over elements
    # Recall: elements are stored as list of vertex index triplets,
    # [ (a0,b0,c0), (a1,b1,c1), etc]
    # "ie" is the loop counter; this isn't needed in calculations, but is
    #      useful to know when debugging.
    # "eVerts" is the triplet of vertex indices for the current element
    for ie,eVerts in enumerate(mesh.elems):
        if debug: print('processing element #%d' % ie, " vertices=", eVerts)
        # Get coordinates of the three elements
        va = mesh.verts[eVerts[0]]
        vb = mesh.verts[eVerts[1]]
        vc = mesh.verts[eVerts[2]]

        # Create a triangle object with the three vertices
        T = Triangle(va, vb, vc)

        # Form local stiffness matrix
        K = LLapl.localMat(T)
        if debug: print('\tLocal Laplacian: ', K)

        # Form local mass matrix
        M = LMass.localMat(T)
        if debug: print('\tLocal mass matrix: ', M)

        # Form local load vector
        load = LLoad.localVec(T)
        if debug: print('\tLocal load vector: ', load)

        # Add stiffness matrix and betaSq-weighted mass matrix
        localA = K + betaSq*M

        # Pack local matrix and vector into global matrix and vector.
        # "eVerts" is vertex index triplets (a,b,c) for this element; these
        # are the *global* row and column indices into which the local values
        # get inserted. Local rows/cols are [0,1,2]. Use enumerate to produce
        # loop counter (i.e., local index) as we loop.
        for localRow, globalRow in enumerate(eVerts):
            # Add in the current element's contribution to the load vector
            b[ globalRow ] += load[ localRow ]
            # Add in the current element's contribution to the system matrix
            for localCol, globalCol in enumerate(eVerts):
                A[ globalRow,globalCol ] += localA[localRow,localCol]


    return (A,b)

    # # We'll solve using the super LU sparse direct solver.
    # factorization = spla.splu(A)
    # # Solve the system using the
    # u = factorization.solve(b)
    # return u

class ConstantFunc:
    def __init__(self, c):
        self.c = c

    def __call__(self, xyPair):
        return self.c

class CosTestLoad:
    def __init__(self, beta):
        self.betaSq = beta*beta

    def __call__(self, xyPair):
        x = xyPair[0]
        y = xyPair[1]
        pi = np.pi
        return (5.0*pi*pi + self.betaSq)*np.cos(pi*x)*np.cos(2.0*pi*y)

class CosTestSoln:
    def __init__(self):
        pass

    def __call__(self, xyPair):
        x = xyPair[0]
        y = xyPair[1]
        pi = np.pi
        return np.cos(pi*x)*np.cos(2.0*pi*y)


if __name__=='__main__':

    beta = 1.0

    for level in range(0,16):

        reader = TriangleMeshReader('./Meshes/triExample.%d' % level)
        mesh = reader.getMesh()


        print('\n\n\n---------- Constant load test problem ---------\n\n')
        load = ConstantFunc(beta*beta)
        (A,b) = DiscretizeDH(mesh, load, beta)
        mmwrite('DH-Matrix-%d.mtx' % level, A,
            comment='Debye-Huckel beta=%g',
            precision=16)
