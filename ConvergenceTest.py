from ElementIntegration import *
from TriangleMeshReader import *
from LoadableMesh import *
import scipy.sparse.linalg as spla
import numpy.linalg as la
import numpy as np
import scipy.sparse as sp

# Function to solve D-H equation with FEM on a mesh. The load function
# is "func" and the beta parameter is beta.
def SolveDH(mesh, func, beta):

    # Matrix will be m by m, with m=len(mesh.verts)
    numNodes = len(mesh.verts)
    A = sp.dok_matrix((numNodes, numNodes))
    b = np.zeros(numNodes)

    betaSq = beta*beta

    # objects that provide functions for computing local matrices and vectors
    LLapl = LocalLaplacian()
    LMass = LocalMass()
    LLoad = LocalLoad(func)

    debug = False
    # Main loop over elements
    # Recall: elements are stored as list of vertex index triplets,
    # [ (a0,b0,c0), (a1,b1,c1), etc]
    # "ie" is the loop counter
    # "eVerts" is the triplet of element indices
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

    # We'll solve using the super LU sparse direct solver.
    factorization = spla.splu(A)
    # Solve the system using the
    u = factorization.solve(b)
    return u

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

    beta = 10.0

    hList = []
    errList = []
    for level in range(0,6):

        reader = TriangleMeshReader('./Meshes/triExample.%d' % level)
        mesh = reader.getMesh()


        # first test: constant load
        # (*) exact solution: u(x,y)=1
        # (*) load: f(x) = beta^2
        print('\n\n\n---------- Constant load test problem ---------\n\n')
        load = ConstantFunc(beta*beta)
        uSoln = SolveDH(mesh, load, beta)
        if level <= 1: print('numerical solution = ', uSoln)


        uEx = evalOnMesh(mesh, ConstantFunc(1.0))
        print('exact soln norm = %g' % la.norm(uEx))
        if level <= 1: print('exact solution = ', uEx)
        print('error norm = %g' % la.norm(uEx - uSoln))


        # second test: cosine load
        # (*) exact solution: u(x,y) = cos(pi x) cos(2 pi y)
        # (*) load: f(x,y) = (5 pi^2 + beta^2) cos(pi x) cos(2 pi y)
        print('\n\n\n---------- Cosine load test problem ---------\n\n')
        load = CosTestLoad(beta)
        uSoln = SolveDH(mesh, load, beta)
        if level <= 1: print('numerical solution = ', uSoln)

        uEx = evalOnMesh(mesh, CosTestSoln())
        if level <= 1: print('exact solution = ', uEx)
        exNorm = float(la.norm(uEx))
        errNorm = float(la.norm(uEx - uSoln))
        h = hMesh(mesh)
        print('h=%g, \terror norm = %g' % (h, errNorm/exNorm))
        hList.append(h)
        errList.append(errNorm/exNorm)

    print('h=', hList)
    print('err=', errList)

    import matplotlib.pyplot as plt
    plt.loglog(hList, errList, 'o')
    hsq = [h**2 for h in hList]
    plt.loglog(hList, hsq, '-')
    plt.xlabel('h')
    plt.xlabel('Error')
    plt.legend(['Numerical error', 'h^2'])
    plt.grid()
    plt.show()
