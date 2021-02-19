
def testConvergenceRate(problem, )
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
