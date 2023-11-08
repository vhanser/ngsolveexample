#!/usr/bin/env python
# coding: utf-8



# In[5]:


from netgen.occ import *
from ngsolve import *
from netgen.meshing import IdentificationType
#from MS_helper_functions import *
from netgen.webgui import Draw as DrawGeo
from ngsolve import Draw
# Draw = lambda *args, **kwargs : None

import matplotlib.pyplot as plt

from myPackage import evalOnLine, __sep__
from stepLapMeshGenerator import sheetsMeshCenter

import numpy as np



def run():
    static = False

    numSheets = 5
    numPeriods = 1
    d = 20e-3/numSheets
    ff = 0.7

    dFe = d*ff
    d0 = d*(1-ff)

    

        

    specialTB1 = [1, 1, 1,1]
    specialTB1 = False
    specialTB2 = [1, 1, 1,1]
    specialTB2 = False
    domainName_insulation = "air"
    halfAir =True
    fullProblemX = True
    fullProblemY = True
    fullProblemZ = True


    x0 = __sep__(10e-3, 10)

    x1 = __sep__(20e-3, 20)

    x1_i = __sep__(2.5*d, 3)
    x2 = np.flipud(x0)


    y0 = __sep__(20e-3, [6, 4, 2, 1.5, 0.5, 0.25])
    y1 = __sep__(3*d, 1)
    y2 = list(np.flipud(y0))

    y0 = []
    y2 = []

    z0=__sep__(10e-3, [10, 10, 10])
    z1_iron = __sep__(dFe, [1, 3, 5, 3, 1])
    z1_air=__sep__(d0, 3)
    z2 = list(np.flipud(z0))

    z1_iron_MS_sep = [0.1, 0.1, 1, 3, 7, 7, 3, 1, 0.1, 0.1]


    # class myDraw:
    #     scenes = {}
    #     def __init__(self, *args, **kwargs):
    #         if id(args[0]) not in myDraw.scenes.keys():
    #             myDraw.scenes.update({id(args[0]):Draw(*args, **kwargs)})
    #         else:
    #             myDraw.scenes[id(args[0])].Draw()
    #     def updateAll():
    #         for s in myDraw.scenes.values():
    #             s.Redraw()


    # In[6]:


    mu0 = 4e-7*np.pi

    muAir = 1 * mu0
    muFe = 10 * mu0

    sigmaFe = 2e6

    omega = 50 * 2*np.pi




    order0 = 2


    # ## 4. All together with Eddy currents
    # ### 4.1. Reference Solution 

    # In[7]:


    cMeshRef = sheetsMeshCenter("test.vol", x0, x1, x2, y0, y1, y2, z0, z1_iron, z1_air, z2, 
        x1_i=x1_i,
        numSheets=numSheets, numPeriods=1,
        multiScale=False, 
        z1_iron_MS_sep=False, 
        specialTB1=False, specialTB2=False, # speTB1 for seperated first sheet, speTB2 for spec domain in first sheet
        domainName_Gap = "gap",
        domainName_insulation="insulation", 
        halfAir=halfAir, 
        specialBoundariesForGapDomain=False, 
        specialBoundariesForInnerOuter=False,
        fullProblemX=True,
        fullProblemY=True, 
        fullProblemZ=True, seperateIronDomains=False)

    meshRef=cMeshRef.Mesh

    print("dFe", sum(cMeshRef.z1_iron))
    print("penetration depth = ", sqrt(2/(muFe*omega*sigmaFe)))

    print(meshRef.GetMaterials())
    print(meshRef.GetBoundaries())


    # In[8]:


    # from myPackage import drawBndAll
    # drawBndAll(meshRef, drawFunc=Draw, block=False)


    # In[9]:


    rotate_excitation = False
    mu = meshRef.MaterialCF({"iron":muFe, "air":muAir, "insulation":muAir, "gap":muAir, "multiscale":muFe}, default=muAir)

    def calcRef():
        fesPhi = H1(meshRef, order=order0+1, dirichlet="top|bottom" if rotate_excitation else "left|right" , complex=not static)
        if not static:
            fesT = HCurl(meshRef, order=order0, dirichlet="itop|ibot|interface_x|left|right", complex=True, definedon=meshRef.Materials("iron"), nograds=True)
            fes = FESpace([fesPhi, fesT])
        
        else:
            fes = FESpace([fesPhi])

        print("ref ndof", fes.ndof)
        trials, tests = fes.TnT()
        sol = GridFunction(fes)

        a = BilinearForm(fes, symmetric=True)
        f = LinearForm(fes)


        if not static:
            a += 1j * omega * mu * (trials[1]-grad(trials[0])) * (tests[1]-grad(tests[0])) * dx
            a += 1/sigmaFe * curl(trials[1]) * curl(tests[1]) * dx("iron")
        else:
            a += mu * (-grad(trials[0])) * (-grad(tests[0])) * dx
        # a += 1e-1 * trials[0] * tests[0] * dx("iron")


        prec = Preconditioner(a, "direct")

        Phi = sol.components[0]
        if not static:
            T = sol.components[1]

        Phi.Set((1 * y/Norm(y)) if rotate_excitation else (1 * x/Norm(x)), BND)

        with TaskManager():
            solvers.BVP(bf = a, lf= f, pre=prec, gf=sol, maxsteps=10)
        
        
        energy =  Integrate( InnerProduct(mu * grad(Phi), grad(Phi)), meshRef, definedon=meshRef.Materials("iron|insulation")).real
        if not static:
            eddyLosses =  Integrate( InnerProduct(1/sigmaFe * curl(T), curl(T)), meshRef, definedon=meshRef.Materials("iron")).real
        else:
            eddyLosses = 0
        
        return sol, energy, eddyLosses

    sol_ref, energy_ref, eddyLosses_ref = calcRef()

    Phi = sol_ref.components[0]

    if not static:
        T = sol_ref.components[1]
        H_ref = T-grad(Phi)
        J_ref = curl(T)
    else:
        J_ref = CF((0, 0, 0))
        H_ref = -grad(Phi)
    B_ref = mu  * H_ref


    # ### 4.2 Multiscale 

    # In[9]:


    cMeshMS = sheetsMeshCenter("test.vol", x0, x1, x2, y0, y1, y2, z0, z1_iron, z1_air, z2, 
            x1_i=x1_i,
            numSheets=numSheets, numPeriods=1,
            multiScale=True, 
            z1_iron_MS_sep=z1_iron_MS_sep, 
            specialTB1=False, specialTB2=False, # speTB1 for seperated first sheet, speTB2 for spec domain in first sheet
            domainName_Gap = "gap",
            domainName_insulation="insulation", 
            halfAir=halfAir, 
            specialBoundariesForGapDomain=False, 
            specialBoundariesForInnerOuter=False,
            fullProblemX=True,
            fullProblemY=True, 
            fullProblemZ=True, seperateIronDomains=False)

    meshMS=cMeshMS.Mesh
    print(meshMS.GetMaterials())
    print(meshMS.GetBoundaries())




    # In[ ]:


    import importlib
    import MS_helper_functions as ms
    ms = importlib.reload(ms)
    cl_Phi = ms.cl_Phi

    getIntegrand4BFI = ms.getIntegrand4BFI
    cl_gradgradMS = ms.cl_gradgradMS
    cl_curlcurlMS = ms.cl_curlcurlMS
    pyLobatto = ms.pyLobatto
    pydxLobatto = ms.pydxLobatto
    getPhiPhiValue = ms.getPhiPhiValue
    pyPhiFunction = ms.pyPhiFunction
    pyPhiZero = ms.pyPhiZero
    pyPhiConst = ms.pyPhiConst


    cl_Phi.numSheets = numSheets
    cl_Phi.dFe = dFe
    cl_Phi.d0 = d0
    cl_Phi.mesh = meshMS

    cl_Phi.modelHalfAir = True
    cl_Phi.orientation = 2 

    if False:
        import cempy as cp
        importlib.reload(cp)
        
        
        cl_Phi.phiFunction = cp.phiFunctions.Lobatto
        cl_Phi.dzPhiFunction = cp.phiFunctions.dxLobatto
    else:
        cl_Phi.phiFunction = pyLobatto
        cl_Phi.dzPhiFunction = pydxLobatto





    # In[ ]:


    force_full_Phi = True


    smoothB_fun_Fe = lambda x: 1e3 * muFe/muAir * (x**2/cl_Phi.dFe * 1 - (cl_Phi.dFe/4 * 1))
    smoothB_fun_0 = lambda x: 1e3 * (-(x - (cl_Phi.d0/2+cl_Phi.dFe/2))**2/ cl_Phi.d0 * 1 + cl_Phi.d0/4 * 1)



    smoothH_fun_Fe = lambda x: 1e3 * (x**2/cl_Phi.dFe * 1 - (cl_Phi.dFe/4 * 1))
    smoothH_fun_0 = lambda x: 1e3*(-(x - (cl_Phi.d0/2+cl_Phi.dFe/2))**2/ cl_Phi.d0 * 1 + cl_Phi.d0/4 * 1)

    smoothBPhi2 = pyPhiFunction(cl_Phi.getZStart(), cl_Phi.getD(), 
        [smoothB_fun_Fe, smoothB_fun_0], numSheets, ff, cl_Phi.orientation, name = "smoothBPhi2", order="smB2", modelHalfAir=False)

    smoothHPhi2 = pyPhiFunction(cl_Phi.getZStart(), cl_Phi.getD(), 
        [smoothH_fun_Fe, smoothH_fun_0], numSheets, ff, cl_Phi.orientation, name = "smoothHPhi2", order="smH2", modelHalfAir=False)

    smoothPhi2_Fe = pyPhiFunction(cl_Phi.getZStart(), cl_Phi.getD(), 
        [smoothB_fun_Fe, smoothB_fun_0], numSheets, ff, cl_Phi.orientation, name = "smoothPhi2", order="sm2", modelHalfAir=False, inAir=False)
    
    
    
    def calcMultiscale(useGradients=True, drawPhis=True):
        domains = "iron|gap"

        roughbnd = "left|right" 
        roughbnd_inner = "interface_x" 
        smoothbnd = "itop|ibot" 



        print("domains", domains)
        print("roughbnd", roughbnd)
        print("smoothbnd", smoothbnd)

        # microshape functions

        
        """
        only roguh

        orderPhi = [
            # do not remove!
            cl_Phi([pyPhiConst(), pyPhiZero()], fes_order=1, material=domains, dirichlet=roughbnd, useGradients=True, useAbsolutes=False), 

            cl_Phi(1, fes_order=1, material=domains, dirichlet=roughbnd, useGradients=True, useAbsolutes=True, modelHalfAir=True), 
        
            cl_Phi([smoothBPhi2, smoothBPhi2.getDiff()], fes_order=1, material=domains, dirichlet=roughbnd, useGradients=True, modelHalfAir=False),
            cl_Phi(2, fes_order=1, material=domains, dirichlet=roughbnd, inIron=False, modelHalfAir=False), 
        ]

        orderT = [
            # cl_Phi([pyPhiConst(val = 1, inAir=False), pyPhiZero()], fes_order=2, material="multiscale", dirichlet=roughbnd_inner+ "|" + smoothbnd, inAir=False), 

            # cl_Phi(1, fes_order=1, material="multiscale", dirichlet=roughbnd_inner, inAir=False, modelHalfAir=False),  
            cl_Phi(2, fes_order=2, material="multiscale", dirichlet=roughbnd_inner, inAir=False, modelHalfAir=False), 
        ]
        """
        orderPhi = [

                cl_Phi(1, fes_order=1, material=domains, dirichlet=roughbnd, useGradients=True, useAbsolutes=True, modelHalfAir=True), 
                cl_Phi([smoothBPhi2, smoothBPhi2.getDiff()], fes_order=1, material=domains, dirichlet=roughbnd, useGradients=True, modelHalfAir=False),
                cl_Phi(2, fes_order=1, material=domains, dirichlet=roughbnd, inIron=False, modelHalfAir=False), 
                # cl_Phi(2, fes_order=1, material=domains, dirichlet=roughbnd+ "|" + smoothbnd, inAir=False, modelHalfAir=False), 
            ]

        orderT = [
            # cl_Phi([pyPhiConst(val = 1, inAir=False), pyPhiZero()], fes_order=2, material="iron", dirichlet=roughbnd_inner+ "|" + smoothbnd, inAir=False, nograds=False), 
            # cl_Phi(1, fes_order=2, material="iron", dirichlet=roughbnd_inner, inAir=False, modelHalfAir=True, nograds=True),  
            cl_Phi(2, fes_order=2, material="iron", dirichlet=roughbnd_inner, inAir=False, modelHalfAir=False, nograds=True), 
        ]




        if static:
            orderT = []

        

        if drawPhis:
            # cl_Phi.plotEvaluated(orderPhi, nFig=1)
            cl_Phi.plotEvaluated(orderPhi, nFig=1)
            cl_Phi.plotDirectEvaluated(orderPhi, nFig=2)

        VSpace = []
        # # u0 
        VSpace.append(H1(meshMS, order=order0+1, dirichlet="top|bottom" if rotate_excitation else "left|right", complex=not static)) 
            
        # ui * phi i
        for phi_i in orderPhi: 
            VSpace.append(H1(meshMS, order=phi_i.fes_oder+1, definedon=meshMS.Materials(phi_i.material), dirichlet=phi_i.dirichlet, complex=not static))

        for phi_i in orderT: 
            VSpace.append(HCurl(meshMS, order=phi_i.fes_oder, definedon=meshMS.Materials(phi_i.material), dirichlet=phi_i.dirichlet, complex=not static, nograds=phi_i.nograds))

        VSpace = FESpace(VSpace)


        # multiscale container
        ansatz = ""
        sol = GridFunction(VSpace, "sol")
        

        gradgradMS = cl_gradgradMS(orderPhi, sol, addPhi0Outer=True, secondOrder=False)
        if not static:
            curlcurlMS = cl_curlcurlMS(orderT, sol, eddy_inplane=False, istart = len(gradgradMS.orderPhi) + 1)
            gradgradMS.addCurlCurlMS(curlcurlMS)



        slice_inner = slice(0, len(gradgradMS.gradu_pack))


        a = BilinearForm(VSpace, symmetric=True)
        f = LinearForm(VSpace)

        if static:
            a += muAir  * grad(gradgradMS.trials[0]) * grad(gradgradMS.tests[0]) * dx("air")
            # a += 1j * 1e-1  * gradgradMS.trials[0] * gradgradMS.tests[0] * dx("multiscale")
            a += gradgradMS.getIntegrand4BFI(gradgradMS.gradu_pack[slice_inner], gradgradMS.gradv_pack[slice_inner], muAir, muAir, force_full_Phi=force_full_Phi) * dx("gap")
            a += gradgradMS.getIntegrand4BFI(gradgradMS.gradu_pack[slice_inner], gradgradMS.gradv_pack[slice_inner], muFe, muAir, force_full_Phi=force_full_Phi) * dx("iron")

        else:
            a += 1j * omega * muAir  * grad(gradgradMS.trials[0]) * grad(gradgradMS.tests[0]) * dx("air")
            # a += 1j * 1e-1  * gradgradMS.trials[0] * gradgradMS.tests[0] * dx("multiscale")
            a += 1j * omega * gradgradMS.getIntegrand4BFI(gradgradMS.gradu_pack[slice_inner], gradgradMS.gradv_pack[slice_inner], muAir, muAir, force_full_Phi=force_full_Phi) * dx("gap")
            a += 1j * omega * gradgradMS.getIntegrand4BFI(gradgradMS.gradu_pack[slice_inner], gradgradMS.gradv_pack[slice_inner], muFe, muAir, force_full_Phi=force_full_Phi) * dx("iron")

            a += curlcurlMS.getIntegrand4BFI(curlcurlMS.curlu_pack, curlcurlMS.curlv_pack, 1/sigmaFe, 0, force_full_Phi=force_full_Phi) * dx("iron")
            
        


        # couple u
        if False:
            alpha = 3.5
            h = specialcf.mesh_size

            u_start = 0
            # alpha = 1e6
            d_coupling = -d/2 


            # absolutes
            u_range = range(u_start, len(gradgradMS.u_pack))
            # u_range = list(range(2, len(gradgradMS.u)))
            # u_range = [0] + u_range

            # u_range = [u_start]
            um_bottom = sum(gradgradMS.u_pack[i][0] * gradgradMS.u_pack[i][1].DirectEvaluate(-d_coupling) for i in u_range)
            vm_bottom = sum(gradgradMS.v_pack[i][0] * gradgradMS.v_pack[i][1].DirectEvaluate(-d_coupling) for i in u_range)

            um_top = sum(gradgradMS.u_pack[i][0] * gradgradMS.u_pack[i][1].DirectEvaluate(d_coupling) for i in u_range)
            vm_top = sum(gradgradMS.v_pack[i][0] * gradgradMS.v_pack[i][1].DirectEvaluate(d_coupling) for i in u_range)

            jump_u_top = gradgradMS.u_pack[0][0] - um_top
            jump_u_bottom = gradgradMS.u_pack[0][0] - um_bottom

            jump_v_top = gradgradMS.v_pack[0][0] - vm_top
            jump_v_bottom = gradgradMS.v_pack[0][0] - vm_bottom


            #[u][v]
            # a += alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.u_pack[u_start:], gradgradMS.v_pack[u_start:],  1, 1, force_full_Phi=force_full_Phi)  *ds(smoothbnd)
            # a += -alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.u_pack[u_start:], gradgradMS.v_pack[:1],  1, 1, force_full_Phi=force_full_Phi)  *ds(smoothbnd)
            # a += alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.u_pack[:1], gradgradMS.v_pack[:1],  1, 1, force_full_Phi=force_full_Phi)  *ds(smoothbnd)
            # a += -alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.u_pack[:1], gradgradMS.v_pack[u_start:],  1, 1, force_full_Phi=force_full_Phi)  *ds(smoothbnd)

            a += alpha*order0**2/h  * (jump_u_top)  * (jump_v_top)  *ds("itop")
            a += alpha*order0**2/h  * (jump_u_bottom)  * (jump_v_bottom)  *ds("ibottom")
            



        # couple fluxes
        if False:
            # [dn u] [ dn v]

            alpha = 2000
            a += alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.gradu_trace_n_pack[u_start:], gradgradMS.gradv_trace_n_pack[u_start:],  muFe, muAir, force_full_Phi=force_full_Phi)  *ds(smoothbnd)
            a += -alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.gradu_trace_n_pack[u_start:], gradgradMS.gradv_trace_n_pack[:1],  muFe, muAir, force_full_Phi=force_full_Phi)  *ds(smoothbnd)
            a += alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.gradu_trace_n_pack[:1], gradgradMS.gradv_trace_n_pack[:1],  muFe, muAir, force_full_Phi=force_full_Phi)  *ds(smoothbnd)
            a += -alpha*order0**2/h  * gradgradMS.getIntegrand4BFI(gradgradMS.gradu_trace_n_pack[:1], gradgradMS.gradv_trace_n_pack[u_start:],  muFe, muAir, force_full_Phi=force_full_Phi)  *ds(smoothbnd)

            # alpha = 2
            # a += alpha*order0**2/h  * (jump_Bu_top)  * (jump_Bv_top)  *ds("itop")
            # a += alpha*order0**2/h  * (jump_Bu_bottom)  * (jump_Bv_bottom)  *ds("ibottom")


        prec = Preconditioner(a,type="direct", inverse="pardiso")  

        print("MS ndofs", VSpace.ndof)


        # dirichlet boundary values
        sol.components[0].Set((1 * y/Norm(y)) if rotate_excitation else (x/Norm(x)), BND)

        with TaskManager():
            solvers.BVP(bf = a, lf= f, pre=prec, gf=sol, maxsteps=30, tol = 1e-20, )


        print("done")

        H_MS = sum(gradgradMS.gradsol_comp)
        if static:
            J_MS = CF((0, 0, 0))
        else:
            J_MS = sum(curlcurlMS.curlsol_comp)



        energy =  Integrate(gradgradMS.getIntegrand4BFI(gradgradMS.gradsol_pack, gradgradMS.gradsol_pack, muFe, muAir), meshMS, 
                                                        definedon=meshMS.Materials("iron")).real

        if static:
            losses = 0
            curlcurlMS = 0
        else:
            losses =  Integrate(curlcurlMS.getIntegrand4BFI(curlcurlMS.curlsol_pack, curlcurlMS.curlsol_pack, 1/sigmaFe, 0), meshMS, 
                                                        definedon=meshMS.Materials("iron")).real
        
        
        print("a norm", a.mat.AsVector().Norm())

        print("ansatz", gradgradMS.ansatz)
        return sol, energy, losses, gradgradMS, curlcurlMS, H_MS, J_MS

        






    # In[ ]:


    if "curlcurlMS" in locals():
        del curlcurlMS
    sol_MS, energy_MS, eddyLosses_MS, gradgradMS, curlcurlMS, H_MS, J_MS = calcMultiscale(False, drawPhis=False)

    #H_MS.Compile()
    # print("Norm soll diff", Integrate((sum(sol_comp_MS) - sol_ref)* (sum(sol_comp_MS) - sol_ref)/( (sol_ref)**2) , meshRef)*100,  "%")
    # print("diff energy", energy_MS, energy_ref, energy_MS - energy_ref, (energy_MS - energy_ref)/energy_ref * 100, "%")

    # print(Integrate(Norm(sum(sol_comp_MS) - sol_ref), meshRef))


    u_MS = sum(gradgradMS.sol_comp[:len(gradgradMS.orderPhi)])

    print("energy_ref, energy_MS", energy_ref, energy_MS)
    print("eddy_ref, eddy_MS", eddyLosses_ref, eddyLosses_MS)


    # In[ ]:


    if not static:
        print(curlcurlMS.orderPhi[0].dirichlet)
        print(set(meshMS.GetBoundaries()))
        from myPackage import drawBnd

        # drawBnd(meshMS, "ibottom", drawFunc=Draw)


    # In[ ]:


        curlcurlMS.printCouplingMatrix(sparsity=True);


    # In[ ]:


    #gradgradMS.generateCouplingMatrix(muFe, muAir, force_full_Phi=force_full_Phi)
    gradgradMS.printCouplingMatrix(sparsity=True);
    #assert gradgradMS.checkCouplingMatrxiSymmetric(1e-3) == True


    # In[ ]:

    if not static:
        curlcurlMS.generateCouplingMatrix(1/sigmaFe, 0, force_full_Phi=force_full_Phi)
        curlcurlMS.printCouplingMatrix(sparsity=1);
    # assert curlcurlMS.checkCouplingMatrxiSymmetric(1e-3) == True



    # In[ ]:


    print(f"energy MS :\t{energy_MS}, energy ref \t{energy_ref}")
    print(f"eddyLosses MS :\t{eddyLosses_MS}, eddyLosses ref \t{eddyLosses_ref}")
    print(meshMS.GetBoundaries())
    #Draw(u_MS, meshRef, settings={"Objects":{"Wireframe":True}, "deformation": 0.01}, deformation=False)
    # Draw( Norm(H_MS), meshRef, settings={"Objects":{"Wireframe":False}, "deformation": False}, max = 200)
    # Draw( lam * Norm(H_MS), meshRef, settings={"Objects":{"Wireframe":False}, "deformation": False}, max=400)


    # print("Norm soll diff", Integrate((sum(sol_comp_MS) - sol_ref)* (sum(sol_comp_MS) - sol_ref)/( (sol_ref)**2) , meshRef)*100,  "%")
    print("diff energy", energy_MS, energy_ref, energy_MS - energy_ref, (energy_MS - energy_ref)/energy_ref * 100, "%")
    if not static:
        print("diff eddylosses", eddyLosses_MS, eddyLosses_ref, eddyLosses_MS - eddyLosses_ref, (eddyLosses_MS - eddyLosses_ref)/eddyLosses_ref * 100, "%")

    sol_MS_dict = {"air": gradgradMS.sol_comp[0], "gap": sum(gradgradMS.sol_comp), "iron": sum(gradgradMS.sol_comp)}
    H_MS_dict = {"air": gradgradMS.gradsol_comp[0], "gap": sum(gradgradMS.gradsol_comp), "iron": sum(gradgradMS.gradsol_comp)}
    
    # Draw(meshRef.MaterialCF(sol_MS_dict), meshRef, "sol_MS")
    # Draw(sol_ref.components[0], meshRef, "sol_Ref")

    # Draw(sum(gradgradMS.gradsol_comp), meshRef, "H_MS")
    # Draw(H_ref, meshRef, "H_Ref")
    Draw(IfPos(x, sum(gradgradMS.gradsol_comp), H_ref), meshRef, "H_diff")
    Draw(mu * IfPos(x, sum(gradgradMS.gradsol_comp), H_ref), meshRef, "B_diff")

    Draw(IfPos(x, J_ref, J_MS), meshRef, "J_diff")


    print("eddy current losses ref", Integrate(InnerProduct(J_ref, J_ref * 1/sigmaFe), meshRef, definedon=meshRef.Materials("iron")).real)
    print("eddy current losses MS", Integrate(InnerProduct(J_MS, J_MS * 1/sigmaFe), meshRef, definedon=meshRef.Materials("iron")).real)
    


    from myPackage import myBreak
    myBreak(locals(), globals(), file=__file__.split('/')[-1])