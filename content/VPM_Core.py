from netgen.csg import *
from ngsolve import *
import numpy as np
from myPackage import colorprint, TextColor
import cempy as cp
import time
import matplotlib.pyplot as plt


def MakeGeometry(modelcoil=True):
    geometry = CSGeometry()
    box = OrthoBrick(Pnt(-1,-1,-1),Pnt(2,1,2)).bc("outer")

    core = OrthoBrick(Pnt(0,-0.05,0),Pnt(0.8,0.05,1))- \
           OrthoBrick(Pnt(0.1,-1,0.1),Pnt(0.7,1,0.9))- \
           OrthoBrick(Pnt(0.5,-1,0.4),Pnt(1,1,0.6)).maxh(0.2).mat("core")
    
    coil = (Cylinder(Pnt(0.05,0,0), Pnt(0.05,0,1), 0.3) - \
            Cylinder(Pnt(0.05,0,0), Pnt(0.05,0,1), 0.15)) * \
            OrthoBrick (Pnt(-1,-1,0.3),Pnt(1,1,0.7)).maxh(0.2).mat("coil")
    
    geometry.Add ((box-core-coil).mat("air"))
    geometry.Add (core)
    geometry.Add (coil)
    return geometry


def calcBiotSavartField(mesh):
    fes = HCurl(mesh, order=2)
    u, v = fes.TnT()
    A = GridFunction(fes)

    a = BilinearForm(fes, symmetric=True)
    f = LinearForm(fes)


    nu = 1/(4e-7*np.pi)
    a = BilinearForm(fes, symmetric=True)

    a += nu*curl(u)*curl(v)*dx + 1e-6*nu*u*v*dx

    c = Preconditioner(a, type="bddc")
    # c = Preconditioner(a, type="multigrid", flags = { "smoother" : "block" } )

    f = LinearForm(fes)
    f += CF((y,0.05-x,0)) * v * dx("coil")


    with TaskManager():
        a.Assemble()
        f.Assemble()
        solver = CGSolver(mat=a.mat, pre=c.mat)
        A.vec.data = solver * f.vec


    B = curl(A)
    H = nu * B

    return H


def calcWithTPhi(mesh, order0, H0_amp, HBS, ti, NperPeriod,omega=50*np.pi*2, sigmaFe=2e6 ):


    print("--"*40)
    print("calc it with H0_amp=", H0_amp)
    print("--"*40)


    mip = mesh(0.05, 0, 0.5)
    
    mask = mesh.MaterialCF({"core":1}, default=0)
    muAir = 4e-7*np.pi
       


    intrule = IntegrationRule(HEX,2*order0+2)
    irSpace = {HEX:intrule}

    Vc = Integrate(mask, mesh)
    Vol = Integrate(1, mesh)

    
    # ------------------------------------------------------------------------------
    # --- excitation
    # ------------------------------------------------------------------------------
    H0_i = np.sin(omega * ti)
    dt = Parameter(ti[1] - ti[0])
    N_ti = len(ti)

    HBS_vec = HBS

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ reference solution
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    H0 = Parameter(0)
    H0_o = Parameter(0)
    print("order0", order0)




    dir_phi = ".*"
    dir_T = ".*"
    

    VPhi_ref = H1(mesh,order=order0+1, dirichlet=dir_phi)
    VT_ref = HCurl(mesh,order=order0, definedon="core", nograds=True, dirichlet=dir_T)

    
    VSpace = FESpace([VPhi_ref, VT_ref])
        

    ndof = sum(VSpace.FreeDofs())	
    print('-----')  
    print('ndof =',ndof)
    print('-----')
    
    # Gridfunctions
    sol_ref = GridFunction(VSpace, "sol") 
    sol_ref_o = GridFunction(VSpace, "sol_old")
    sol_ref_it_o = GridFunction(VSpace, "sol_it_old")


    Phi_ref = sol_ref.components[0]
    Phi_ref_o = sol_ref_o.components[0]
    Phi_ref_it_o = sol_ref_it_o.components[0]


    trials = VSpace.TrialFunction()
    tests  = VSpace.TestFunction()


    uPhi_ref = trials[0]
    vPhi_ref = tests[0]

    T_ref = sol_ref.components[1]
    T_ref_o = sol_ref_o.components[1]
    T_ref_it_o = sol_ref_it_o.components[1]
    uT_ref = trials[1]
    vT_ref = tests[1]
        

    
    H_ref = - grad(Phi_ref) + H0*HBS_vec + T_ref
    H_ref_o = - grad(Phi_ref_o) + H0_o*HBS_vec + T_ref_o
    H_red = T_ref - grad(Phi_ref) 
    H_red_it_old = T_ref_it_o - grad(Phi_ref_it_o) 

    J_ref = curl(T_ref)

    # ------------------------------------------------------------------------------
    # ------ Preisach model
    # ------------------------------------------------------------------------------
    ev = cp.preisachScalar.Everett_Lorentzian(401, 1640, 1.5)
    dist = cp.preisachVector.sphereLebedev(146)
    # create a half sphere as distribution of the half spheres 

    ev.Generate3DAdaption(4)
    with TaskManager():
        Preisach_ref = cp.preisachVector.ngPreisachVector3(mesh, intrule , H_ref, ev, dist, mask=mask)
    Preisach_ref.maxValueForOutside=ev.maxH*2
    # Preisach_ref.conventionalMatDiff = False
    
    print("ngPreisachVector3 passt mit ev: " + ev.name +", " + str(ev.NA) + ", isnonlin? " + str(ev.isNonLin) +" , perfDemag? " + str(Preisach_ref.usePerfectDemag) + ", prefValOnly? " + str(Preisach_ref.usePreviousForMuDiffOnly ))
    with TaskManager():
        dmu_nonlin=Preisach_ref.GetMuDiff()
        # mu_nonlin = Preisach_ref.GetMu()
        B_preisach = Preisach_ref.GetB(autoupdate=False)
        B_preisach_o = B_preisach.copy()

        H_preisach = Preisach_ref.GetH(autoupdate=False)


    Preisach_ref.Update()
    Preisach_ref.UpdatePast()

    mu_FP = dmu_nonlin.copy()
    


    E_ref = 1/sigmaFe*J_ref
    B_ref = mesh.MaterialCF({"core":B_preisach}, default= H_ref * muAir)
    # ------------------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------------------
    with TaskManager():
        # Bilinear form with Tracesf
        ah_ref = BilinearForm(VSpace, symmetric=True)


        # iron
        ah_ref += dt * 1/sigmaFe *curl(uT_ref) * curl(vT_ref) * dx("iron", intrules=irSpace) #ok
        ah_ref += mu_FP * (uT_ref - grad(uPhi_ref)) * (vT_ref - grad(vPhi_ref)) * dx("iron", intrules=irSpace) #ok

        # air
        ah_ref += muAir * grad(uPhi_ref) * grad(vPhi_ref) * dx("air", intrules=irSpace) #ok

        
        # regularisation / penalisation
        ah_ref += 1e-5 * uPhi_ref * vPhi_ref * dx("iron", intrules=irSpace) #ok

        prec_ref = Preconditioner(ah_ref, type = "direct")  
        ah_ref.Assemble()
        print("ah vec norm", ah_ref.mat.AsVector().Norm())

    # ------------------------------------------------------------------------------
    # --- right hand side
    # ------------------------------------------------------------------------------

    with TaskManager():
        f_ref = LinearForm(VSpace)

        # backward Euler 
        f_ref += dmu_nonlin*(T_ref_o + H0_o *HBS_vec - grad(Phi_ref_o)) * (vT_ref - grad(vPhi_ref)) * dx("iron", intrules=irSpace) 
        f_ref += muAir*(H0_o *HBS_vec - grad(Phi_ref_o)) * (- grad(vPhi_ref)) * dx("air", intrules=irSpace) 

        # Fixpoint
        f_ref += (mu_FP - dmu_nonlin)*(T_ref_it_o - grad(Phi_ref_it_o)) * (vT_ref -grad(vPhi_ref)) * dx("iron", intrules=irSpace) 

        #BSF
        f_ref += -dmu_nonlin * H0 * HBS_vec * (vT_ref -grad(vPhi_ref)) * dx("iron", intrules=irSpace) 
        f_ref += -muAir * H0 * HBS_vec * (-grad(vPhi_ref)) * dx("air", intrules=irSpace) 

    # ------------------------------------------------------------------------------
    # --- Data memory
    # ------------------------------------------------------------------------------
    q_i = np.zeros(N_ti)
    p_eddy_i = np.zeros(N_ti)
    p_hyst_i = np.zeros(N_ti)

    bi = np.zeros(N_ti)
    hi = np.zeros(N_ti)


    
    sol_ref_o.vec[:] = 0
    sol_ref_it_o.vec[:] = 0
    sol_ref.vec[:] = 0

    time_sim_i = np.zeros(N_ti)





    # ------------------------------------------------------------------------------
    # --- make it saveable
    # ------------------------------------------------------------------------------

    # B_ref_vec = cp.basicClasses.ngVector_class3(meshRef, intrule)
    # H_ref_vec = cp.basicClasses.ngVector_class3(meshRef, intrule)
    # J_ref_vec = cp.basicClasses.ngVector_class3(meshRef, intrule, mask = mask)

    # N_save_it = 1

    Preisach_ref.Demagnetise()
    print(f"--------------------{H0_amp}---------------------")
    with TaskManager():
        for i in range(N_ti):
            time_sim = time.time()

            print(f"\r ******************************* {i+1}/{N_ti} *******************************", end="")
            if i > 0:
                dt.Set(ti[i] - ti[i-1])
                mu_FP.Set(dmu_nonlin * 3)
            ah_ref.Assemble()

                

            # save old values
            sol_ref_o.vec.data = sol_ref.vec
            B_preisach_o.Set(B_preisach)
            B_preisach.autoUpdate = False
            H_preisach.autoUpdate = False


            # update HBS
            H0_o.Set(H0.Get())
            H0.Set(H0_amp * H0_i[i])



            from myPackage import myNonLinSolver
            def error_func(gfu, gfu_o):
                errL2 = Integrate(mask * InnerProduct((H_red - H_red_it_old), (H_red - H_red_it_old)), mesh, definedon=mesh.Materials("iron"))
                solL2 = Integrate(mask * InnerProduct(H_red, H_red), mesh, definedon=mesh.Materials("iron"))
                return solL2, errL2

            conv, it = myNonLinSolver(ah_ref, f_ref, sol_ref, prec_ref, gfu_o = sol_ref_it_o, eps=1e-3, Nmax=20, 
                    a_assemble=False, f_assemble=True, callback = lambda gfu, it: Preisach_ref.Update(), 
                    printrates=False, error_func=error_func, useBVP=True)


            # Updating Variables
            B_preisach.autoUpdate = True
            H_preisach.autoUpdate = True
            Preisach_ref.UpdatePast()

            time_sim = time.time()  - time_sim 
            time_sim_i[i] = time_sim



            q_i[i] = 1/2 * Integrate(InnerProduct(H_ref, B_ref), mesh, definedon=mesh.Materials("iron"))
            p_hyst_i[i] = Integrate(1/dt * InnerProduct(0.5 * (H_ref + H_ref_o), B_preisach-B_preisach_o), mesh, definedon=mesh.Materials("iron"))
            p_eddy_i[i] = Integrate(InnerProduct(E_ref, J_ref),mesh, definedon=mesh.Materials("iron"))

        






            # if(i % N_save_it == 0)  or i == N_ti -1:
            #     prefix = getPrefix(H0_amp)
            #     if not os.path.exists(prefix):
            #         os.makedirs(prefix)
            #     sol_ref.Save(prefix + f"sol_{i}.dat")

            #     toSave={ "i": i,
            #         "q_i":q_i[:i+1], "p_eddy_i": p_eddy_i[:i+1], "p_hyst_i":p_hyst_i[:i+1], "time_sim":sum(time_sim_i)}

            #     if i == 0:
            #         toSave.update({"order0":order0,   "ti":ti, "N_per_period": N_per_period, "N_periods": N_periods,
            #         "Vc":Vc, "VMS":VMS, "Vol":Vol,
            #         "meshRef":meshRef, "meshMS": meshMS,
            #         "H_KL":H_KL, "B_KL":B_KL, 
            #         "ev.name":ev.name, "ev.NA":ev.NA, "ev.isNonLin":ev.isNonLin, "ev.SUM":ev.GetSumOverAllElements(), 
            #         "omega":omega, "H0_i": H0_i, "sigmaFe":sigmaFe, "numSheets":numSheets,
            #         "N_save_it":N_save_it, 
            #         "mu_FP":mu_FP, "NVPM":Preisach_ref.CountVectorPreisachPoints()})
                    

            #         HBS_vec.Save(prefix + "HBS_vec.dat")
                

            #     pickle.dump(toSave, open( prefix + f"data_{i}.pkl", "wb" ))


        T = 2*np.pi/omega
        Q = 1/(Vc * T**2) * np.trapz(x=ti[-NperPeriod:], y=q_i[-NperPeriod:])


        P_eddy = 1/ (Vc * T) * np.trapz(x=ti[-NperPeriod:], y=p_eddy_i[-NperPeriod:])
        P_hyst = 1/(Vc*T) * np.trapz(x=ti[-NperPeriod:], y=p_hyst_i[-NperPeriod:])
        

        colorprint(f"simulation time {sum(time_sim_i):0.5} s", TextColor.YELLOW)
        colorprint(f"leave with \n Q:\t\t{Q} \n P_eddy:\t{P_eddy}\n P_hyst:\t{P_hyst}", TextColor.GREEN)

    print("f vec norm", f_ref.vec.Norm())



    return B_ref, H_ref, E_ref, J_ref, q_i, p_eddy_i, p_hyst_i, Q, P_eddy, P_hyst, ti, bi, mesh



def calcWithPhi(mesh, order0, H0_amp, HBS, ti, NperPeriod,omega=50*np.pi*2):


    print("--"*40)
    print("calc it with H0_amp=", H0_amp)
    print("--"*40)


    mip = mesh(0.05, 0, 0.5)
    
    mask = mesh.MaterialCF({"core":1}, default=0)
    muAir = 4e-7*np.pi
       


    intrule = IntegrationRule(HEX,2*order0+2)
    irSpace = {HEX:intrule}

    Vc = Integrate(mask, mesh)
    Vol = Integrate(1, mesh)

    
    # ------------------------------------------------------------------------------
    # --- excitation
    # ------------------------------------------------------------------------------
    H0_i = np.sin(omega * ti)
    dt = Parameter(ti[1] - ti[0])
    N_ti = len(ti)

    HBS_vec = HBS

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ reference solution
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    H0 = Parameter(0)
    H0_o = Parameter(0)
    print("order0", order0)




    dir_phi = ".*"
    dir_T = ".*"
    

    VPhi_ref = H1(mesh,order=order0+1, dirichlet=dir_phi)

    
    VSpace = FESpace([VPhi_ref])
        

    ndof = sum(VSpace.FreeDofs())	
    print('-----')  
    print('ndof =',ndof)
    print('-----')
    
    # Gridfunctions
    sol_ref = GridFunction(VSpace, "sol") 
    sol_ref_o = GridFunction(VSpace, "sol_old")
    sol_ref_it_o = GridFunction(VSpace, "sol_it_old")


    Phi_ref = sol_ref.components[0]
    Phi_ref_o = sol_ref_o.components[0]
    Phi_ref_it_o = sol_ref_it_o.components[0]


    trials = VSpace.TrialFunction()
    tests  = VSpace.TestFunction()


    uPhi_ref = trials[0]
    vPhi_ref = tests[0]


    
    H_ref = - grad(Phi_ref) + H0*HBS_vec 
    H_ref_o = - grad(Phi_ref_o) + H0_o*HBS_vec 
    H_red =  - grad(Phi_ref) 
    H_red_it_old = - grad(Phi_ref_it_o) 


    # ------------------------------------------------------------------------------
    # ------ Preisach model
    # ------------------------------------------------------------------------------
    ev = cp.preisachScalar.Everett_Lorentzian(401, 1640, 1.5)
    dist = cp.preisachVector.sphereLebedev(146)
    # create a half sphere as distribution of the half spheres 

    ev.Generate3DAdaption(4)
    with TaskManager():
        Preisach_ref = cp.preisachVector.ngPreisachVector3(mesh, intrule , H_ref, ev, dist, mask=mask)
    Preisach_ref.maxValueForOutside=ev.maxH*2
    # Preisach_ref.conventionalMatDiff = False
    
    print("ngPreisachVector3 passt mit ev: " + ev.name +", " + str(ev.NA) + ", isnonlin? " + str(ev.isNonLin) +" , perfDemag? " + str(Preisach_ref.usePerfectDemag) + ", prefValOnly? " + str(Preisach_ref.usePreviousForMuDiffOnly ))
    with TaskManager():
        dmu_nonlin=Preisach_ref.GetMuDiff()
        # mu_nonlin = Preisach_ref.GetMu()
        B_preisach = Preisach_ref.GetB(autoupdate=False)
        B_preisach_o = B_preisach.copy()

        H_preisach = Preisach_ref.GetH(autoupdate=False)


    Preisach_ref.Update()
    Preisach_ref.UpdatePast()

    mu_FP = dmu_nonlin.copy()
    


    B_ref = mesh.MaterialCF({"core":B_preisach}, default= H_ref * muAir)
    # ------------------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------------------
    with TaskManager():
        # Bilinear form with Tracesf
        ah_ref = BilinearForm(VSpace, symmetric=True)


        # iron
        ah_ref += mu_FP * grad(uPhi_ref) * grad(vPhi_ref) * dx("core", intrules=irSpace) #ok

        # air
        ah_ref += muAir * grad(uPhi_ref) * grad(vPhi_ref) * dx("coil|air", intrules=irSpace) #ok


        prec_ref = Preconditioner(ah_ref, type = "direct")  
        ah_ref.Assemble()
        print("ah vec norm", ah_ref.mat.AsVector().Norm())

    # ------------------------------------------------------------------------------
    # --- right hand side
    # ------------------------------------------------------------------------------

    with TaskManager():
        f_ref = LinearForm(VSpace)


        # Fixpoint
        f_ref += (mu_FP - dmu_nonlin)* grad(Phi_ref_it_o) * grad(vPhi_ref) * dx("core", intrules=irSpace) 

        #BSF
        f_ref += -dmu_nonlin * H0 * HBS_vec * ( -grad(vPhi_ref)) * dx("core", intrules=irSpace) 
        f_ref += -muAir * H0 * HBS_vec * (-grad(vPhi_ref)) * dx("coil|air", intrules=irSpace) 

    # ------------------------------------------------------------------------------
    # --- Data memory
    # ------------------------------------------------------------------------------
    q_i = np.zeros(N_ti)
    p_eddy_i = np.zeros(N_ti)
    p_hyst_i = np.zeros(N_ti)

    bi = np.zeros(N_ti)
    hi = np.zeros(N_ti)


    
    sol_ref_o.vec[:] = 0
    sol_ref_it_o.vec[:] = 0
    sol_ref.vec[:] = 0

    time_sim_i = np.zeros(N_ti)


    # ------------------------------------------------------------------------------
    # --- make it saveable
    # ------------------------------------------------------------------------------

    # B_ref_vec = cp.basicClasses.ngVector_class3(meshRef, intrule)
    # H_ref_vec = cp.basicClasses.ngVector_class3(meshRef, intrule)
    # J_ref_vec = cp.basicClasses.ngVector_class3(meshRef, intrule, mask = mask)

    # N_save_it = 1

    Preisach_ref.Demagnetise()
    print(f"--------------------{H0_amp}---------------------")
    with TaskManager():
        for i in range(N_ti):
            time_sim = time.time()

            print(f"\r ******************************* {i+1}/{N_ti} *******************************", end="")
            if i > 0:
                dt.Set(ti[i] - ti[i-1])
                mu_FP.Set(dmu_nonlin * 3)
            ah_ref.Assemble()

                

            # save old values
            sol_ref_o.vec.data = sol_ref.vec
            B_preisach_o.Set(B_preisach)
            B_preisach.autoUpdate = False
            H_preisach.autoUpdate = False


            # update HBS
            H0_o.Set(H0.Get())
            H0.Set(H0_amp * H0_i[i])



            from myPackage import myNonLinSolver
            def error_func(gfu, gfu_o):
                errL2 = Integrate(mask * InnerProduct((H_red - H_red_it_old), (H_red - H_red_it_old)), mesh, definedon=mesh.Materials("core"))
                solL2 = Integrate(mask * InnerProduct(H_red, H_red), mesh, definedon=mesh.Materials("core"))
                return solL2, errL2

            conv, it = myNonLinSolver(ah_ref, f_ref, sol_ref, prec_ref, gfu_o = sol_ref_it_o, eps=1e-3, Nmax=20, 
                    a_assemble=False, f_assemble=True, callback = lambda gfu, it: Preisach_ref.Update(), 
                    printrates=False, error_func=error_func, useBVP=True)
            print(f" num iterations {it} ", end = "\r")

            # Updating Variables
            B_preisach.autoUpdate = True
            H_preisach.autoUpdate = True
            Preisach_ref.UpdatePast()

            time_sim = time.time()  - time_sim 
            time_sim_i[i] = time_sim



            q_i[i] = 1/2 * Integrate(InnerProduct(H_ref, B_ref), mesh, definedon=mesh.Materials("core"))
            p_hyst_i[i] = Integrate(1/dt * InnerProduct(0.5 * (H_ref + H_ref_o), B_preisach-B_preisach_o), mesh, definedon=mesh.Materials("core"))
            p_eddy_i[i] = 0


            bi[i] = B_ref.Norm()(mip)
            hi[i] = H_ref.Norm()(mip)


            plt.figure(1)
            plt.clf()
            plt.plot(hi[:i+1], bi[:i+1])
            plt.pause(0.1)






        T = 2*np.pi/omega
        Q = 1/(Vc * T**2) * np.trapz(x=ti[-NperPeriod:], y=q_i[-NperPeriod:])


        P_eddy = 1/ (Vc * T) * np.trapz(x=ti[-NperPeriod:], y=p_eddy_i[-NperPeriod:])
        P_hyst = 1/(Vc*T) * np.trapz(x=ti[-NperPeriod:], y=p_hyst_i[-NperPeriod:])
        

        colorprint(f"simulation time {sum(time_sim_i):0.5} s", TextColor.YELLOW)
        colorprint(f"leave with \n Q:\t\t{Q} \n P_eddy:\t{P_eddy}\n P_hyst:\t{P_hyst}", TextColor.GREEN)

    print("f vec norm", f_ref.vec.Norm())



    return B_ref, H_ref, 0, 0, q_i, p_eddy_i, p_hyst_i, Q, P_eddy, P_hyst, ti, hi, bi, mesh

def run():

    ngmesh = MakeGeometry(modelcoil=True).GenerateMesh(maxh=0.5)
    # ngmesh.Save("coil.vol")
    mesh = Mesh(ngmesh)

    HBS = calcBiotSavartField(mesh)

    ngmesh = MakeGeometry(modelcoil=False).GenerateMesh(maxh=0.5)
    mesh = Mesh(ngmesh)

    Draw(HBS, mesh, "HBS")

    freq = 50
    T = 1/freq
    NperPeriod=600
    Nperiods=1.25
    ti = np.linspace(0, T*Nperiods, int(Nperiods*NperPeriod))
    ret = calcWithPhi(mesh, 0, H0_amp=1000, HBS=HBS,ti=ti, NperPeriod=NperPeriod,omega=2*np.pi*freq )



    from myPackage import myBreak
    myBreak(locals(), globals(), file=__file__.split('/')[-1])




if __name__ == "__main__":
    run()
