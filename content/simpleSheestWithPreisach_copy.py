from netgen.occ import *
import numpy as np
from ngsolve import *
# import netgen.gui
import cempy as cp
import matplotlib.pyplot as plt

from myPackage import colorprint, TextColor, boundedCF

SetNumThreads(2)
maxh = 1/10



freq = 50
mu0 = 4e-7*np.pi
nu0 = 1/mu0

mu_Air = mu0

sigma_Fe =2e6


omega = freq*2*np.pi

delta = np.sqrt(2/(sigma_Fe*omega*1000*mu_Air))
Nsheets = 1
ff = 0.9
d = delta/2



order0 = 2

B0 = 0.1

dFe = d*ff
d0 = d*(1-ff)

H_core = Nsheets*dFe + (Nsheets-1)*d0
W_core = H_core

W = 2*W_core
H = 2*H_core

wp = WorkPlane()
outer = wp.RectangleC(W, H).Face()
outer.name = "air"
outer.edges.Max(X).name = "right"
outer.edges.Min(X).name = "left"
outer.edges.Max(Y).name = "top"
outer.edges.Min(Y).name = "bottom"


rec_sheets =[]
x_pos = - W_core/2


for i in range(Nsheets):
    wp.MoveTo(x_pos, -H_core/2)

    rec_sheets.append(wp.Rectangle(dFe, H_core).Face())
    rec_sheets[-1].name = f"iron{i}"


    x_pos += d

print(x_pos, W_core/2)

rec_sheets = Glue(rec_sheets)
rec_sheets.edges.maxh = delta/10

geo = Glue([outer - rec_sheets, rec_sheets])



meshRef = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=delta))
# Draw(meshRef.MaterialCF({"iron.*":1, "air":2}, default=0), meshRef)
# Draw(x, meshRef)


print("domains", set(meshRef.GetMaterials()))
print("bnds", set(meshRef.GetBoundaries()))

print("penetration depth", delta)

excitation_orientation = "y"

# ------------------------------------------------------------------------------
# --- Excitation
# ------------------------------------------------------------------------------
H0_amp = 1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ reference solution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("order0", order0)
print("numSheets", Nsheets)

if excitation_orientation == "x":
    dir_A = "top|bottom"
else:
    dir_A = "left|right"
VA = H1(meshRef,order=order0, dirichlet=dir_A)
VNum = []

for i in range(Nsheets):
    VNum.append(NumberSpace(meshRef, definedon=meshRef.Materials(f"iron{i}")))


V = FESpace([VA] + VNum)
ndof = V.ndof	
print(f"VA  :{VA.ndof}")    
print(f"ndof  :{ndof}")    

# Gridfunctions
sol_ref = GridFunction(V, "sol") 
sol_ref_o = GridFunction(V, "sol") 
sol_ref_it_o = GridFunction(V, "sol") 
from myPackage import myBreak
myBreak(locals(), globals(), file=__file__.split('/')[-1])
A_ref = sol_ref.components[0] 
A_ref_o = sol_ref_o.components[0] 
A_ref_it_o = sol_ref_it_o.components[0] 



trials = V.TrialFunction()
tests  = V.TestFunction()

uA = trials[0]
vA = tests[0]

# ------------------------------------------------------------------------------
# --- field quantities
# ------------------------------------------------------------------------------
gradA = grad(A_ref)
B = CF((gradA[1], -gradA[0])) 


ev = cp.preisachScalar.Everett_Lorentzian(401, 1640, 1.5)

filename = "save/EverettMatrix_inverse"+  ev.name + "_" + str(ev.NA)  + ("_nonlin" if ev.isNonLin else "")

iev1D = ev.copy(False)
iev2D = ev.copy(False)
ev2D = ev.copy(False)

if not iev2D.Load(filename + "_2D.bin"):
    
    if not iev2D.Load(filename + "_1D.bin"):
        p = cp.preisachScalar.preisach(ev)
        print("calculate 1D inverse")
        iev2D = p.getInverseEverettMatrix()
        iev2D.Save(filename + "_1D.bin")
    else:
        print("loaded 1D")


    print("calculate 2D inverse")
    iev2D.Generate2DAdaption(4)
    iev2D.Save(filename + "_2D.bin")
else:
    if not iev1D.Load(filename + "_1D.bin"):
        p = cp.preisachScalar.preisach(ev)
        print("calculate 1D inverse")
        iev1D = p.getInverseEverettMatrix()
    print("loaded 2D")









mask = meshRef.MaterialCF({"iron.*":1}, default=0)

intrule = IntegrationRule(TRIG, 2*order0)
dist = cp.preisachVector.circleDistribution(40)

Binput = B
Preisach_ref = cp.preisachVector.ngPreisachVector2(meshRef, intrule, Binput, iev2D, dist, field="B", mask = mask )

nu_preisach = Preisach_ref.GetNu()
H_preisach = Preisach_ref.GetH()


nu = meshRef.MaterialCF({"iron.*":nu_preisach, "air":1/mu_Air}, default=0)
sigma = meshRef.MaterialCF({"iron.*":sigma_Fe}, default=0)
rho = 1/sigma


ti = np.linspace(0, 1/freq*1.25, 600)
dt = ti[1] - ti[0]
Bin = B0*np.sin(ti * omega)

E = (-1/dt * (sum(sol_ref.components) - sum(sol_ref_o.components)))[0]
H = meshRef.MaterialCF({"iron.*":H_preisach, "air":1/mu_Air  * B}, default=0)
J = sigma * E



fesFP = MatrixValued(L2(meshRef, definedon=meshRef.Materials("iron.*")))
nu_FP = GridFunction(fesFP)
nu_FP_init = 1640/1.5
nu_FP.Set(CF((nu_FP_init, 0, 0, nu_FP_init), dims=(2,2)))

from myPackage import myBreak
myBreak(locals(), globals(), file=__file__.split('/')[-1])


# ------------------------------------------------------------------------------
# Matrix
# ------------------------------------------------------------------------------
with TaskManager():
    # Bilinear form with 
    ah_ref = BilinearForm(V, symmetric=True)

    # A:
    ah_ref += dt * nu_FP *grad(uA) * grad(vA) * dx("iron.*")
    ah_ref += dt * nu0 *grad(uA) * grad(vA) * dx("air")

    # lagrange multipliers
    for i in range(Nsheets):
        ah_ref += sigma_Fe * (uA + trials[1+i]) * (vA + tests[1+i]) * dx(f"iron{i}")


    prec = Preconditioner(ah_ref, type="direct")
    ah_ref.Assemble()

print(ah_ref.mat.AsVector().Norm())

f_ref = LinearForm(V) 


# lagrange multipliers
for i in range(Nsheets):
    # Euler
    f_ref += sigma_Fe * (A_ref_o + sol_ref_o.components[i+1]) * (vA  + tests[i+1]) * dx(f"iron{i}")

    # nonlin
    gradvA = grad(vA)
    curlvA = CF((gradA[1], -gradA[0]))
    f_ref += dt * (nu_FP * grad(A_ref)) * grad(vA) * dx(f"iron{i}")
    f_ref += dt * (- H_preisach)* curlvA * dx(f"iron{i}")







# ------------------------------------------------------------------------------
# ------ Solve It
# ------------------------------------------------------------------------------

if Nsheets % 2 == 0:
    mip = meshRef(-dFe/2- d0/2, 0)
else:
    mip = meshRef(0, 0)

myBreak(locals(), globals(), file=__file__.split('/')[-1])  

Draw(B, meshRef,"B")

Draw(J, meshRef, "J" )
Draw(B.Norm(), meshRef,"Bnorm")

Hout = []
Bout = []
with TaskManager():
    

    for i in range(len(Bin)):
        print("-------------", i, len(Bin)-1)
        sol_ref_o.vec.data = sol_ref.vec

        if i> 1:
            # nu_FP.Set(IfPos(Norm(nu_preisach[0]) + Norm(nu_preisach[3]), nu_preisach, nu_FP))

            ah_ref.Assemble()
            None


        if excitation_orientation == "x":
            A_ref.Set(Bin[i]*y, BND)
        else:
            A_ref.Set(Bin[i]*x, BND)

        for it in range(100):

            sol_ref_it_o.vec.data = sol_ref.vec
            
            
            # solve it
            f_ref.Assemble()
            solvers.BVP(bf=ah_ref, lf=f_ref, gf=sol_ref, pre=prec, maxsteps=5, print=False)
            Preisach_ref.Update()
            
            errL2 = Integrate(InnerProduct(A_ref - A_ref_it_o, A_ref - A_ref_it_o), meshRef, definedon=meshRef.Materials("iron.*"))
            solL2 = Integrate(InnerProduct(A_ref, A_ref), meshRef, definedon=meshRef.Materials("iron.*"))


            if solL2 == 0:
                break


            # print(it, errL2/solL2)
            if errL2/solL2 < 1e-5:
                colorprint("has convereged", TextColor.GREEN)
                break
        else:
            colorprint("has not converged", TextColor.RED)
            
            


        Preisach_ref.UpdatePast()


        Bout.append(B(mip)[1])
        Hout.append(H(mip)[1])
            


        # Redraw()



        # from myPackage import myBreak
        # myBreak(locals(), globals(), file=__file__.split('/')[-1])
        if i % 10 == 0:
            plt.clf()
            plt.plot(Hout, Bout)
            plt.pause(0.1)
        

myBreak(locals(), globals(), file=__file__.split('/')[-1])