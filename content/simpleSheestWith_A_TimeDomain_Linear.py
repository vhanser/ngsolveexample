from netgen.occ import *
import numpy as np
from ngsolve import *
import netgen.gui
import cempy as cp
import matplotlib.pyplot as plt

SetNumThreads(2)
maxh = 1/10



freq = 50
mu0 = 4e-7*np.pi
nu0 = 1/mu0

mu_Air = mu0
mu_Fe = 1000*mu0

sigma_Fe =2e6


omega = freq*2*np.pi

delta = np.sqrt(2/(sigma_Fe*omega*1000*mu_Air))
Nsheets = 10
ff = 0.9
d = delta/2



order0 = 2

B0 = 1

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
A_ref = sol_ref.components[0] 
A_ref_o = sol_ref_o.components[0] 


trials = V.TrialFunction()
tests  = V.TestFunction()

uA = trials[0]
vA = tests[0]

# ------------------------------------------------------------------------------
# --- field quantities
# ------------------------------------------------------------------------------
gradA = grad(A_ref)
B = CF((gradA[1], -gradA[0])) 



nu = meshRef.MaterialCF({"iron.*":1/mu_Fe, "air":1/mu_Air}, default=0)
sigma = meshRef.MaterialCF({"iron.*":sigma_Fe}, default=0)
rho = 1/sigma


ti = np.linspace(0, 1/freq*1.25, 200)
dt = ti[1] - ti[0]
Bin = B0*np.sin(ti * omega)

E = (-1/dt * (sum(sol_ref.components) - sum(sol_ref_o.components)))[0]
H = nu * B
J = sigma * E


# ------------------------------------------------------------------------------
# Matrix
# ------------------------------------------------------------------------------
with TaskManager():
    # Bilinear form with 
    ah_ref = BilinearForm(V, symmetric=True)

    # A:
    ah_ref += dt * nu *grad(uA) * grad(vA) * dx

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



# ------------------------------------------------------------------------------
# ------ Solve It
# ------------------------------------------------------------------------------
#with TaskManager():
mip = meshRef(-dFe/2- d0/2, 0)

input("press to start")
Draw(B, meshRef,"B")
Draw(J, meshRef, "J" )


import time
with TaskManager():
    

    for i in range(len(Bin)):
        print("-------------", i, len(Bin)-1)
        sol_ref_o.vec.data = sol_ref.vec


        if excitation_orientation == "x":
            A_ref.Set(Bin[i]*y, BND)
        else:
            A_ref.Set(Bin[i]*x, BND)


            
        # solve it
        f_ref.Assemble()
        solvers.BVP(bf=ah_ref, lf=f_ref, gf=sol_ref, pre=prec, maxsteps=5)



        Redraw()
        time.sleep(0.1)

from myPackage import myBreak
myBreak(locals(), globals(), file=__file__.split('/')[-1])
    