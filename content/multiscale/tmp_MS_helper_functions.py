from ngsolve import *
import numpy as np


def H1Curl(scal_GF, coordinate):
    tmp = grad(scal_GF)
    if coordinate == 0:
        ret = CF((0, tmp[2], -tmp[1]))
    elif coordinate == 1:
        ret = CF((-tmp[2], 0, tmp[0]))
    elif coordinate == 2:
        ret = CF((tmp[1], -tmp[0], 0))
    return ret


def PhiH1Curl(phi, scal_GF, coordinate):
    return phi * H1Curl(scal_GF, coordinate)

def dzPhiezCrossCF(scal_z, b_CF):
    # cmdInput(locals(), globals(), "tea")
    return CF((-scal_z*b_CF[1], scal_z*b_CF[0], 0))

def dyPhieyCrossCF(scal_y, b_CF, for2D=False):
    # cmdInput(locals(), globals(), "tea")
    if for2D:
        return -scal_y*b_CF[0]
    else:
        return CF((scal_y*b_CF[2], 0, -scal_y*b_CF[0]))

# 2D1D
def get2DFieldFrom2D1DContainer(cont, level):
    return sum([comp[0] * comp[1].DirectEvaluate(level) for comp in cont])

def SimpleDrawTo3D_Vec(cf, mesh):
    import cempy as cp

    cf_x = cp.SimpleDrawTo3D(cf[0], mesh)
    cf_y = cp.SimpleDrawTo3D(cf[1], mesh)
    cf_z = cp.SimpleDrawTo3D(cf[2], mesh)

    if cf.is_complex:
        return CF((cf_x, cf_y, cf_z))
    else:
        return CF((cf_x, cf_y, cf_z)).real

def get3DFieldFrom2D1DContainer(cont, mesh2D):
    return sum([SimpleDrawTo3D_Vec(comp[0], mesh2D)*comp[1] for comp in cont])




class pyPhiFunction(CF):

    def __init__(self, zstart, d, fun, numSheets, ff, orientation, inIron=True, inAir=True, inOuter=False, douter=None, modelHalfAir=True, name = "pyPhiFunction", **kwargs):
        self.zstart = zstart
        self.xStart = zstart
        self.d = d
        self.fun = fun
        self.numSheets = numSheets
        self.ff = ff
        self.orientation = orientation
        self.inIron = inIron
        self.inAir = inAir
        self.inOuter = inOuter
        self.douter = douter
        self.modelHalfAir=modelHalfAir

        self.dFe = d*ff
        self.d0 = d*(1-ff)

        self.name = name


        self.fullPhi = False
        if "fullPhi" in kwargs.keys():
            self.fullPhi = kwargs["fullPhi"]
            kwargs.pop("fullPhi")

        if len(kwargs.keys()) != 0:
            print("unknown kewords:")
            print(kwargs.keys())


        self.createMask()
        self.createLocX()
        self.fun = fun
            
        super().__init__(self.createCFs())


    def createMask(self):
        x_CF = (x, y, z)[self.orientation]
        if self.modelHalfAir:
            self.mask = IfPos(x_CF - self.xStart, IfPos(x_CF + self.xStart, 0, 1), 0) 
        else:
            self.mask = IfPos(x_CF - self.xStart - self.d0/2, IfPos(x_CF + self.xStart + self.d0/2, 0, 1), 0) 
    def createCFs(self):

        funFe = self.fun[0] if self.inIron else lambda x : CF(0)
        funAir = self.fun[1] if self.inAir else lambda x : CF(0)

        # dxfunFe = self.dxfun[0] if self.inIron else lambda x : CF(0)
        # dxfunAir = self.dxfun[1] if self.inAir else lambda x : CF(0)

        self.phi = IfPos(self.localX - self.dFe/2, funAir(self.localX), funFe(self.localX)) * self.mask
        # self.dzphi = IfPos(self.localX - self.dFe/2, dxfunAir(self.localX), dxfunFe(self.localX)) * self.mask
        return self.phi #, self.dzphi


    def createLocX(self):
        # get coordinate
        x_CF = (x, y, z)[self.orientation]
        # start point

        if self.modelHalfAir:
            xPos = self.xStart - self.d0/2 - self.dFe
            istart = -1
        else:
            xPos = self.xStart - self.d0/2
            istart = 0

        ret = CF(0)
        for i in range(istart, self.numSheets):

            sheetOrigin = x_CF - xPos - self.dFe/2
            # ret += IfPos(tmp + self.dFe/2, IfPos(tmp - self.dFe/2 - self.d0, 0, 1), 0)
            ret += IfPos(sheetOrigin  + self.dFe/2 , IfPos(sheetOrigin  - self.dFe/2 - self.d0, 0, sheetOrigin), 0)

            xPos += self.d0 + self.dFe

        ret *= self.mask

        self.localX = ret

        return self.localX

    def map2sheet(self, x):
        # map to sheet
        x = x  - self.xStart - self.d0/2
        
        while x > self.d:
            x -= self.d



        while x < 0:
            x += self.d
        
        x -= self.dFe/2


        return x

    def DirectEvaluate(self, x):
        #outside 
        if x < self.xStart or x > -self.xStart:
            return 0

        x = self.map2sheet(x)

        # iron or air
        if x <= self.dFe/2 :
            return self.fun[0](x) if self.inIron else 0
        else: 
            return self.fun[1](x) if self.inAir else 0


class pydxLobatto(pyPhiFunction):
    def __init__(self, zstart, d, order, numSheets, ff, orientation, inIron=True, inAir=True, inOuter=False, douter=None, modelHalfAir=True): 

        dFe = d*ff
        d0 = d*(1-ff)
        xs = (d0/2 + dFe/2)
        self.order = order
        self.dxFunDict = {
            0: [lambda x : 0, lambda x : 0],
            1: [lambda x : 2.0/dFe , lambda x : -2.0/d0 ],
            2: [lambda x : sqrt(3.0/8)*(8/(dFe**2) *x), lambda x : sqrt(3.0/8)*(8/(d0**2) *(x-dFe/2-d0/2))]
        }   

        super().__init__(zstart, d, self.dxFunDict[self.order], numSheets, ff, orientation, inIron=inIron, inAir=inAir, 
                        inOuter=inOuter, douter=douter, modelHalfAir=modelHalfAir, name="pydxLobatto")

class pyLobatto(pyPhiFunction):
    def __init__(self, zstart, d, order, numSheets, ff, orientation, inIron=True, inAir=True, inOuter=False, douter=None, modelHalfAir=True): 
        dFe = d*ff
        d0 = d*(1-ff)
        xs = (d0/2 + dFe/2)
        self.order = order
        self.funDict = {
            0: [lambda x : 1, lambda x : 1],
            1: [lambda x :  2.0/dFe * x, lambda x : -2.0/d0 * (x - xs)],
            2: [lambda x : sqrt(3.0/8) * (4/(dFe**2) *(x**2)- 1), lambda x : sqrt(3.0/8) * (4/(d0**2)*((x-xs)**2)- 1)]
        }

        super().__init__(zstart, d, self.funDict[self.order], numSheets, ff, orientation, inIron=inIron, inAir=inAir, 
                                inOuter=inOuter, douter=douter, modelHalfAir=modelHalfAir, name="pyLobatto")

        



# common
# construct a cl_Phi from Lobatto polynomials
# phi_Fe = Lobatto(zstart, d, 2 , numSheets, ff, 2, inIron=True, inAir=False)
# phi_0 = Lobatto(zstart, d, 3 , numSheets, ff, 2, inIron=False, inAir=True)
# dzphi_Fe = dxLobatto(zstart, d, 2 , numSheets, ff, 2, inIron=True, inAir=False, name="mydzBase_Fe")
# dzphi_0 = dxLobatto(zstart, d, 3 , numSheets, ff, 2, inIron=True, inAir=False, name="mydzBase_0")
# myBase = pyPhiFunction(zstart, d, phi_Fe.DirectEvaluate , phi_0.DirectEvaluate , numSheets, ff, 2, inIron=True, inAir=True, name="myBase")
# mydzBase = pyPhiFunction(zstart, d, dzphi_Fe.DirectEvaluate , dzphi_0.DirectEvaluate , numSheets, ff, 2, inIron=True, inAir=True, name="mydzBase")
# cl_Phi([myBase, mydzBase])

class cl_Phi:
    """ microshape functions for multiscale approaches

        define class parameters before using the class:
        cl_Phi.dFe = ...
        cl_Phi.d0 = ....
        cl_Phi.numSheets = ... 
        cl_Phi.mesh = ... 


        pydzPhi = cp.phiFunctions.pyPhiFunction(-d/2, d, lambda x: -2/dFe, lambda x: 2/d0,1,  ff, 0)
        pyPhi = cp.phiFunctions.pyPhiFunction(-d/2, d, lambda x: -x*2/dFe, lambda x: 2/d0 * (x - d/2),1,  ff, 0)
        c = cl_Phi([pyPhi, pydzPhi], fes_order=0)
        c.plot(phi = True, dzPhi=True)
        plt.show()

    """
    numSheets = None
    dFe = None 
    d0 = None
    douter = np.nan
    mesh = None
    modelHalfAir = True
    phiFunction = pyLobatto
    dzPhiFunction = pydxLobatto
    orientation = 2
    useGradients = True
    useAbsolutes = True

    def __init__(self, phi_order, fes_order, inIron=True, inAir=True, inOuter=False, material=".*", dirichlet="", **kwargs):
        if isinstance(phi_order, (int, float)):

            if any([k not in ["numSheets", "dFe", "d0", "phiFun", "dxphiFun", "dyphiFun", "dzphiFun", "orientation", "zstart", "douter", "modelHalfAir", "phiFunction", "dxPhiFunction", "dyPhiFunction", "dzPhiFunction", "checkValidBoundaryAndDomainNames", "useGradients", "useAbsolutes"] for k in kwargs.keys()]):
                raise ValueError("undocumented keyword in cl_Phi class")

            if "numSheets" in kwargs.keys():
                self.numSheets = kwargs["numSheets"]
            else:
                self.numSheets = cl_Phi.numSheets
            
            if "dFe" in kwargs.keys():
                self.dFe = kwargs["dFe"]
            else:
                self.dFe = cl_Phi.dFe

            if "d0" in kwargs.keys():
                self.d0 = kwargs["d0"]
            else:
                self.d0 = cl_Phi.d0

            if "douter" in kwargs.keys():
                self.douter = kwargs["douter"]
            else:
                self.douter = cl_Phi.douter

            if "orientation" in kwargs.keys():
                self.orientation = kwargs["orientation"]
            else:
                self.orientation = cl_Phi.orientation

            if "modelHalfAir" in kwargs.keys():
                self.modelHalfAir = kwargs["modelHalfAir"]
            else:
                self.modelHalfAir = cl_Phi.modelHalfAir


            if self.numSheets == None or self.dFe == None or self.d0 == None:
                raise ValueError("set numSheets, dFe and d0 first -> cl_Phi.numSheets = N, cl_Phi.dFe = ... ")

            if "phiFunction" in kwargs.keys():
                self.phiFunction = kwargs["phiFunction"]
            else:
                self.phiFunction=cl_Phi.phiFunction
            
            
            if "dxPhiFunction" in kwargs.keys():
                self.dzPhiFunction = kwargs["dxPhiFunction"]
            elif "dyPhiFunction" in kwargs.keys():
                self.dzPhiFunction =  kwargs["dyPhiFunction"]
            elif "dzPhiFunction" in kwargs.keys():
                self.dzPhiFunction =  kwargs["dzPhiFunction"]           
            else:
                self.dzPhiFunction = cl_Phi.dzPhiFunction 

            if "useGradients" in kwargs.keys():
                self.useGradients = kwargs["useGradients"]
            else:
                self.useGradients=cl_Phi.useGradients

            if "useAbsolutes" in kwargs.keys():
                self.useAbsolutes = kwargs["useAbsolutes"]
            else:
                self.useAbsolutes=cl_Phi.useAbsolutes


                

            if "zstart" in kwargs.keys():
                self.zstart = kwargs["zstart"]
            else:
                self.zstart = -self.numSheets*(self.dFe+self.d0)/2  

            checkValidBoundaryAndDomainNames = True
            if "checkValidBoundaryAndDomainNames" in kwargs.keys():
                checkValidBoundaryAndDomainNames = kwargs["checkValidBoundaryAndDomainNames"]
                kwargs.pop("checkValidBoundaryAndDomainNames")


            self.ff = self.dFe/(self.d0+self.dFe)

                        

            try:
                self.phi = self.phiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir)
                self.dzphi = self.dzPhiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir)
            except:
                self.phi = self.phiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir)
                self.dzphi = self.dzPhiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir)
            self.name = self.phi.name

        else:
            self.phi = phi_order[0]
            self.dzphi = phi_order[1]

            

            self.orientation = self.phi.coordinate
            self.zstart = self.phi.xStart
            self.name = self.phi.name

            if self.phi.inAir != inAir:
                print("warning: cl_Phi parameter in Air is ignored. Set it in definition of pyPhiFunction")
            if self.phi.inIron != inIron:
                print("warning: cl_Phi parameter in Air is ignored. Set it in definition of pyPhiFunction")

        self.fes_oder = fes_order
        self.inIron = self.phi.inIron
        self.inAir = self.dzphi.inAir
        self.dirichlet = dirichlet
        self.material = material
        
        if cl_Phi.mesh != None and checkValidBoundaryAndDomainNames:
            self.checkValidBoundaryAndDomainNames(cl_Phi.mesh)
        else:
            if cl_Phi.mesh == None:
                print("set cl_Phi.mesh to auto check boundary names and dirichlet names")



    # def __str__(self):
    #     s = str("name") + self.name
    #     s = str("in Iron")
    #     s = str("in Iron")
    #     return s

    def plot(self, phi=True, dzPhi=False, N=30):
        import matplotlib.pyplot as plt
        
        if phi:
            data = np.array(self.phi.getPlotData(N))    
            plt.plot(data[:, 0], data[:, 1], '-o', label=self.phi.name)
        if dzPhi:
            data = np.array(self.dzphi.getPlotData(N))    
            plt.plot(data[:, 0], data[:, 1], '-o', label=self.dzphi.name)
        

        plt.legend()

        return plt.gcf()

    def plotEvaluated(orderPhi, xi= None, nFig=1):
        import matplotlib.pyplot as plt

        if len(orderPhi) == 0:
            return

        if type(xi) == type(None):
            xi = np.linspace(orderPhi[0].zstart, -orderPhi[0].zstart, 10000)
        plt.figure(nFig)
        plt.clf()
        
        plt.subplot(2, 1, 1)
        plt.title("phis for Phi")

        xpos = orderPhi[0].zstart + cl_Phi.d0/2
        d = orderPhi[0].d0 + orderPhi[0].dFe
        for i in range(orderPhi[0].numSheets):
            plt.fill([xpos, xpos+cl_Phi.dFe, xpos+cl_Phi.dFe, xpos], [-1, -1, 1, 1], c="lightgray")
            plt.plot([xpos, xpos], [-1, 1], "--k")
            plt.plot([xpos+cl_Phi.dFe, xpos+cl_Phi.dFe], [-1, 1], "--k")
            xpos += d

        # [plt.plot(xi, [o.phi.DirectEvaluate(xx) for xx in xi], label=o.phi.name) for o in orderPhi ]

        if cl_Phi.orientation == 0:
            [plt.plot(xi, [o.phi(cl_Phi.mesh(xx, 0, 0)) for xx in xi], label=o.phi.name) for o in orderPhi ]
        if cl_Phi.orientation == 1:
            [plt.plot(xi, [o.phi(cl_Phi.mesh(0, yy, 0)) for yy in xi], label=o.phi.name) for o in orderPhi ]
        if cl_Phi.orientation == 2:
            [plt.plot(xi, [o.phi(cl_Phi.mesh(0, 0, zz)) for zz in xi], label=o.phi.name) for o in orderPhi ]
        
        plt.subplot(2, 1, 2)
        xpos = orderPhi[0].zstart + cl_Phi.d0/2
        for i in range(orderPhi[0].numSheets):
            plt.fill([xpos, xpos+cl_Phi.dFe, xpos+cl_Phi.dFe, xpos], [-2/cl_Phi.dFe, -2/cl_Phi.dFe, 2/cl_Phi.dFe, 2/cl_Phi.dFe], c="lightgray")
            plt.plot([xpos, xpos], [-1, 1], "--k")
            plt.plot([xpos+cl_Phi.dFe, xpos+cl_Phi.dFe], [-1, 1], "--k")
            xpos += d

        # [plt.plot(xi, [o.dzphi.DirectEvaluate(x) for x in xi], label=o.phi.name) for o in orderPhi ]
        if cl_Phi.orientation == 0:
            [plt.plot(xi, [o.dzphi(cl_Phi.mesh(xx, 0, 0)) for xx in xi], label=o.dzphi.name) for o in orderPhi ]
        if cl_Phi.orientation == 1:
            [plt.plot(xi, [o.dzphi(cl_Phi.mesh(0, yy, 0)) for yy in xi], label=o.dzphi.name) for o in orderPhi ]
        if cl_Phi.orientation == 2:
            [plt.plot(xi, [o.dzphi(cl_Phi.mesh(0, 0, zz)) for zz in xi], label=o.dzphi.name) for o in orderPhi ]
        plt.legend()
        # [o.plot() for o in orderPhi]


        plt.show()

    def __str__(self):

        ret = "cl_Phi parameter\n=====\n"
        ret += f"numSheets\t{cl_Phi.numSheets}\n"
        ret += f"dFe\t{cl_Phi.dFe}\n"
        ret += f"d0\t{cl_Phi.d0}\n"
        ret += f"douter\t{cl_Phi.douter}\n"
        ret += f"mesh\t{cl_Phi.mesh}\n"
        ret += f"modelHalfAir\t{cl_Phi.modelHalfAir}\n"
        ret += f"phiFunction\t{cl_Phi.phiFunction}\n"
        ret += f"dzPhiFunction\t{cl_Phi.dzPhiFunction}\n"
        ret += f"orientation\t{cl_Phi.orientation}\n"
        ret += "\n\ninstance parameter\n=====\n"


        ret += f"self.d0\t{self.d0}\n"
        ret += f"self.dFe\t{self.dFe}\n"
        ret += f"self.dirichlet\t{self.dirichlet}\n"
        ret += f"self.douter\t{self.douter}\n"
        ret += f"self.dzphi\t{self.dzphi.name}\n"
        ret += f"self.dzPhiFunction\t{self.dzPhiFunction}\n"
        ret += f"self.fes_oder\t{self.fes_oder}\n"
        ret += f"self.ff\t{self.ff}\n"
        ret += f"self.inAir\t{self.inAir}\n"
        ret += f"self.inIron\t{self.inIron}\n"
        ret += f"self.material\t{self.material}\n"
        ret += f"self.mesh\t{self.mesh}\n"
        ret += f"self.modelHalfAir\t{self.modelHalfAir}\n"
        ret += f"self.name\t{self.name}\n"
        ret += f"self.numSheets\t{self.numSheets}\n"
        ret += f"self.orientation\t{self.orientation}\n"
        ret += f"self.phi\t{self.phi.name}\n"
        ret += f"self.phiFunction\t{self.phiFunction}\n"
        ret += f"self.zstart\t{self.zstart}\n"


        return ret
    def __repr__(self):

        return self.name


    def checkValidBoundaryAndDomainNames(self, mesh):
        checkValidBoundaryAndDomainNames([self], mesh)



class gradgradMS():
    def __init__(self, orderPhi, sol, **kwargs):

        
        self.orderPhi = orderPhi
        self.dim = sol.space.mesh.dim
        self.sol = sol
        sol_c = self.sol.components
        self.trials, self.tests = sol.space.TnT()
        
        # ------------------------------------------------------------------------------
        # --- kwargs
        # ------------------------------------------------------------------------------
        self.orientation = cl_Phi.orientation
        if "self.orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]
            kwargs.pop("orientation")

        istart = 0
        if "istart" in kwargs.keys():
            istart = kwargs["istart"]
            kwargs.pop("istart")

        self.ansatz  = ""
        if "ansatz" in kwargs.keys():
            self.ansatz = kwargs["ansatz"]
            kwargs.pop("ansatz")

        addPhi0Outer = True
        if "addPhi0Outer" in kwargs.keys():
            addPhi0Outer = kwargs["addPhi0Outer"]
            kwargs.pop("addPhi0Outer")

        # ------------------------------------------------------------------------------
        # --- end kwargs
        # ------------------------------------------------------------------------------
    
        if len(kwargs.keys()) != 0:
            print("unknown kewords:")
            print(kwargs.keys())



        if self.dim == 1:
            self.e = 1
        elif self.dim == 2:
            if self.orientation == 0:
                self.e = CF((1, 0))
            else:
                self.e = CF((0, 1))
        elif self.dim == 3:
            if self.orientation == 0:
                self.e = CF((1, 0, 0))
            elif self.orientation == 1:
                self.e = CF((0, 1, 0))
            else:
                self.e = CF((0, 0, 1))       

    
        self.u = []
        self.v = []

        self.gradu = []
        self.gradv = []
        self.gradu_trace = []
        self.gradv_trace = []
        

        self.gradsol_comp = []
        self.gradsol_pack = []

        self.sol_comp = []
        self.sol_pack = []



        i = istart
        if addPhi0Outer:
            clPhi0 = cl_Phi(0, 0, material=".*") 
            phi0 = clPhi0.phi
            # phi 0 outer
            self.gradu += [[-grad(self.trials[i]), phi0]] # ok
            self.gradv += [[-grad(self.tests[i]), phi0]]

            self.gradsol_comp.append(-grad(sol_c[i]))
            self.sol_comp.append(sol_c[i] )

            self.u += [[self.trials[i], phi0]]
            self.v += [[self.tests[i], phi0]]

            self.sol_pack += [[sol_c[i], phi0]]
            self.gradsol_pack += [[-grad(sol_c[i]), phi0]]

            self.ansatz += "-grad(Phi0_outer)"
            i+= 1

        
        # Phi: grad(Phi*phi) = grad(Phi) * phi + Phi * dy phi ey
        for phi_i  in orderPhi:
            phi = phi_i.phi
            dxphi = phi_i.dzphi

            if phi_i.useAbsolutes:
                self.gradu += [[-self.trials[i]*self.e,  dxphi]] # ok
                self.gradv += [[-self.tests[i]*self.e,  dxphi]]
                self.gradu_trace += [[-self.trials[i]*self.e,  dxphi]] # ok
                self.gradv_trace += [[-self.tests[i]*self.e,  dxphi]]

                self.gradsol_comp.append(-(dxphi*self.e*sol_c[i]))
                self.gradsol_pack += [[-sol_c[i]*self.e, dxphi]]

            if phi_i.useGradients:
                self.gradu += [[-grad(self.trials[i]), phi]] # ok
                self.gradv += [[-grad(self.tests[i]), phi]]
                self.gradu_trace += [[-grad(self.trials[i]).Trace(), phi]] # ok
                self.gradv_trace += [[-grad(self.tests[i]).Trace(), phi]]



                self.gradsol_comp.append(-(phi *grad(sol_c[i]) ))
                self.gradsol_pack += [[-grad(sol_c[i]), phi]]



            self.u += [[self.trials[i], phi]]
            self.v += [[self.tests[i], phi]]

            self.sol_comp.append(phi *sol_c[i])
            self.sol_pack += [[sol_c[i], phi]]
            
            self.ansatz += " - grad(Phi" + str(i) + " * "+ phi.name + ")"
        
            i +=1   
        

def getPeriodicH1BaseFunctions(order):
    from ngsolve import H1, L2
    return getPeriodicBaseFunctions(order, H1, L2)

def getPeriodicL2BaseFunctions(order):
    from ngsolve import L2
    return getPeriodicBaseFunctions(order, L2, L2)


def getPeriodicBaseFunctions(order, space, dzspace):
    import netgen.meshing as nm
    from ngsolve import GridFunction, Mesh, Periodic, grad
    mesh1D = nm.Mesh(dim=1)
    pids = []
    tmp = getMeshData()
    dFe = sum(tmp["z1_iron"])*tmp["scale"]
    d0 = sum(tmp["z1_air"])*tmp["scale"]
    
    # mesh 
    pids.append(mesh1D.Add (nm.MeshPoint(nm.Pnt(-dFe/2, 0, 0))))
    pids.append(mesh1D.Add (nm.MeshPoint(nm.Pnt(dFe/2, 0, 0))))
    pids.append(mesh1D.Add (nm.MeshPoint(nm.Pnt(dFe/2+d0, 0, 0))))

    
    for i in range(len(pids)-1):
        mesh1D.Add(nm.Element1D([pids[i],pids[i+1]],index=1))


    mesh1D.Add (nm.Element0D( pids[0], index=1))
    mesh1D.Add (nm.Element0D( pids[-1], index=2))
    mesh1D.AddPointIdentification(pids[0],pids[-1],1,2)
    
    mesh1D = Mesh(mesh1D)

    # base functions
    fes = Periodic(space(mesh1D, order=order))
    fesL2 = Periodic(dzspace(mesh1D, order=order))

    gfu = GridFunction(fes)
    N = len(gfu.vec)

    
    baseFunc_GF = []
    dzbaseFunc_GF = []
    

    baseFunc = []
    dzbaseFunc = []
    
    for i in range(N):
        baseFunc_GF.append(GridFunction(fes))
        dzbaseFunc_GF.append(GridFunction(fesL2))
        baseFunc_GF[i].vec[:] = 0
        baseFunc_GF[i].vec[i] = 1

        dzbaseFunc_GF[i].Set(grad(baseFunc_GF[i]))

        baseFunc.append(lambda z, loc_GF=baseFunc_GF[i] : loc_GF(mesh1D(z)))
        dzbaseFunc.append(lambda z, loc_dzGF=dzbaseFunc_GF[i]: loc_dzGF(mesh1D(z)))
        
    return mesh1D,  baseFunc_GF, dzbaseFunc_GF, baseFunc, dzbaseFunc




def parsePhiToCode(phi):
    ret_code = ""
    if isinstance(phi, pyLobatto):
        ret_code += "phi"
    elif isinstance(phi, pydxLobatto):
        ret_code += "dzphi"
    elif isinstance(phi, str):
        return phi
    # complete with other variants 
    elif "symCF" in globals().keys() and isinstance(phi, symCF):
        return phi.__str__()[1:-1]
    else:
        
        from cempy.phiFunctions import Lobatto, dxLobatto
        if isinstance(phi, Lobatto):
            ret_code += "phi"
        elif isinstance(phi, dxLobatto):
            ret_code += "dzphi"

        else: 
            return False # -> numerical integration neccessary

    



    ret_code += str(phi.order)
    if phi.inIron == True and phi.inAir == False:
        ret_code += "_Fe"
    if phi.inIron == False and phi.inAir == True:
        ret_code += "_air"

    
    return ret_code

def parseCodeToPhi(code, dFe, d0, numSheets):
    if isinstance(code, str):
        
        phi_order = int(code.split("_")[0].split("phi")[1])
        dz = code.split("_")[0].split("phi")[0] == "dz"

        inAir = True
        inIron = True

        if code.split("_")[-1] == "air": inIron=False
        if code.split("_")[-1] == "Fe": inAir=False


        fun = Lobatto if not dz else dxLobatto


        zstart = -numSheets*(dFe+d0)/2
        phi = fun(zstart , dFe+d0, phi_order, numSheets, dFe/(dFe+d0), 2, inIron=inIron, inAir=inAir)    
    
    
    return phi

def num_int(a, b,  fun1, fun2=lambda x: 1, int_order=3, int_order_max=7):
    ret_old = 0
    ret = 0
    int_order0 = int_order
    while (int_order <= int_order0+1 or (ret_old != 0 and abs((ret - ret_old))/abs(ret_old)> 1e-5)) and int_order <= int_order_max:
        ret_old  = ret
        xi, wi = np.polynomial.legendre.leggauss(int_order)
        ret = (b-a)/2 * sum([w * fun1((b-a)/2*x+(a+b)/2) * fun2((b-a)/2*x+(a+b)/2) for x, w in zip(xi, wi)])  
        
        
        # print("integration order",int_order, "error", abs((ret - ret_old))/abs(ret_old))
        int_order += 1

    return ret   

def __phiXphi__(phi1, phi2, lambdaFe, lambda0, dFe=None, d0=None, force_num_int=False, **kwargs) -> float:
    # valid names: phi0, phi1, phi2, dzphi1, dzphi2
    # names = ["phi0", "dzphi0", 
    #     "phi_I_1", "dzphi_I_1","phi_I_1_0", "dzphi_I_1_0","phi_I_1_Fe", "dzphi_I_1_Fe", 
    #     "phi_I_2", "dzphi_I_2","phi_I_2_0", "dzphi_I_2_0","phi_I_2_Fe", "dzphi_I_2_Fe",
    #     "phi_I_3", "dzphi_I_3","phi_I_3_0", "dzphi_I_3_0","phi_I_3_Fe", "dzphi_I_3_Fe",
    #     "phi_I_4", "dzphi_I_4","phi_I_4_0", "dzphi_I_4_0","phi_I_4_Fe", "dzphi_I_4_Fe",
        
    #     "phi0b0", "phi1b0", "phi1b1", "phi2b0", "phi2b1", "phi2b2", 
    #     "dzphi0b0","dzphi1b0", "dzphi1b1", "dzphi2b0", "dzphi2b1", "dzphi2b2", 
    #     "phi2b1_air", "dzphi2b1_air", "phi1b1_Fe", "dzphi1b1_Fe", "phi1b0_Fe", "dzphi1b0_Fe", 
    #     "phi2b0_air", "dzphi2b0_air","phi2b2_air", "dzphi2b2_air", 
    #     "phi2b1_Fe", "dzphi2b1_Fe", "phi2b0_Fe", "dzphi2b0_Fe","phi2b2_Fe", "dzphi2b2_Fe",
    #     "phi1b1_air", "dzphi1b1_air", "phi1b0_air", "dzphi1b0_air"]
    # if name1 not in names or name2 not in names:
    #     raise ValueError("invalid name of phi " + name1 + " or " + name2)

    for kw in kwargs.keys():
        kwords = ["checkDimensions", "modelHalfAir", "force_full_Phi", "numSheets"]
        if kw not in kwords:
            print("undocumented keyword\t", kw)
            print("valid keywords are:", kwords)
            input()
        

    try:
        from symCF import symCF
        if isinstance(phi1, symCF) and isinstance(phi2, symCF):
            return phi1 * phi2
    except:
        pass

    checkDimensions = kwargs["checkDimensions"] if "checkDimensions" in kwargs.keys() else True
    force_full_Phi = kwargs["force_full_Phi"] if "force_full_Phi" in kwargs.keys() else False

   

    if "numSheets" in kwargs.keys():
        numSheets = kwargs["numSheets"]        
    elif hasattr(phi1, "numSheets"):
        numSheets = phi1.numSheets
    elif hasattr(phi2, "numSheets"):
        numSheets = phi2.numSheets
    else:
        numSheets = cl_Phi.numSheets

    if "modelHalfAir" in kwargs.keys():
        modelHalfAir = kwargs["modelHalfAir"]        
    elif hasattr(phi1, "modelHalfAir"):
        modelHalfAir = phi1.modelHalfAir
    elif hasattr(phi2, "modelHalfAir"):
        modelHalfAir = phi2.modelHalfAir
    else:
        modelHalfAir = cl_Phi.modelHalfAir




    numSheetsMinus1 = numSheets if modelHalfAir else (numSheets - 1) 

    # get the dFe from the phis
    if dFe == None:
        if hasattr(phi1, "dFe") and hasattr(phi1, "d0"):
            dFe = phi1.dFe
            d0 = phi1.d0
        elif hasattr(phi2, "dFe") and hasattr(phi2, "d0"):
            dFe = phi2.dFe
            d0 = phi2.d0
        else:
            dFe = cl_Phi.dFe
            d0 = cl_Phi.d0



    # if not isinstance(phi1, str) and phi1.fullPhi:
    #     force_num_int = True
    # if not isinstance(phi2, str) and phi2.fullPhi:
    #     force_num_int = True

    # if second argument is a microshape function
    code1 = parsePhiToCode(phi1)
    code2 = parsePhiToCode(phi2)

    if force_num_int and isinstance(phi1, str):
        phi1 = parseCodeToPhi(phi1, dFe, d0, numSheets)
    if force_num_int and isinstance(phi2, str):
        phi2 = parseCodeToPhi(phi2, dFe, d0, numSheets)

    
    # if True:
    if code2 == False or code1 == False or force_num_int :
        # print("numerical homogenisation: ", phi1.name, " x ", phi2.name)

        if checkDimensions and (abs(phi1.dFe- dFe) > 1e-8 or abs(phi2.dFe- dFe) > 1e-8 or abs(phi1.d0- d0) > 1e-8 or abs(phi2.d0- d0) > 1e-8):
            print("??????????????????????????")
            print("phi1.dFe", phi1.dFe, "dFe", dFe)
            print("phi2.dFe", phi2.dFe, "dFe", dFe)
            print("phi1.d0", phi1.d0, "d0", d0)
            print("phi2.d0", phi2.d0, "d0", d0)
            raise ValueError("dimensions of parameters dont match")
        
        ret = 0

        

        # integrate over a single sheet and multipy
        if phi1.fullPhi == False and phi2.fullPhi == False and force_full_Phi==False and modelHalfAir:
            # integrate iron
            if phi1.inIron and phi2.inIron:
                ret += numSheets * lambdaFe * num_int(phi1.xStart + phi1.d0/2, phi1.xStart + phi1.d0/2 + phi1.dFe, phi1.DirectEvaluate, phi2.DirectEvaluate, 4, 30)
                
            # integrate air
            if phi1.inAir and phi2.inAir:
                ret += numSheetsMinus1 * lambda0 * num_int(phi1.xStart, phi1.xStart + phi1.d0/2, phi1.DirectEvaluate, phi2.DirectEvaluate, 4, 30)
                ret += numSheetsMinus1 * lambda0 * num_int(phi1.xStart + phi1.d - phi1.d0/2, phi1.xStart + phi1.d, phi1.DirectEvaluate, phi2.DirectEvaluate, 4, 30)
        
            
            denom = (numSheets * dFe + numSheetsMinus1 * d0)

        # integrate over full stack (inperiodic phi function)
        else:
            # no overlapping section
            # if phi1.inInner != phi2.inInner and phi1.inOuter != phi2.inOuter:
            #     return 0

            denom = 0

            sameIntervalls = (phi1.dFe == phi2.dFe and phi1.d0 == phi2.d0 and phi1.numSheets == phi2.numSheets and phi1.xStart == phi2.xStart)

            # matching intervals in phis
            if sameIntervalls:
                xstart = phi1.xStart
                for n in range(numSheets):
                    
                    # only on used range
                    # if (n == 0 or n == numSheets-1) and (not phi1.inOuter or not phi2.inOuter):
                    #     xstart += phi1.d
                    #     continue
                    # if (n > 0 and n < numSheets-1) and (not phi1.inInner or not phi2.inInner):
                    #     xstart += phi1.d
                    #     continue
                    
                    if phi1.inIron and phi2.inIron:
                        ret += lambdaFe * num_int(xstart + phi1.d0/2, xstart + phi1.d0/2 + phi1.dFe, phi1.DirectEvaluate, phi2.DirectEvaluate)
                        
                    # if air is modelled
                    
                    # integrate air
                    if phi1.inAir and phi2.inAir:
                        # if half air is modelled
                        if modelHalfAir or n > 0:
                            ret += lambda0 * num_int(xstart, xstart + phi1.d0/2, phi1.DirectEvaluate, phi2.DirectEvaluate)

                        if modelHalfAir or n < numSheets - 1:
                            ret += lambda0 * num_int(xstart + phi1.d - phi1.d0/2, xstart + phi1.d, phi1.DirectEvaluate, phi2.DirectEvaluate)

                    xstart += dFe + d0
                denom = numSheets * dFe + d0 * numSheetsMinus1 
                             
                
            else:
                # get overlapping area
                xstart = max(phi1.xStart, phi2.xStart)
                xend = min(phi1.xEnd, phi2.xEnd)
                
                if xstart >= xend:
                    return 0
                # get integration intervalls (continuous functions on each interval)
                xi_phi_1 = [phi1.xStart, phi1.xStart + phi1.d0/2]
                
                for i in range(phi1.numSheets):
                    xi_phi_1.append(xi_phi_1[-1] + phi1.dFe)
                    xi_phi_1.append(xi_phi_1[-1] + phi1.d0)
                xi_phi_1[-1] = phi2.xEnd
                
                xi_phi_2 = [phi2.xStart, phi2.xStart + phi2.d0/2]
                for i in range(phi2.numSheets):
                    xi_phi_2.append(xi_phi_2[-1] + phi2.dFe)
                    xi_phi_2.append(xi_phi_2[-1] + phi2.d0)
                xi_phi_2[-1] = phi2.xEnd
                
                # remove double entries and sort them in ascending order
                xi = np.array(list(set(xi_phi_1 + xi_phi_2)), dtype=float)
                xi.sort()
                # crop to overlapping section
                xi = xi[np.logical_and(xi > xstart, xi < xend)]
                # add begin and end
                xi = np.hstack([xstart, xi, xend])
                
                N = len(xi) - 1
                
                # integrate over elements
                ret = 0
                for n in range(N):
                    
                    xleft = xi[n]
                    xright = xi[n+1]

                    # phi1 is the trial function
                    lam = lambdaFe if phi1.isIron((xleft+xright)/2) else lambda0
                    ret += lam * num_int(xleft, xright, phi1.DirectEvaluate, phi2.DirectEvaluate)
                        
                
                denom = max(phi1.xEnd, phi2.xEnd) -min(phi1.xStart, phi2.xStart) 
        
        ret /= denom
        # print("numerical homogenisation: ", phi1.name, " x ", phi2.name, "=", ret)
        
        return ret

    
    # replace names to derived functions for integral types
    # if name1.count("I") == 1 and name1.count("dz") == 1:
    #     _ = name1.split("_")
    #     if int(_[2]) > 1:
    #         _[2] = str(int(_[2]) - 1)
    #         _[0] = _[0][2:]
    #         name1 = _[0]+"_"+_[1]+"_"+_[2]+"_"+_[3] if len(_) == 4 else _[0]+"_"+_[1]+"_"+_[2]
    # if name2.count("I") == 1 and name2.count("dz") == 1:
    #     _ = name2.split("_")
    #     if int(_[2]) > 1:
    #         _[2] = str(int(_[2]) - 1)
    #         _[0] = _[0][2:]
    #         name2 = _[0]+"_"+_[1]+"_"+_[2]+"_"+_[3] if len(_) == 4 else _[0]+"_"+_[1]+"_"+_[2]         
    #print("\t ", name1," ",  name2)
    


    # alphabetic order to reduce cases (no commutative law is needed)
    if code1 > code2:
        _ = code1
        code1=code2
        code2=_

    d = dFe + d0

    if code2[:6] == "dzphi_I_0" or code1[:6] == "dzphi_I_0":
        return 0

    # non overlapping intervall
    if code1.split("_")[-1] == "air" and code2.split("_")[-1] == "Fe":
        return 0
    if code1.split("_")[-1] == "Fe" and code2.split("_")[-1] == "air":
        return 0
    # print("-> ", code1," ",  code2)

    # reduce to overlapping interval
    if code1.split("_")[-1] == "air" and code2.split("_")[-1] != "air":
        code2 += "_air"
    if code1.split("_")[-1] != "air" and code2.split("_")[-1] == "air":
        code1 += "_air"
    if code1.split("_")[-1] == "Fe" and code2.split("_")[-1] != "Fe":
        code2 += "_Fe"
    if code1.split("_")[-1] != "Fe" and code2.split("_")[-1] == "Fe":
        code1 += "_Fe"

    if code1 == "phi0_Fe" and code2 == "phi0_Fe":
            return (dFe*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi0_air" and code2 == "phi0_air":
            return (d0*lambda0*numSheetsMinus1)/(d0*numSheetsMinus1+dFe*numSheets)    
    if code1 == "phi0" and code2 == "phi3":
            return 0
    if code1 == "dzphi3" and code2 == "phi0":
            return 0
    if code1 == "phi1" and code2 == "phi3":
            return (sqrt(2)*d0*lambda0*numSheetsMinus1-sqrt(2)*dFe*lambdaFe*numSheets)/(6*sqrt(5)*d0*numSheetsMinus1+6*sqrt(5)*dFe*numSheets)
    if code1 == "dzphi3" and code2 == "phi1":
            return 0
    if code1 == "dzphi1" and code2 == "phi3":
            return 0
    if code1 == "dzphi1" and code2 == "dzphi3":
            return 0
    if code1 == "phi3" and code2 == "phi3":
            return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(21*d0*numSheetsMinus1+21*dFe*numSheets)
    if code1 == "dzphi3" and code2 == "phi3":
            return 0
    if code1 == "phi2_air" and code2 == "phi3_air":
            return 0
    if code1 == "dzphi2_air" and code2 == "phi3_air":
            return -(sqrt(3)*sqrt(5)*lambda0*numSheetsMinus1)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi3" and code2 == "dzphi3":
            return (2*dFe*lambda0*numSheetsMinus1+2*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)
    if code1 == "dzphi3_air" and code2 == "phi2_air":
            return (sqrt(3)*sqrt(5)*lambda0*numSheetsMinus1)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi2_air" and code2 == "dzphi3_air":
            return 0
    if code1 == "phi0" and code2 == "phi0":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi0" and code2 == "phi1":
        return 0
    if code1 == "dzphi1" and code2 == "phi0":
        return -(2*lambda0*numSheetsMinus1-2*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi0_Fe" and code2 == "phi2_Fe":
        return -(dFe*lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi2_Fe" and code2 == "phi0_Fe":
        return 0
    if code1 == "phi0_Fe" and code2 == "phi3_Fe":
        return 0
    if code1 == "dzphi3_Fe" and code2 == "phi0_Fe":
        return 0
    if code1 == "phi0_air" and code2 == "phi2_air":
        return -(sqrt(3)*d0*lambda0*numSheetsMinus1)/(3*sqrt(2)*d0*numSheetsMinus1+3*sqrt(2)*dFe*numSheets)
    if code1 == "dzphi2_air" and code2 == "phi0_air":
        return 0
    if code1 == "phi1" and code2 == "phi1":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1" and code2 == "phi1":
        return 0
    if code1 == "phi0_Fe" and code2 == "phi1_Fe":
        return 0
    if code1 == "dzphi1_Fe" and code2 == "phi0_Fe":
        return (2*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi1_Fe" and code2 == "phi1_Fe":
        return (dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1_Fe" and code2 == "phi1_Fe":
        return 0
    if code1 == "dzphi1_Fe" and code2 == "dzphi1_Fe":
        return (4*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)
    if code1 == "phi1_Fe" and code2 == "phi2_Fe":
        return 0
    if code1 == "dzphi2_Fe" and code2 == "phi1_Fe":
        return (sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*numSheetsMinus1+sqrt(3)*dFe*numSheets)
    if code1 == "phi1_Fe" and code2 == "phi3_Fe":
        return -(dFe*lambdaFe*numSheets)/(3*sqrt(2)*sqrt(5)*d0*numSheetsMinus1+3*sqrt(2)*sqrt(5)*dFe*numSheets)
    if code1 == "dzphi3_Fe" and code2 == "phi1_Fe":
        return 0
    if code1 == "phi1_air" and code2 == "phi2_air":
        return 0
    if code1 == "dzphi2_air" and code2 == "phi1_air":
        return -(sqrt(2)*sqrt(3)*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1" and code2 == "dzphi1":
        return (4*dFe*lambda0*numSheetsMinus1+4*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)
    if code1 == "dzphi1_Fe" and code2 == "phi2_Fe":
        return -(sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*numSheetsMinus1+sqrt(3)*dFe*numSheets)
    if code1 == "dzphi1_Fe" and code2 == "dzphi2_Fe":
        return 0
    if code1 == "dzphi1_Fe" and code2 == "phi3_Fe":
        return 0
    if code1 == "dzphi1_Fe" and code2 == "dzphi3_Fe":
        return 0
    if code1 == "dzphi1_air" and code2 == "phi2_air":
        return (2*sqrt(3)*lambda0*numSheetsMinus1)/(3*sqrt(2)*d0*numSheetsMinus1+3*sqrt(2)*dFe*numSheets)
    if code1 == "dzphi1_air" and code2 == "dzphi2_air":
        return 0
    if code1 == "phi0_air" and code2 == "phi1_air":
        return 0
    if code1 == "dzphi1_air" and code2 == "phi0_air":
        return -(2*lambda0*numSheetsMinus1)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi1_air" and code2 == "phi1_air":
        return (d0*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1_air" and code2 == "phi1_air":
        return 0
    if code1 == "dzphi1_air" and code2 == "dzphi1_air":
        return (4*lambda0*numSheetsMinus1)/(d0**2*numSheetsMinus1+d0*dFe*numSheets)

    if code1 == "phi2_Fe" and code2 == "phi2_Fe":
        return (dFe*lambdaFe*numSheets)/(5*d0*numSheetsMinus1+5*dFe*numSheets)
    if code1 == "dzphi2_Fe" and code2 == "phi2_Fe":
        return 0
    if code1 == "phi2_Fe" and code2 == "phi3_Fe":
        return 0
    if code1 == "dzphi3_Fe" and code2 == "phi2_Fe":
        return (lambdaFe*numSheets)/(sqrt(3)*sqrt(5)*d0*numSheetsMinus1+sqrt(3)*sqrt(5)*dFe*numSheets)
    if code1 == "dzphi2_Fe" and code2 == "dzphi2_Fe":
        return (2*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)
    if code1 == "dzphi2_Fe" and code2 == "phi3_Fe":
        return -(lambdaFe*numSheets)/(sqrt(3)*sqrt(5)*d0*numSheetsMinus1+sqrt(3)*sqrt(5)*dFe*numSheets)
    if code1 == "dzphi2_Fe" and code2 == "dzphi3_Fe":
        return 0
    if code1 == "phi3_Fe" and code2 == "phi3_Fe":
        return (dFe*lambdaFe*numSheets)/(21*d0*numSheetsMinus1+21*dFe*numSheets)
    if code1 == "dzphi3_Fe" and code2 == "phi3_Fe":
        return 0
    if code1 == "dzphi3_Fe" and code2 == "dzphi3_Fe":
        return (2*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)
    if code1 == "phi2_air" and code2 == "phi2_air":
        return (d0*lambda0*numSheetsMinus1)/(5*d0*numSheetsMinus1+5*dFe*numSheets)
    if code1 == "dzphi2_air" and code2 == "phi2_air":
        return 0
    if code1 == "dzphi2_air" and code2 == "dzphi2_air":
        return (2*lambda0*numSheetsMinus1)/(d0**2*numSheetsMinus1+d0*dFe*numSheets)


    if code1 == "phi0" and code2 == "phi21":
            return (3*d0*lambda0*numSheetsMinus1+2*dFe*lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi21" and code2 == "phi0":
            return -(lambda0*numSheetsMinus1-lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi0" and code2 == "phi22":
            return (3*d0*lambda0*numSheetsMinus1+2*dFe*lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi22" and code2 == "phi0":
            return (lambda0*numSheetsMinus1-lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi1" and code2 == "phi21":
            return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi21" and code2 == "phi1":
            return (lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "phi1" and code2 == "phi22":
            return -(d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi22" and code2 == "phi1":
            return (lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1" and code2 == "phi21":
            return -(3*lambda0*numSheetsMinus1-2*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1" and code2 == "dzphi21":
            return (2*dFe*lambda0*numSheetsMinus1+2*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)
    if code1 == "dzphi1" and code2 == "phi22":
            return -(3*lambda0*numSheetsMinus1-2*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1" and code2 == "dzphi22":
            return -(2*dFe*lambda0*numSheetsMinus1+2*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)
    if code1 == "phi21" and code2 == "phi21":
            return (5*d0*lambda0*numSheetsMinus1+3*dFe*lambdaFe*numSheets)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi21" and code2 == "phi21":
            return -(lambda0*numSheetsMinus1-lambdaFe*numSheets)/(2*d0*numSheetsMinus1+2*dFe*numSheets)
    if code1 == "phi21" and code2 == "phi22":
            return (5*d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(30*d0*numSheetsMinus1+30*dFe*numSheets)
    if code1 == "dzphi22" and code2 == "phi21":
            return (3*lambda0*numSheetsMinus1-lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi21" and code2 == "dzphi21":
            return (3*dFe*lambda0*numSheetsMinus1+4*d0*lambdaFe*numSheets)/(3*d0**2*dFe*numSheetsMinus1+3*d0*dFe**2*numSheets)
    if code1 == "dzphi21" and code2 == "phi22":
            return -(3*lambda0*numSheetsMinus1-lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi21" and code2 == "dzphi22":
            return -(3*dFe*lambda0*numSheetsMinus1+2*d0*lambdaFe*numSheets)/(3*d0**2*dFe*numSheetsMinus1+3*d0*dFe**2*numSheets)
    if code1 == "phi22" and code2 == "phi22":
            return (5*d0*lambda0*numSheetsMinus1+3*dFe*lambdaFe*numSheets)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi22" and code2 == "phi22":
            return (lambda0*numSheetsMinus1-lambdaFe*numSheets)/(2*d0*numSheetsMinus1+2*dFe*numSheets)
    if code1 == "dzphi22" and code2 == "dzphi22":
            return (3*dFe*lambda0*numSheetsMinus1+4*d0*lambdaFe*numSheets)/(3*d0**2*dFe*numSheetsMinus1+3*d0*dFe**2*numSheets) 

    if code1 == "phi0_Fe" and code2 == "phi22_Fe":
            return (dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi22_Fe" and code2 == "phi0_Fe":
            return -(lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi1_Fe" and code2 == "phi22_Fe":
            return -(dFe*lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi22_Fe" and code2 == "phi1_Fe":
            return (lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1_Fe" and code2 == "phi22_Fe":
            return (2*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1_Fe" and code2 == "dzphi22_Fe":
            return -(2*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)
    if code1 == "phi22_Fe" and code2 == "phi22_Fe":
            return (dFe*lambdaFe*numSheets)/(5*d0*numSheetsMinus1+5*dFe*numSheets)
    if code1 == "dzphi22_Fe" and code2 == "phi22_Fe":
            return -(lambdaFe*numSheets)/(2*d0*numSheetsMinus1+2*dFe*numSheets)
    if code1 == "dzphi22_Fe" and code2 == "dzphi22_Fe":
            return (4*lambdaFe*numSheets)/(3*d0*dFe*numSheetsMinus1+3*dFe**2*numSheets)


    if code1 == "phi0_Fe" and code2 == "phi21_Fe":
            return (dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi21_Fe" and code2 == "phi0_Fe":
            return (lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi1_Fe" and code2 == "phi21_Fe":
            return (dFe*lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi21_Fe" and code2 == "phi1_Fe":
            return (lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1_Fe" and code2 == "phi21_Fe":
            return (2*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1_Fe" and code2 == "dzphi21_Fe":
            return (2*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)
    if code1 == "phi21_Fe" and code2 == "phi21_Fe":
            return (dFe*lambdaFe*numSheets)/(5*d0*numSheetsMinus1+5*dFe*numSheets)
    if code1 == "dzphi21_Fe" and code2 == "phi21_Fe":
            return (lambdaFe*numSheets)/(2*d0*numSheetsMinus1+2*dFe*numSheets)
    if code1 == "dzphi21_Fe" and code2 == "dzphi21_Fe":
            return (4*lambdaFe*numSheets)/(3*d0*dFe*numSheetsMinus1+3*dFe**2*numSheets)

    if code1 == "phi0" and code2 == "phi2":
            return -(sqrt(2)*d0*lambda0*numSheetsMinus1+sqrt(2)*dFe*lambdaFe*numSheets)/(2*sqrt(3)*d0*numSheetsMinus1+2*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi2" and code2 == "phi0":
            return 0
    if code1 == "phi1" and code2 == "phi2":
            return 0
    if code1 == "dzphi2" and code2 == "phi1":
            return -(sqrt(2)*lambda0*numSheetsMinus1-sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*numSheetsMinus1+sqrt(3)*dFe*numSheets)
    if code1 == "dzphi1" and code2 == "phi2":
            return (2*lambda0*numSheetsMinus1-2*lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi1" and code2 == "dzphi2":
            return 0
    if code1 == "phi2" and code2 == "phi2":
            return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(5*d0*numSheetsMinus1+5*dFe*numSheets)
    if code1 == "dzphi2" and code2 == "phi2":
            return 0
    if code1 == "dzphi2" and code2 == "dzphi2":
            return (2*dFe*lambda0*numSheetsMinus1+2*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)

    if code1 == "phi2" and code2 == "phi21":
            return -(5*sqrt(2)*sqrt(3)*d0*lambda0*numSheetsMinus1+sqrt(2)*3**(3/2)*dFe*lambdaFe*numSheets)/(60*d0*numSheetsMinus1+60*dFe*numSheets)
    if code1 == "dzphi21" and code2 == "phi2":
            return (sqrt(2)*lambda0*numSheetsMinus1-sqrt(2)*lambdaFe*numSheets)/(2*sqrt(3)*d0*numSheetsMinus1+2*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi2" and code2 == "phi21":
            return -(lambda0*numSheetsMinus1-lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi2" and code2 == "dzphi21":
            return (sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*dFe*numSheetsMinus1+sqrt(3)*dFe**2*numSheets)

    if code1 == "phi2" and code2 == "phi22":
            return -(5*sqrt(2)*sqrt(3)*d0*lambda0*numSheetsMinus1+sqrt(2)*3**(3/2)*dFe*lambdaFe*numSheets)/(60*d0*numSheetsMinus1+60*dFe*numSheets)
    if code1 == "dzphi22" and code2 == "phi2":
            return -(sqrt(2)*lambda0*numSheetsMinus1-sqrt(2)*lambdaFe*numSheets)/(2*sqrt(3)*d0*numSheetsMinus1+2*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi2" and code2 == "phi22":
            return (lambda0*numSheetsMinus1-lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi2" and code2 == "dzphi22":
            return (sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*dFe*numSheetsMinus1+sqrt(3)*dFe**2*numSheets)

    if code1 == "phi2_Fe" and code2 == "phi21_Fe":
            return -(sqrt(3)*dFe*lambdaFe*numSheets)/(5*2**(3/2)*d0*numSheetsMinus1+5*2**(3/2)*dFe*numSheets)
    if code1 == "dzphi2_Fe" and code2 == "phi21_Fe":
            return (lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe*numSheets)


    if code1 == "phi21_Fe" and code2 == "phi2_Fe":
            return -(sqrt(3)*dFe*lambdaFe*numSheets)/(5*2**(3/2)*d0*numSheetsMinus1+5*2**(3/2)*dFe*numSheets)
    if code1 == "phi21_air" and code2 == "phi2_air":
            return -(sqrt(3)*d0*lambda0*numSheetsMinus1)/(3*2**(3/2)*d0*numSheetsMinus1+3*2**(3/2)*dFe*numSheets)
    if code1 == "dzphi2_air" and code2 == "phi21_air":
            return -(sqrt(2)*sqrt(3)*lambda0*numSheetsMinus1)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi21_Fe" and code2 == "phi2_Fe":
            return -(lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi21_Fe" and code2 == "dzphi2_Fe":
            return (sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*dFe*numSheetsMinus1+sqrt(3)*dFe**2*numSheets)
    if code1 == "dzphi21_air" and code2 == "phi2_air":
            return (sqrt(3)*lambda0*numSheetsMinus1)/(3*sqrt(2)*d0*numSheetsMinus1+3*sqrt(2)*dFe*numSheets)
    if code1 == "dzphi21_air" and code2 == "dzphi2_air":
            return 0


    if code1 == "phi0_air" and code2 == "phi21_air":
            return (d0*lambda0*numSheetsMinus1)/(2*d0*numSheetsMinus1+2*dFe*numSheets)
    if code1 == "dzphi21_air" and code2 == "phi0_air":
            return -(lambda0*numSheetsMinus1)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi1_air" and code2 == "phi21_air":
            return (d0*lambda0*numSheetsMinus1)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "dzphi21_air" and code2 == "phi1_air":
            return 0
    if code1 == "phi2_air" and code2 == "phi21_air":
            return -(sqrt(3)*d0*lambda0*numSheetsMinus1)/(3*2**(3/2)*d0*numSheetsMinus1+3*2**(3/2)*dFe*numSheets)
    if code1 == "phi21_air" and code2 == "phi21_air":
            return (d0*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi1_air" and code2 == "phi21_air":
            return -(lambda0*numSheetsMinus1)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "dzphi21_air" and code2 == "phi21_air":
            return -(lambda0*numSheetsMinus1)/(2*d0*numSheetsMinus1+2*dFe*numSheets)
    if code1 == "dzphi1_air" and code2 == "dzphi21_air":
            return (2*lambda0*numSheetsMinus1)/(d0**2*numSheetsMinus1+d0*dFe*numSheets)
    if code1 == "dzphi2_Fe" and code2 == "dzphi21_Fe":
            return (sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*dFe*numSheetsMinus1+sqrt(3)*dFe**2*numSheets)
    if code1 == "dzphi2_air" and code2 == "dzphi21_air":
            return 0
    if code1 == "dzphi21_air" and code2 == "dzphi21_air":
            return (lambda0*numSheetsMinus1)/(d0**2*numSheetsMinus1+d0*dFe*numSheets)

#----------------------------------------------------------------------

    if code1 == "phi_I_0" and code2 == "phi_I_0":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi_I_0" and code2 == "phi_I_1":
        return 0
    if code1 == "dzphi_I_1" and code2 == "phi_I_0":
        return -(2*lambda0*numSheetsMinus1-2*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi_I_0_Fe" and code2 == "phi_I_2_Fe":
        return -(dFe**2*lambdaFe*numSheets)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "phi_I_0_Fe" and code2 == "phi_I_1_Fe":
        return 0
    if code1 == "phi_I_0_Fe" and code2 == "phi_I_3_Fe":
        return 0
    if code1 == "phi_I_0_air" and code2 == "phi_I_2_air":
        return (d0**2*lambda0*numSheetsMinus1)/(6*d0*numSheetsMinus1+6*dFe*numSheets)
    if code1 == "phi_I_0_air" and code2 == "phi_I_1_air":
        return 0
    if code1 == "phi_I_0_air" and code2 == "phi_I_3_air":
        return 0
    if code1 == "phi_I_1" and code2 == "phi_I_1":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_I_1" and code2 == "phi_I_1":
        return 0
    if code1 == "phi_I_1_Fe" and code2 == "phi_I_2_Fe":
        return 0
    if code1 == "phi_I_1_Fe" and code2 == "phi_I_1_Fe":
        return (dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "phi_I_1_Fe" and code2 == "phi_I_3_Fe":
        return -(dFe**3*lambdaFe*numSheets)/(30*d0*numSheetsMinus1+30*dFe*numSheets)
    if code1 == "phi_I_1_air" and code2 == "phi_I_2_air":
        return 0
    if code1 == "phi_I_1_air" and code2 == "phi_I_1_air":
        return (d0*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "phi_I_1_air" and code2 == "phi_I_3_air":
        return -((5*d0*dFe**2-d0**3)*lambda0*numSheetsMinus1)/(120*d0*numSheetsMinus1+120*dFe*numSheets)
    if code1 == "dzphi_I_1" and code2 == "dzphi_I_1":
        return (4*dFe*lambda0*numSheetsMinus1+4*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)
    if code1 == "dzphi_I_1_Fe" and code2 == "phi_I_2_Fe":
        return -(dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_I_1_Fe" and code2 == "phi_I_1_Fe":
        return 0
    if code1 == "dzphi_I_1_Fe" and code2 == "phi_I_3_Fe":
        return 0
    if code1 == "dzphi_I_1_air" and code2 == "phi_I_2_air":
        return -(d0*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_I_1_air" and code2 == "phi_I_1_air":
        return 0
    if code1 == "dzphi_I_1_air" and code2 == "phi_I_3_air":
        return 0
    if code1 == "phi_I_2_Fe" and code2 == "phi_I_2_Fe":
        return (dFe**3*lambdaFe*numSheets)/(30*d0*numSheetsMinus1+30*dFe*numSheets)
    if code1 == "phi_I_2_Fe" and code2 == "phi_I_3_Fe":
        return 0
    if code1 == "phi_I_3_Fe" and code2 == "phi_I_3_Fe":
        return (17*dFe**5*lambdaFe*numSheets)/(5040*d0*numSheetsMinus1+5040*dFe*numSheets)
    if code1 == "phi_I_2_air" and code2 == "phi_I_2_air":
        return (d0**3*lambda0*numSheetsMinus1)/(30*d0*numSheetsMinus1+30*dFe*numSheets)
    if code1 == "phi_I_2_air" and code2 == "phi_I_3_air":
        return 0
    if code1 == "phi_I_3_air" and code2 == "phi_I_3_air":
        return ((140*d0*dFe**4-105*d0**3*dFe**2+33*d0**5)*lambda0*numSheetsMinus1)/(20160*d0*numSheetsMinus1+20160*dFe*numSheets)
#--------------------------------------------------------------------
    if code1 == "phi_B_0" and code2 == "phi_B_0":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi_B_0" and code2 == "phi_B_1":
        return 0
    if code1 == "dzphi_B_1" and code2 == "phi_B_0":
        return -(2*lambda0*numSheetsMinus1-2*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi_B_0" and code2 == "phi_B_3":
        return 0
    if code1 == "dzphi_B_3" and code2 == "phi_B_0":
        return 0
    if code1 == "phi_B_0_air" and code2 == "phi_B_2_air":
        return -(2*d0*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_B_2_air" and code2 == "phi_B_0_air":
        return 0
    if code1 == "phi_B_1" and code2 == "phi_B_1":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_B_1" and code2 == "phi_B_1":
        return 0
    if code1 == "phi_B_1" and code2 == "phi_B_3":
        return -(2*d0*lambda0*numSheetsMinus1+2*dFe*lambdaFe*numSheets)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi_B_3" and code2 == "phi_B_1":
        return 0
    if code1 == "phi_B_1_air" and code2 == "phi_B_2_air":
        return 0
    if code1 == "dzphi_B_2_air" and code2 == "phi_B_1_air":
        return -(4*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_B_1" and code2 == "dzphi_B_1":
        return (4*dFe*lambda0*numSheetsMinus1+4*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)
    if code1 == "dzphi_B_1" and code2 == "phi_B_3":
        return 0
    if code1 == "dzphi_B_1" and code2 == "dzphi_B_3":
        return 0
    if code1 == "dzphi_B_1_air" and code2 == "phi_B_2_air":
        return (4*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_B_1_air" and code2 == "dzphi_B_2_air":
        return 0
    if code1 == "phi_B_3" and code2 == "phi_B_3":
        return (8*d0*lambda0*numSheetsMinus1+8*dFe*lambdaFe*numSheets)/(105*d0*numSheetsMinus1+105*dFe*numSheets)
    if code1 == "dzphi_B_3" and code2 == "phi_B_3":
        return 0
    if code1 == "phi_B_2_air" and code2 == "phi_B_3_air":
        return 0
    if code1 == "dzphi_B_2_air" and code2 == "phi_B_3_air":
        return (8*lambda0*numSheetsMinus1)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi_B_3" and code2 == "dzphi_B_3":
        return (16*dFe*lambda0*numSheetsMinus1+16*d0*lambdaFe*numSheets)/(5*d0**2*dFe*numSheetsMinus1+5*d0*dFe**2*numSheets)
    if code1 == "dzphi_B_3_air" and code2 == "phi_B_2_air":
        return -(8*lambda0*numSheetsMinus1)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi_B_2_air" and code2 == "dzphi_B_3_air":
        return 0
    if code1 == "phi_B_2_air" and code2 == "phi_B_2_air":
        return (8*d0*lambda0*numSheetsMinus1)/(15*d0*numSheetsMinus1+15*dFe*numSheets)
    if code1 == "dzphi_B_2_air" and code2 == "phi_B_2_air":
        return 0
    if code1 == "dzphi_B_2_air" and code2 == "dzphi_B_2_air":
        return (16*lambda0*numSheetsMinus1)/(3*d0**2*numSheetsMinus1+3*d0*dFe*numSheets)
#--------------------------------------------------------------------

    if code1 == "phi_C_0" and code2 == "phi_C_0":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)#, (lambda0*lambdaFe*(numSheetsMinus1*d0+numSheets*dFe))/(lambdaFe*numSheetsMinus1*d0 + lambda0*numSheets*dFe)
    if code1 == "phi_C_0" and code2 == "phi_C_1":
        return 0
    if code1 == "dzphi_C_1" and code2 == "phi_C_0":
        return -(2*lambda0*numSheetsMinus1-2*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi_C_0_air" and code2 == "phi_C_2_air":
        return (sqrt(3)*d0**2*lambda0*numSheetsMinus1)/(3*sqrt(2)*d0*dFe*numSheetsMinus1+3*sqrt(2)*dFe**2*numSheets)
    if code1 == "dzphi_C_2_air" and code2 == "phi_C_0_air":
        return 0
    if code1 == "phi_C_1" and code2 == "phi_C_1":
        return (d0*lambda0*numSheetsMinus1+dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_C_1" and code2 == "phi_C_1":
        return 0
    if code1 == "phi_C_1_air" and code2 == "phi_C_2_air":
        return 0
    if code1 == "dzphi_C_2_air" and code2 == "phi_C_1_air":
        return (2*sqrt(3)*d0*lambda0*numSheetsMinus1)/(3*sqrt(2)*d0*dFe*numSheetsMinus1+3*sqrt(2)*dFe**2*numSheets)
    if code1 == "dzphi_C_1" and code2 == "dzphi_C_1":
        return (4*dFe*lambda0*numSheetsMinus1+4*d0*lambdaFe*numSheets)/(d0**2*dFe*numSheetsMinus1+d0*dFe**2*numSheets)
    if code1 == "dzphi_C_1_air" and code2 == "phi_C_2_air":
        return -(sqrt(2)*sqrt(3)*d0*lambda0*numSheetsMinus1)/(3*d0*dFe*numSheetsMinus1+3*dFe**2*numSheets)
    if code1 == "dzphi_C_1_air" and code2 == "dzphi_C_2_air":
        return 0
    if code1 == "phi_C_2_air" and code2 == "phi_C_2_air":
        return (d0**3*lambda0*numSheetsMinus1)/(5*d0*dFe**2*numSheetsMinus1+5*dFe**3*numSheets)
    if code1 == "dzphi_C_2_air" and code2 == "phi_C_2_air":
        return 0
    if code1 == "dzphi_C_2_air" and code2 == "dzphi_C_2_air":
        return (2*d0*lambda0*numSheetsMinus1)/(d0*dFe**2*numSheetsMinus1+dFe**3*numSheets)

    if code1 == "phi_C_0_Fe" and code2 == "phi_C_2_Fe":
        return -(dFe*lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe*numSheets)
    if code1 == "dzphi_C_2_Fe" and code2 == "phi_C_0_Fe":
        return 0
    if code1 == "phi_C_1_Fe" and code2 == "phi_C_2_Fe":
        return 0
    if code1 == "dzphi_C_2_Fe" and code2 == "phi_C_1_Fe":
        return (sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*numSheetsMinus1+sqrt(3)*dFe*numSheets)
    if code1 == "dzphi_C_1_Fe" and code2 == "phi_C_2_Fe":
        return -(sqrt(2)*lambdaFe*numSheets)/(sqrt(3)*d0*numSheetsMinus1+sqrt(3)*dFe*numSheets)
    if code1 == "dzphi_C_1_Fe" and code2 == "dzphi_C_2_Fe":
        return 0
    if code1 == "phi_C_2_Fe" and code2 == "phi_C_2_Fe":
        return (dFe*lambdaFe*numSheets)/(5*d0*numSheetsMinus1+5*dFe*numSheets)
    if code1 == "dzphi_C_2_Fe" and code2 == "phi_C_2_Fe":
        return 0
    if code1 == "dzphi_C_2_Fe" and code2 == "dzphi_C_2_Fe":
        return (2*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)

    if code1 == "phi_C_0" and code2 == "phi_C_2":
        return (sqrt(2)*d0**2*lambda0*numSheetsMinus1-sqrt(2)*dFe**2*lambdaFe*numSheets)/(2*sqrt(3)*d0*dFe*numSheetsMinus1+2*sqrt(3)*dFe**2*numSheets)
    if code1 == "dzphi_C_2" and code2 == "phi_C_0":
        return 0
    if code1 == "phi_C_1" and code2 == "phi_C_2":
        return 0
    if code1 == "dzphi_C_2" and code2 == "phi_C_1":
        return (2*d0*lambda0*numSheetsMinus1+2*dFe*lambdaFe*numSheets)/(sqrt(2)*sqrt(3)*d0*dFe*numSheetsMinus1+sqrt(2)*sqrt(3)*dFe**2*numSheets)
    if code1 == "dzphi_C_1" and code2 == "phi_C_2":
        return -(sqrt(2)*d0*lambda0*numSheetsMinus1+sqrt(2)*dFe*lambdaFe*numSheets)/(sqrt(3)*d0*dFe*numSheetsMinus1+sqrt(3)*dFe**2*numSheets)
    if code1 == "dzphi_C_1" and code2 == "dzphi_C_2":
        return 0
    if code1 == "phi_C_2" and code2 == "phi_C_2":
        return (d0**3*lambda0*numSheetsMinus1+dFe**3*lambdaFe*numSheets)/(5*d0*dFe**2*numSheetsMinus1+5*dFe**3*numSheets)
    if code1 == "dzphi_C_2" and code2 == "phi_C_2":
        return 0
    if code1 == "dzphi_C_2" and code2 == "dzphi_C_2":
        return (2*d0*lambda0*numSheetsMinus1+2*dFe*lambdaFe*numSheets)/(d0*dFe**2*numSheetsMinus1+dFe**3*numSheets)


    if code1 == "phi_C_0" and code2 == "phi_C_3":
        return 0
    if code1 == "dzphi_C_3" and code2 == "phi_C_0":
        return 0
    if code1 == "phi_C_0_Fe" and code2 == "phi_C_1_Fe":
        return 0
    if code1 == "dzphi_C_1_Fe" and code2 == "phi_C_0_Fe":
        return (2*lambdaFe*numSheets)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi_C_0_Fe" and code2 == "phi_C_3_Fe":
        return 0
    if code1 == "dzphi_C_3_Fe" and code2 == "phi_C_0_Fe":
        return 0
    if code1 == "phi_C_0_air" and code2 == "phi_C_1_air":
        return 0
    if code1 == "dzphi_C_1_air" and code2 == "phi_C_0_air":
        return -(2*lambda0*numSheetsMinus1)/(d0*numSheetsMinus1+dFe*numSheets)
    if code1 == "phi_C_0_air" and code2 == "phi_C_3_air":
        return 0
    if code1 == "dzphi_C_3_air" and code2 == "phi_C_0_air":
        return 0
    if code1 == "phi_C_1" and code2 == "phi_C_3":
        return (sqrt(2)*d0**2*lambda0*numSheetsMinus1-sqrt(2)*dFe**2*lambdaFe*numSheets)/(6*sqrt(5)*d0*dFe*numSheetsMinus1+6*sqrt(5)*dFe**2*numSheets)
    if code1 == "dzphi_C_3" and code2 == "phi_C_1":
        return 0
    if code1 == "phi_C_1_Fe" and code2 == "phi_C_1_Fe":
        return (dFe*lambdaFe*numSheets)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_C_1_Fe" and code2 == "phi_C_1_Fe":
        return 0
    if code1 == "phi_C_1_Fe" and code2 == "phi_C_3_Fe":
        return -(dFe*lambdaFe*numSheets)/(3*sqrt(2)*sqrt(5)*d0*numSheetsMinus1+3*sqrt(2)*sqrt(5)*dFe*numSheets)
    if code1 == "dzphi_C_3_Fe" and code2 == "phi_C_1_Fe":
        return 0
    if code1 == "phi_C_1_air" and code2 == "phi_C_1_air":
        return (d0*lambda0*numSheetsMinus1)/(3*d0*numSheetsMinus1+3*dFe*numSheets)
    if code1 == "dzphi_C_1_air" and code2 == "phi_C_1_air":
        return 0
    if code1 == "phi_C_1_air" and code2 == "phi_C_3_air":
        return (sqrt(5)*d0**2*lambda0*numSheetsMinus1)/(15*sqrt(2)*d0*dFe*numSheetsMinus1+15*sqrt(2)*dFe**2*numSheets)
    if code1 == "dzphi_C_3_air" and code2 == "phi_C_1_air":
        return 0
    if code1 == "dzphi_C_1" and code2 == "phi_C_3":
        return 0
    if code1 == "dzphi_C_1" and code2 == "dzphi_C_3":
        return 0
    if code1 == "dzphi_C_1_Fe" and code2 == "dzphi_C_1_Fe":
        return (4*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)
    if code1 == "dzphi_C_1_Fe" and code2 == "phi_C_3_Fe":
        return 0
    if code1 == "dzphi_C_1_Fe" and code2 == "dzphi_C_3_Fe":
        return 0
    if code1 == "dzphi_C_1_air" and code2 == "dzphi_C_1_air":
        return (4*lambda0*numSheetsMinus1)/(d0**2*numSheetsMinus1+d0*dFe*numSheets)
    if code1 == "dzphi_C_1_air" and code2 == "phi_C_3_air":
        return 0
    if code1 == "dzphi_C_1_air" and code2 == "dzphi_C_3_air":
        return 0
    if code1 == "phi_C_2" and code2 == "phi_C_3":
        return 0
    if code1 == "dzphi_C_3" and code2 == "phi_C_2":
        return -(d0**2*lambda0*numSheetsMinus1-dFe**2*lambdaFe*numSheets)/(sqrt(3)*sqrt(5)*d0*dFe**2*numSheetsMinus1+sqrt(3)*sqrt(5)*dFe**3*numSheets)
    if code1 == "phi_C_2_Fe" and code2 == "phi_C_3_Fe":
        return 0
    if code1 == "dzphi_C_3_Fe" and code2 == "phi_C_2_Fe":
        return (lambdaFe*numSheets)/(sqrt(3)*sqrt(5)*d0*numSheetsMinus1+sqrt(3)*sqrt(5)*dFe*numSheets)
    if code1 == "phi_C_2_air" and code2 == "phi_C_3_air":
        return 0
    if code1 == "dzphi_C_3_air" and code2 == "phi_C_2_air":
        return -(sqrt(3)*sqrt(5)*d0**2*lambda0*numSheetsMinus1)/(15*d0*dFe**2*numSheetsMinus1+15*dFe**3*numSheets)
    if code1 == "dzphi_C_2" and code2 == "phi_C_3":
        return (d0**2*lambda0*numSheetsMinus1-dFe**2*lambdaFe*numSheets)/(sqrt(3)*sqrt(5)*d0*dFe**2*numSheetsMinus1+sqrt(3)*sqrt(5)*dFe**3*numSheets)
    if code1 == "dzphi_C_2" and code2 == "dzphi_C_3":
        return 0
    if code1 == "dzphi_C_2_Fe" and code2 == "phi_C_3_Fe":
        return -(lambdaFe*numSheets)/(sqrt(3)*sqrt(5)*d0*numSheetsMinus1+sqrt(3)*sqrt(5)*dFe*numSheets)
    if code1 == "dzphi_C_2_Fe" and code2 == "dzphi_C_3_Fe":
        return 0
    if code1 == "dzphi_C_2_air" and code2 == "phi_C_3_air":
        return (sqrt(3)*sqrt(5)*d0**2*lambda0*numSheetsMinus1)/(15*d0*dFe**2*numSheetsMinus1+15*dFe**3*numSheets)
    if code1 == "dzphi_C_2_air" and code2 == "dzphi_C_3_air":
        return 0
    if code1 == "phi_C_3" and code2 == "phi_C_3":
        return (d0**3*lambda0*numSheetsMinus1+dFe**3*lambdaFe*numSheets)/(21*d0*dFe**2*numSheetsMinus1+21*dFe**3*numSheets)
    if code1 == "dzphi_C_3" and code2 == "phi_C_3":
        return 0
    if code1 == "phi_C_3_Fe" and code2 == "phi_C_3_Fe":
        return (dFe*lambdaFe*numSheets)/(21*d0*numSheetsMinus1+21*dFe*numSheets)
    if code1 == "dzphi_C_3_Fe" and code2 == "phi_C_3_Fe":
        return 0
    if code1 == "phi_C_3_air" and code2 == "phi_C_3_air":
        return (d0**3*lambda0*numSheetsMinus1)/(21*d0*dFe**2*numSheetsMinus1+21*dFe**3*numSheets)
    if code1 == "dzphi_C_3_air" and code2 == "phi_C_3_air":
        return 0
    if code1 == "dzphi_C_3" and code2 == "dzphi_C_3":
        return (2*d0*lambda0*numSheetsMinus1+2*dFe*lambdaFe*numSheets)/(d0*dFe**2*numSheetsMinus1+dFe**3*numSheets)
    if code1 == "dzphi_C_3_Fe" and code2 == "dzphi_C_3_Fe":
        return (2*lambdaFe*numSheets)/(d0*dFe*numSheetsMinus1+dFe**2*numSheets)
    if code1 == "dzphi_C_3_air" and code2 == "dzphi_C_3_air":
        return (2*d0*lambda0*numSheetsMinus1)/(d0*dFe**2*numSheetsMinus1+dFe**3*numSheets)


#----------------------------------------------------------------------
    if code1 == "phi0b0" and code2 == "phi0b0":
        return (dFe*lambdaFe+d0*lambda0)/d
    if code1 == "phi0b0_Fe" and code2 == "phi0b0_Fe":
        return (dFe*lambdaFe)/d
    if code1 == "phi0b0" and code2 == "phi2b1":
        return (2*dFe*lambdaFe+2*d0*lambda0)/(3*d)
    if code1 == "dzphi2b1" and code2 == "phi0b0":
        return 0
    if code1 == "phi0b0" and code2 == "phi2b0":
        return (dFe*lambdaFe+d0*lambda0)/(6*d)
    if code1 == "dzphi2b0" and code2 == "phi0b0":
        return -(lambdaFe+lambda0)/d
    if code1 == "phi0b0" and code2 == "phi2b2":
        return (dFe*lambdaFe+d0*lambda0)/(6*d)
    if code1 == "dzphi2b2" and code2 == "phi0b0":
        return (lambdaFe+lambda0)/d
    if code1 == "phi0b0" and code2 == "phi1b1":
        return (dFe*lambdaFe+d0*lambda0)/(2*d)
    if code1 == "dzphi1b1" and code2 == "phi0b0":
        return (lambdaFe+lambda0)/d
    if code1 == "phi0b0" and code2 == "phi1b0":
        return (dFe*lambdaFe+d0*lambda0)/(2*d)
    if code1 == "dzphi1b0" and code2 == "phi0b0":
        return -(lambdaFe+lambda0)/d
    if code1 == "phi0b0_Fe" and code2 == "phi2b1_Fe":
        return (2*dFe*lambdaFe)/(3*d)
    if code1 == "dzphi2b1_Fe" and code2 == "phi0b0_Fe":
        return 0
    if code1 == "phi0b0_Fe" and code2 == "phi2b0_Fe":
        return (dFe*lambdaFe)/(6*d)
    if code1 == "dzphi2b0_Fe" and code2 == "phi0b0_Fe":
        return -lambdaFe/d
    if code1 == "phi0b0_Fe" and code2 == "phi2b2_Fe":
        return (dFe*lambdaFe)/(6*d)
    if code1 == "dzphi2b2_Fe" and code2 == "phi0b0_Fe":
        return lambdaFe/d
    if code1 == "phi0b0_Fe" and code2 == "phi1b1_Fe":
        return (dFe*lambdaFe)/(2*d)
    if code1 == "dzphi1b1_Fe" and code2 == "phi0b0_Fe":
        return lambdaFe/d
    if code1 == "phi0b0_Fe" and code2 == "phi1b0_Fe":
        return (dFe*lambdaFe)/(2*d)
    if code1 == "dzphi1b0_Fe" and code2 == "phi0b0_Fe":
        return -lambdaFe/d
    if code1 == "phi0b0_air" and code2 == "phi2b1_air":
        return (2*d0*lambda0)/(3*d)
    if code1 == "dzphi2b1_air" and code2 == "phi0b0_air":
        return 0
    if code1 == "phi0b0_air" and code2 == "phi2b0_air":
        return (d0*lambda0)/(6*d)
    if code1 == "dzphi2b0_air" and code2 == "phi0b0_air":
        return -lambda0/d
    if code1 == "phi0b0_air" and code2 == "phi2b2_air":
        return (d0*lambda0)/(6*d)
    if code1 == "dzphi2b2_air" and code2 == "phi0b0_air":
        return lambda0/d
    if code1 == "phi0b0_air" and code2 == "phi1b1_air":
        return (d0*lambda0)/(2*d)
    if code1 == "dzphi1b1_air" and code2 == "phi0b0_air":
        return lambda0/d
    if code1 == "phi0b0_air" and code2 == "phi1b0_air":
        return (d0*lambda0)/(2*d)
    if code1 == "dzphi1b0_air" and code2 == "phi0b0_air":
        return -lambda0/d
    if code1 == "phi2b1" and code2 == "phi2b1":
        return (8*dFe*lambdaFe+8*d0*lambda0)/(15*d)
    if code1 == "dzphi2b1" and code2 == "phi2b1":
        return 0
    if code1 == "phi2b0" and code2 == "phi2b1":
        return (dFe*lambdaFe+d0*lambda0)/(15*d)
    if code1 == "dzphi2b0" and code2 == "phi2b1":
        return -(2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "phi2b1" and code2 == "phi2b2":
        return (dFe*lambdaFe+d0*lambda0)/(15*d)
    if code1 == "dzphi2b2" and code2 == "phi2b1":
        return (2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "phi1b1" and code2 == "phi2b1":
        return (dFe*lambdaFe+d0*lambda0)/(3*d)
    if code1 == "dzphi1b1" and code2 == "phi2b1":
        return (2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "phi1b0" and code2 == "phi2b1":
        return (dFe*lambdaFe+d0*lambda0)/(3*d)
    if code1 == "dzphi1b0" and code2 == "phi2b1":
        return -(2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "phi2b1_Fe" and code2 == "phi2b1_Fe":
        return (8*dFe*lambdaFe)/(15*d)
    if code1 == "dzphi2b1_Fe" and code2 == "phi2b1_Fe":
        return 0
    if code1 == "phi2b0_Fe" and code2 == "phi2b1_Fe":
        return (dFe*lambdaFe)/(15*d)
    if code1 == "dzphi2b0_Fe" and code2 == "phi2b1_Fe":
        return -(2*lambdaFe)/(3*d)
    if code1 == "phi2b1_Fe" and code2 == "phi2b2_Fe":
        return (dFe*lambdaFe)/(15*d)
    if code1 == "dzphi2b2_Fe" and code2 == "phi2b1_Fe":
        return (2*lambdaFe)/(3*d)
    if code1 == "phi1b1_Fe" and code2 == "phi2b1_Fe":
        return (dFe*lambdaFe)/(3*d)
    if code1 == "dzphi1b1_Fe" and code2 == "phi2b1_Fe":
        return (2*lambdaFe)/(3*d)
    if code1 == "phi1b0_Fe" and code2 == "phi2b1_Fe":
        return (dFe*lambdaFe)/(3*d)
    if code1 == "dzphi1b0_Fe" and code2 == "phi2b1_Fe":
        return -(2*lambdaFe)/(3*d)
    if code1 == "phi2b1_air" and code2 == "phi2b1_air":
        return (8*d0*lambda0)/(15*d)
    if code1 == "dzphi2b1_air" and code2 == "phi2b1_air":
        return 0
    if code1 == "phi2b0_air" and code2 == "phi2b1_air":
        return (d0*lambda0)/(15*d)
    if code1 == "dzphi2b0_air" and code2 == "phi2b1_air":
        return -(2*lambda0)/(3*d)
    if code1 == "phi2b1_air" and code2 == "phi2b2_air":
        return (d0*lambda0)/(15*d)
    if code1 == "dzphi2b2_air" and code2 == "phi2b1_air":
        return (2*lambda0)/(3*d)
    if code1 == "phi1b1_air" and code2 == "phi2b1_air":
        return (d0*lambda0)/(3*d)
    if code1 == "dzphi1b1_air" and code2 == "phi2b1_air":
        return (2*lambda0)/(3*d)
    if code1 == "phi1b0_air" and code2 == "phi2b1_air":
        return (d0*lambda0)/(3*d)
    if code1 == "dzphi1b0_air" and code2 == "phi2b1_air":
        return -(2*lambda0)/(3*d)
    if code1 == "dzphi2b1" and code2 == "dzphi2b1":
        return (16*d0*lambdaFe+16*dFe*lambda0)/(3*d*d0*dFe)
    if code1 == "dzphi2b1" and code2 == "phi2b0":
        return (2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "dzphi2b0" and code2 == "dzphi2b1":
        return -(8*d0*lambdaFe+8*dFe*lambda0)/(3*d*d0*dFe)
    if code1 == "dzphi2b1" and code2 == "phi2b2":
        return -(2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "dzphi2b1" and code2 == "dzphi2b2":
        return -(8*d0*lambdaFe+8*dFe*lambda0)/(3*d*d0*dFe)
    if code1 == "dzphi2b1" and code2 == "phi1b1":
        return -(2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "dzphi1b1" and code2 == "dzphi2b1":
        return 0
    if code1 == "dzphi2b1" and code2 == "phi1b0":
        return (2*lambdaFe+2*lambda0)/(3*d)
    if code1 == "dzphi1b0" and code2 == "dzphi2b1":
        return 0
    if code1 == "dzphi2b1_Fe" and code2 == "dzphi2b1_Fe":
        return (16*lambdaFe)/(3*d*dFe)
    if code1 == "dzphi2b1_Fe" and code2 == "phi2b0_Fe":
        return (2*lambdaFe)/(3*d)
    if code1 == "dzphi2b0_Fe" and code2 == "dzphi2b1_Fe":
        return -(8*lambdaFe)/(3*d*dFe)
    if code1 == "dzphi2b1_Fe" and code2 == "phi2b2_Fe":
        return -(2*lambdaFe)/(3*d)
    if code1 == "dzphi2b1_Fe" and code2 == "dzphi2b2_Fe":
        return -(8*lambdaFe)/(3*d*dFe)
    if code1 == "dzphi2b1_Fe" and code2 == "phi1b1_Fe":
        return -(2*lambdaFe)/(3*d)
    if code1 == "dzphi1b1_Fe" and code2 == "dzphi2b1_Fe":
        return 0
    if code1 == "dzphi2b1_Fe" and code2 == "phi1b0_Fe":
        return (2*lambdaFe)/(3*d)
    if code1 == "dzphi1b0_Fe" and code2 == "dzphi2b1_Fe":
        return 0
    if code1 == "dzphi2b1_air" and code2 == "dzphi2b1_air":
        return (16*lambda0)/(3*d*d0)
    if code1 == "dzphi2b1_air" and code2 == "phi2b0_air":
        return (2*lambda0)/(3*d)
    if code1 == "dzphi2b0_air" and code2 == "dzphi2b1_air":
        return -(8*lambda0)/(3*d*d0)
    if code1 == "dzphi2b1_air" and code2 == "phi2b2_air":
        return -(2*lambda0)/(3*d)
    if code1 == "dzphi2b1_air" and code2 == "dzphi2b2_air":
        return -(8*lambda0)/(3*d*d0)
    if code1 == "dzphi2b1_air" and code2 == "phi1b1_air":
        return -(2*lambda0)/(3*d)
    if code1 == "dzphi1b1_air" and code2 == "dzphi2b1_air":
        return 0
    if code1 == "dzphi2b1_air" and code2 == "phi1b0_air":
        return (2*lambda0)/(3*d)
    if code1 == "dzphi1b0_air" and code2 == "dzphi2b1_air":
        return 0
    if code1 == "phi2b0" and code2 == "phi2b0":
        return (2*dFe*lambdaFe+2*d0*lambda0)/(15*d)
    if code1 == "dzphi2b0" and code2 == "phi2b0":
        return -(lambdaFe+lambda0)/(2*d)
    if code1 == "phi2b0" and code2 == "phi2b2":
        return -(dFe*lambdaFe+d0*lambda0)/(30*d)
    if code1 == "dzphi2b2" and code2 == "phi2b0":
        return -(lambdaFe+lambda0)/(6*d)
    if code1 == "phi1b1" and code2 == "phi2b0":
        return 0
    if code1 == "dzphi1b1" and code2 == "phi2b0":
        return (lambdaFe+lambda0)/(6*d)
    if code1 == "phi1b0" and code2 == "phi2b0":
        return (dFe*lambdaFe+d0*lambda0)/(6*d)
    if code1 == "dzphi1b0" and code2 == "phi2b0":
        return -(lambdaFe+lambda0)/(6*d)
    if code1 == "phi2b0_Fe" and code2 == "phi2b0_Fe":
        return (2*dFe*lambdaFe)/(15*d)
    if code1 == "dzphi2b0_Fe" and code2 == "phi2b0_Fe":
        return -lambdaFe/(2*d)
    if code1 == "phi2b0_Fe" and code2 == "phi2b2_Fe":
        return -(dFe*lambdaFe)/(30*d)
    if code1 == "dzphi2b2_Fe" and code2 == "phi2b0_Fe":
        return -lambdaFe/(6*d)
    if code1 == "phi1b1_Fe" and code2 == "phi2b0_Fe":
        return 0
    if code1 == "dzphi1b1_Fe" and code2 == "phi2b0_Fe":
        return lambdaFe/(6*d)
    if code1 == "phi1b0_Fe" and code2 == "phi2b0_Fe":
        return (dFe*lambdaFe)/(6*d)
    if code1 == "dzphi1b0_Fe" and code2 == "phi2b0_Fe":
        return -lambdaFe/(6*d)
    if code1 == "phi2b0_air" and code2 == "phi2b0_air":
        return (2*d0*lambda0)/(15*d)
    if code1 == "dzphi2b0_air" and code2 == "phi2b0_air":
        return -lambda0/(2*d)
    if code1 == "phi2b0_air" and code2 == "phi2b2_air":
        return -(d0*lambda0)/(30*d)
    if code1 == "dzphi2b2_air" and code2 == "phi2b0_air":
        return -lambda0/(6*d)
    if code1 == "phi1b1_air" and code2 == "phi2b0_air":
        return 0
    if code1 == "dzphi1b1_air" and code2 == "phi2b0_air":
        return lambda0/(6*d)
    if code1 == "phi1b0_air" and code2 == "phi2b0_air":
        return (d0*lambda0)/(6*d)
    if code1 == "dzphi1b0_air" and code2 == "phi2b0_air":
        return -lambda0/(6*d)
    if code1 == "dzphi2b0" and code2 == "dzphi2b0":
        return (7*d0*lambdaFe+7*dFe*lambda0)/(3*d*d0*dFe)
    if code1 == "dzphi2b0" and code2 == "phi2b2":
        return (lambdaFe+lambda0)/(6*d)
    if code1 == "dzphi2b0" and code2 == "dzphi2b2":
        return (d0*lambdaFe+dFe*lambda0)/(3*d*d0*dFe)
    if code1 == "dzphi2b0" and code2 == "phi1b1":
        return -(lambdaFe+lambda0)/(6*d)
    if code1 == "dzphi1b1" and code2 == "dzphi2b0":
        return -(d0*lambdaFe+dFe*lambda0)/(d*d0*dFe)
    if code1 == "dzphi2b0" and code2 == "phi1b0":
        return -(5*lambdaFe+5*lambda0)/(6*d)
    if code1 == "dzphi1b0" and code2 == "dzphi2b0":
        return (d0*lambdaFe+dFe*lambda0)/(d*d0*dFe)
    if code1 == "dzphi2b0_Fe" and code2 == "dzphi2b0_Fe":
        return (7*lambdaFe)/(3*d*dFe)
    if code1 == "dzphi2b0_Fe" and code2 == "phi2b2_Fe":
        return lambdaFe/(6*d)
    if code1 == "dzphi2b0_Fe" and code2 == "dzphi2b2_Fe":
        return lambdaFe/(3*d*dFe)
    if code1 == "dzphi2b0_Fe" and code2 == "phi1b1_Fe":
        return -lambdaFe/(6*d)
    if code1 == "dzphi1b1_Fe" and code2 == "dzphi2b0_Fe":
        return -lambdaFe/(d*dFe)
    if code1 == "dzphi2b0_Fe" and code2 == "phi1b0_Fe":
        return -(5*lambdaFe)/(6*d)
    if code1 == "dzphi1b0_Fe" and code2 == "dzphi2b0_Fe":
        return lambdaFe/(d*dFe)
    if code1 == "dzphi2b0_air" and code2 == "dzphi2b0_air":
        return (7*lambda0)/(3*d*d0)
    if code1 == "dzphi2b0_air" and code2 == "phi2b2_air":
        return lambda0/(6*d)
    if code1 == "dzphi2b0_air" and code2 == "dzphi2b2_air":
        return lambda0/(3*d*d0)
    if code1 == "dzphi2b0_air" and code2 == "phi1b1_air":
        return -lambda0/(6*d)
    if code1 == "dzphi1b1_air" and code2 == "dzphi2b0_air":
        return -lambda0/(d*d0)
    if code1 == "dzphi2b0_air" and code2 == "phi1b0_air":
        return -(5*lambda0)/(6*d)
    if code1 == "dzphi1b0_air" and code2 == "dzphi2b0_air":
        return lambda0/(d*d0)
    if code1 == "phi2b2" and code2 == "phi2b2":
        return (2*dFe*lambdaFe+2*d0*lambda0)/(15*d)
    if code1 == "dzphi2b2" and code2 == "phi2b2":
        return (lambdaFe+lambda0)/(2*d)
    if code1 == "phi1b1" and code2 == "phi2b2":
        return (dFe*lambdaFe+d0*lambda0)/(6*d)
    if code1 == "dzphi1b1" and code2 == "phi2b2":
        return (lambdaFe+lambda0)/(6*d)
    if code1 == "phi1b0" and code2 == "phi2b2":
        return 0
    if code1 == "dzphi1b0" and code2 == "phi2b2":
        return -(lambdaFe+lambda0)/(6*d)
    if code1 == "phi2b2_Fe" and code2 == "phi2b2_Fe":
        return (2*dFe*lambdaFe)/(15*d)
    if code1 == "dzphi2b2_Fe" and code2 == "phi2b2_Fe":
        return lambdaFe/(2*d)
    if code1 == "phi1b1_Fe" and code2 == "phi2b2_Fe":
        return (dFe*lambdaFe)/(6*d)
    if code1 == "dzphi1b1_Fe" and code2 == "phi2b2_Fe":
        return lambdaFe/(6*d)
    if code1 == "phi1b0_Fe" and code2 == "phi2b2_Fe":
        return 0
    if code1 == "dzphi1b0_Fe" and code2 == "phi2b2_Fe":
        return -lambdaFe/(6*d)
    if code1 == "phi2b2_air" and code2 == "phi2b2_air":
        return (2*d0*lambda0)/(15*d)
    if code1 == "dzphi2b2_air" and code2 == "phi2b2_air":
        return lambda0/(2*d)
    if code1 == "phi1b1_air" and code2 == "phi2b2_air":
        return (d0*lambda0)/(6*d)
    if code1 == "dzphi1b1_air" and code2 == "phi2b2_air":
        return lambda0/(6*d)
    if code1 == "phi1b0_air" and code2 == "phi2b2_air":
        return 0
    if code1 == "dzphi1b0_air" and code2 == "phi2b2_air":
        return -lambda0/(6*d)
    if code1 == "dzphi2b2" and code2 == "dzphi2b2":
        return (7*d0*lambdaFe+7*dFe*lambda0)/(3*d*d0*dFe)
    if code1 == "dzphi2b2" and code2 == "phi1b1":
        return (5*lambdaFe+5*lambda0)/(6*d)
    if code1 == "dzphi1b1" and code2 == "dzphi2b2":
        return (d0*lambdaFe+dFe*lambda0)/(d*d0*dFe)
    if code1 == "dzphi2b2" and code2 == "phi1b0":
        return (lambdaFe+lambda0)/(6*d)
    if code1 == "dzphi1b0" and code2 == "dzphi2b2":
        return -(d0*lambdaFe+dFe*lambda0)/(d*d0*dFe)
    if code1 == "dzphi2b2_Fe" and code2 == "dzphi2b2_Fe":
        return (7*lambdaFe)/(3*d*dFe)
    if code1 == "dzphi2b2_Fe" and code2 == "phi1b1_Fe":
        return (5*lambdaFe)/(6*d)
    if code1 == "dzphi1b1_Fe" and code2 == "dzphi2b2_Fe":
        return lambdaFe/(d*dFe)
    if code1 == "dzphi2b2_Fe" and code2 == "phi1b0_Fe":
        return lambdaFe/(6*d)
    if code1 == "dzphi1b0_Fe" and code2 == "dzphi2b2_Fe":
        return -lambdaFe/(d*dFe)
    if code1 == "dzphi2b2_air" and code2 == "dzphi2b2_air":
        return (7*lambda0)/(3*d*d0)
    if code1 == "dzphi2b2_air" and code2 == "phi1b1_air":
        return (5*lambda0)/(6*d)
    if code1 == "dzphi1b1_air" and code2 == "dzphi2b2_air":
        return lambda0/(d*d0)
    if code1 == "dzphi2b2_air" and code2 == "phi1b0_air":
        return lambda0/(6*d)
    if code1 == "dzphi1b0_air" and code2 == "dzphi2b2_air":
        return -lambda0/(d*d0)
    if code1 == "phi1b1" and code2 == "phi1b1":
        return (dFe*lambdaFe+d0*lambda0)/(3*d)
    if code1 == "dzphi1b1" and code2 == "phi1b1":
        return (lambdaFe+lambda0)/(2*d)
    if code1 == "phi1b0" and code2 == "phi1b1":
        return (dFe*lambdaFe+d0*lambda0)/(6*d)
    if code1 == "dzphi1b0" and code2 == "phi1b1":
        return -(lambdaFe+lambda0)/(2*d)
    if code1 == "phi1b1_Fe" and code2 == "phi1b1_Fe":
        return (dFe*lambdaFe)/(3*d)
    if code1 == "dzphi1b1_Fe" and code2 == "phi1b1_Fe":
        return lambdaFe/(2*d)
    if code1 == "phi1b0_Fe" and code2 == "phi1b1_Fe":
        return (dFe*lambdaFe)/(6*d)
    if code1 == "dzphi1b0_Fe" and code2 == "phi1b1_Fe":
        return -lambdaFe/(2*d)
    if code1 == "phi1b1_air" and code2 == "phi1b1_air":
        return (d0*lambda0)/(3*d)
    if code1 == "dzphi1b1_air" and code2 == "phi1b1_air":
        return lambda0/(2*d)
    if code1 == "phi1b0_air" and code2 == "phi1b1_air":
        return (d0*lambda0)/(6*d)
    if code1 == "dzphi1b0_air" and code2 == "phi1b1_air":
        return -lambda0/(2*d)
    if code1 == "dzphi1b1" and code2 == "dzphi1b1":
        return (d0*lambdaFe+dFe*lambda0)/(d*d0*dFe)
    if code1 == "dzphi1b1" and code2 == "phi1b0":
        return (lambdaFe+lambda0)/(2*d)
    if code1 == "dzphi1b0" and code2 == "dzphi1b1":
        return -(d0*lambdaFe+dFe*lambda0)/(d*d0*dFe)
    if code1 == "dzphi1b1_Fe" and code2 == "dzphi1b1_Fe":
        return lambdaFe/(d*dFe)
    if code1 == "dzphi1b1_Fe" and code2 == "phi1b0_Fe":
        return lambdaFe/(2*d)
    if code1 == "dzphi1b0_Fe" and code2 == "dzphi1b1_Fe":
        return -lambdaFe/(d*dFe)
    if code1 == "dzphi1b1_air" and code2 == "dzphi1b1_air":
        return lambda0/(d*d0)
    if code1 == "dzphi1b1_air" and code2 == "phi1b0_air":
        return lambda0/(2*d)
    if code1 == "dzphi1b0_air" and code2 == "dzphi1b1_air":
        return -lambda0/(d*d0)
    if code1 == "phi1b0" and code2 == "phi1b0":
        return (dFe*lambdaFe+d0*lambda0)/(3*d)
    if code1 == "dzphi1b0" and code2 == "phi1b0":
        return -(lambdaFe+lambda0)/(2*d)
    if code1 == "phi1b0_Fe" and code2 == "phi1b0_Fe":
        return (dFe*lambdaFe)/(3*d)
    if code1 == "dzphi1b0_Fe" and code2 == "phi1b0_Fe":
        return -lambdaFe/(2*d)
    if code1 == "phi1b0_air" and code2 == "phi1b0_air":
        return (d0*lambda0)/(3*d)
    if code1 == "dzphi1b0_air" and code2 == "phi1b0_air":
        return -lambda0/(2*d)
    if code1 == "dzphi1b0" and code2 == "dzphi1b0":
        return (d0*lambdaFe+dFe*lambda0)/(d*d0*dFe)
    if code1 == "dzphi1b0_Fe" and code2 == "dzphi1b0_Fe":
        return lambdaFe/(d*dFe)
    if code1 == "dzphi1b0_air" and code2 == "dzphi1b0_air":
        return lambda0/(d*d0)




    # if  (name1[0] == "d" and name2[0] == "p" and int(name1[-1]) % 2 == int(name2[-1]) % 2) or \
    #     (name2[0] == "d" and name1[0] == "p" and int(name1[-1]) % 2 == int(name2[-1]) % 2) or \
    #     (name1[0] == "p" and name2[0] == "p" and int(name1[-1]) % 2 != int(name2[-1]) % 2) or \
    #     (name1[0] == "d" and name2[0] == "d" and int(name1[-1]) % 2 != int(name2[-1]) % 2):
    #     print(name1, name2, "odd")
    #     return 0

    errString = "        if code1 == \"" + code1 + "\" and code2 == \"" + code2 + "\":\n            return 0"

    print("[\""+ code1 +"\", \""+ code2 + "\"], ")
    # print(errString)
    # copyToClipboard(errString)

    # from createWxMaximaCode import deWrapp4WxMaxima
    # wxMaximaString = deWrapp4WxMaxima(code1, code2)
    # print(wxMaximaString)
    # copyToClipboard(wxMaximaString)




        # raise ValueError("invalid arguemtents "+ name1+ ", "+ name2)
    return __phiXphi__(phi1, phi2, lambdaFe, lambda0, dFe, d0, force_num_int=True)


def getIntegrand4BFINonLin(trialList, testList, lambdanonLin, lambda0, dFe, d0, **kwargs):

    registerd_kwargs = ["conductivity"]
    [print(f"unregistered kwargs '{k}'") for k in kwargs.keys() if k not in registerd_kwargs]

    ret = 0

    if hasattr(lambdanonLin, "__iter__"):
        lambdaFP = lambdanonLin[0]
        lambdanonLin = lambdanonLin[1]

    else:
        lambdaFP = 0
        # lambdanonLin = lambdanonLin

    if isinstance(lambdanonLin, MuHysteresisMS3):
        import myPackage as myP
        diagCF = myP.diagCF
    else:
        diagCF = lambda x:x


    for i in range(len(trialList)):
        phi1 = trialList[i][1]
        for j in range(len(testList)):
            phi2 = testList[j][1]
  
            # both in Iron/air ? 
            inIron = phi1.inIron * phi2.inIron
            inAir = phi1.inAir * phi2.inAir

            #diagCF = diagCF
            lambdaBar = diagCF(0)

            if inIron:
                if phi1.name < phi2.name:
                    name = phi1.name +"*"+ phi2.name
                else:
                    name = phi2.name +"*"+ phi1.name
                if name not in lambdanonLin.GetParts().keys():
                    #register
                    lambdanonLin.RegisterPart(name, phi1, phi2)
                lambdaBar += dFe * lambdanonLin.Average(name) 
            if inAir:
                # numerical integration in air
                lambdaBar += diagCF(lambda0) * num_int(phi1.xStart, phi1.xStart + phi1.d0/2, phi1.DirectEvaluate, phi2.DirectEvaluate)
                lambdaBar += diagCF(lambda0) * num_int(phi1.xStart + phi1.d - phi1.d0/2, phi1.xStart + phi1.d, phi1.DirectEvaluate, phi2.DirectEvaluate)
        
            lambdaBar /= (dFe +  d0)

            if lambdaFP == 0:
                # return value                    
                ret += InnerProduct((lambdaBar  * trialList[i][0]),testList[j][0]) 
            else:
                lambdaFP_bar = __phiXphi__(phi1, phi2, lambdaFP, lambda0, dFe, d0, force_num_int=True)  
                # to do: detect if it should be a diagCF
                lambdaFP_bar = diagCF(lambdaFP_bar)
                ret += InnerProduct(((lambdaFP_bar - lambdaBar)  * trialList[i][0]),testList[j][0]) 


    return ret

def getIntegrand4LFINonLin(f_val, testList, lambdanonLin, lambda0, dFe, d0, **kwargs):

    registerd_kwargs = ["conductivity", "material"]
    [print(f"unregistered kwargs '{k}'") for k in kwargs.keys() if k not in registerd_kwargs]


    # f_val * (v1 + v2 + v3)
    if "material" in kwargs.keys():
        material = kwargs["material"]
    else:
        material = "iron|gap|air"

    phi0 = cl_Phi(0, 0, inIron=True, inAir=True, phiFun=Lobatto, dxphiFun=dxLobatto, material=material).phi

    funcList = [[f_val, phi0]]


    return getIntegrand4BFINonLin(funcList, testList, lambdanonLin, lambda0, dFe, d0)



def getIntegrand4BFI(trialList, testList, lambdaFe, lambda0, dFe=None, d0=None, conductivity=True, force_anisotropy = False, force_num_int=True, **kwargs):


    lambdaFP = 0
    if "lambdaFP" in kwargs.keys():
        lambdaFP = kwargs["lambdaFP"]

    if type(dFe) == type(None):
        dFe = cl_Phi.dFe
    if type(d0) == type(None):
        d0 = cl_Phi.d0

    useTwistetGradient = False
    if "useTwistetGradient" in kwargs.keys():
        useTwistetGradient = bool(int(kwargs["useTwistetGradient"]))

    # get container
    if trialList[0][0].dim == 2 or testList[0][0].dim == 2:
        ret = CF((0, 0))
    else:
        ret = 0

    # execute the multiplication of sums with the averaged micro-shape functions
    for i in range(len(trialList)):
        phi1 = trialList[i][1]
        
        for j in range(len(testList)):
            phi2 = testList[j][1]
            #phi_i * phi_j * u * v

            # anisotropy only for A0
            if force_anisotropy == True or  \
                    (callable(force_anisotropy) and  force_anisotropy(trialList[i][1], testList[j][1]) == True):
                # input("anisotropy")
                if phi1.orientation != 2 or phi2.orientation != 2:
                    input("be carefull with orientation of phi functions != 2")
                if conductivity:
                    conFe = lambdaFe
                    con0 = lambda0
                    resFe = 1/lambdaFe
                    res0 = 1/lambda0
                else:
                    conFe = 1/lambdaFe
                    con0 = 1/lambda0
                    resFe = lambdaFe
                    res0 = lambda0

                # calc arithmetic and harmonic mean
                res_normal = __phiXphi__(phi1, phi2, resFe, res0, dFe, d0, force_num_int=force_num_int, **kwargs)  
                con_laminar = __phiXphi__(phi1, phi2, conFe, con0, dFe, d0, force_num_int=force_num_int, **kwargs)  

                
                
                if con_laminar == 0 and res_normal == 0:
                    continue
                
                if conductivity:
                    lambda_laminar = con_laminar
                    lambda_norm = 1/res_normal
                    
                else:
                    lambda_laminar = 1/con_laminar
                    lambda_norm = res_normal

                
                if __name__ == "__main__" or False:
                    print("conductivity:\t", conductivity)
                    print("laminar:", lambda_laminar)
                    print("normal:\t", lambda_norm)

                # print("order 0 times order 0")
                # cmdInput(locals(), globals())


                if trialList[i][0].dim == 1:
                    lambda_bar = lambda_norm
                if useTwistetGradient and trialList[i][0].dim == 2:
                    if phi1.orientation == 1:
                        # if conductivity:
                        lambda_bar = CF((lambda_laminar, 0, 0, lambda_norm), dims=(2,2))  # verdrehte reihenfolge für rotA rotA' -> gradA gradA'
                        # else:
                        #     lambda_bar = CF((lambda_norm, 0, 0, lambda_laminar), dims=(2,2))  # verdrehte reihenfolge für rotA rotA' -> gradA gradA'
                    elif phi1.orientation == 0:
                        lambda_bar = CF((lambda_norm, 0, 0, lambda_laminar), dims=(2,2))  
                elif not useTwistetGradient and trialList[i][0].dim == 2:
                    if phi1.orientation == 1:
                        lambda_bar = CF((lambda_laminar, 0, 0, lambda_norm), dims=(2,2))  

                    elif phi1.orientation == 0:
                        lambda_bar = CF((lambda_norm , 0, 0, lambda_laminar), dims=(2,2))  
                        

                elif trialList[i][0].dim == 3:
                    lambda_bar = CF((lambda_laminar, 0, 0, 0, lambda_laminar, 0, 0, 0, lambda_norm), dims=(3,3))  
                else:
                    raise RuntimeError("undefined state, think about it")

            else:
                lambda_bar = __phiXphi__(trialList[i][1], testList[j][1], lambdaFe, lambda0, dFe, d0, force_num_int=force_num_int, **kwargs)  

                # print(lambda_bar)

                # print(trialList[i][1].order, testList[j][1].order, lambda_bar)
                if lambda_bar == 0:
                    continue
            

            if lambdaFP == 0:
                # return value      
                              
                # ret += InnerProduct((lambda_bar  * trialList[i][0]),testList[j][0]) 

                ret += (lambda_bar  * trialList[i][0]) * testList[j][0]
            else:
                lambdaFP_bar = __phiXphi__(phi1, phi2, lambdaFP, lambda0, dFe, d0, force_num_int=True)  
                # to do: detect if it should be a diagCF
                lambdaFP_bar = lambdaFP_bar
                ret += InnerProduct(((lambdaFP_bar - lambda_bar)  * trialList[i][0]),testList[j][0]) 
    
    return ret

def getIntegrand4LFI(f_val, testList, lambdaFe, lambda0, dFe, d0, conductivity, force_anisotropy = False, force_num_int=False, **kwargs):
    # f_val * (v1 + v2 + v3)


    registerd_kwargs = ["material"]
    [print(f"unregistered kwargs '{k}'") for k in kwargs.keys() if k not in registerd_kwargs]


    if "material" in kwargs.keys():
        material = kwargs["material"]
    else:
        material = "iron|gap|air"

    phi0 = cl_Phi(0, 0, inIron=True, inAir=True, phiFun=Lobatto, dxphiFun=dxLobatto, orientation=testList[0][1].orientation, material=material).phi


    funcList = [[f_val, phi0]]

    return getIntegrand4BFI(funcList, testList, lambdaFe, lambda0, dFe, d0, conductivity, force_anisotropy=force_anisotropy, force_num_int=force_num_int, **kwargs)


def checkValidBoundaryAndDomainNames(order, mesh):
    for i in order:
        # check boundaries
        bnds = i.dirichlet.split("|")
        for bnd in bnds:
            if bnd not in mesh.GetBoundaries() and bnd !="":
                print(bnd)
                input("boundary name unknown")

        # check domains
        mats = i.material.split("|")
        for mat in mats:
            if mat not in mesh.GetMaterials() and mat != ".*":
                print(mat)
                input("material name unkown >>")



def getPhiPhiValue(phi1, phi2, lambdaFe = None, lambda0 = None, dFe = None, d0 = None, force_num_int=False, **kwargs):
    #symbolic evaluation
    if isinstance(phi1, list) and isinstance(phi2, list):
        if callable(phi1[0]) and callable(phi1[1]) and \
            callable(phi2[0]) and callable(phi2[1]):
            # define phi_Fe :[-dFe/2, dFe/2] -> float
            # define phi_0 :[dFe/2, dFe/2+d0] -> float

            from sympy import integrate, Symbol, ratsimp

            lambdaFe = Symbol("lambdaFe")
            lambdaAir = Symbol("lambdaAir")
            dFe = Symbol("dFe")
            d0 = Symbol("d0")

            z = Symbol("x")
            phi1_Fe = phi1[0](z, dFe, d0)
            phi1_air = phi1[1](z, dFe, d0)
            phi2_Fe = phi2[0](z, dFe, d0)
            phi2_air = phi2[1](z, dFe, d0)

            return ratsimp(1/(dFe+d0) * (   integrate(lambdaFe * phi1_Fe * phi2_Fe, [z, -dFe/2, dFe/2])+\
                                    integrate(lambdaAir * phi1_air * phi2_air, [z, dFe/2, dFe/2 + d0])))

        else:
            raise ValueError("symbolic evaluation of phi*phi only with getPhiPhiValue([phi1_Fe, phi1_0], [phi2_Fe, phi2_0])")
        
    # numerical evaluation
    else:
        return __phiXphi__(phi1, phi2, lambdaFe, lambda0, dFe, d0, force_num_int=force_num_int, **kwargs)



if __name__ == "__main__":
    from symCF import SymScal

    phi0 = SymScal("phi0")
    phi1 = SymScal("phi1")
    
    fun_lin_iron = lambda x, dFe, d0 : 2*x/dFe
    fun_lin_air = lambda x, dFe, d0 : -2*(x-dFe/2-d0/2)/d0
    fun_const_iron = lambda x, dFe, d0 : 1
    fun_const_air = lambda x, dFe, d0 : 1
    # phi1 * phi1
    print(getPhiPhiValue(phi1, phi1)) # as symbols
    print(getPhiPhiValue([fun_lin_iron, fun_lin_air],[fun_lin_iron, fun_lin_air])) # symbolically evaluated
    # phi0 * phi0
    print(getPhiPhiValue(phi0, phi0)) # as symbols
    print(getPhiPhiValue([fun_const_iron, fun_const_air],[fun_const_iron, fun_const_air])) # symbolically evaluated
    # phi0 * phi1
    print(getPhiPhiValue(phi0, phi1)) # as symbols
    print(getPhiPhiValue([fun_const_iron, fun_const_air],[fun_lin_iron, fun_lin_air])) # symbolically evaluated




    from myPackage import myBreak
    myBreak(locals(), globals(), file=__file__.split('/')[-1])




    
