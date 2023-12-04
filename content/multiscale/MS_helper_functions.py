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
        self.xEnd = -zstart
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
        if inAir==False:
            self.name += "_Fe"
        if inIron==False:
            self.name += "_ins"


        self.fullPhi = False
        if "fullPhi" in kwargs.keys():
            self.fullPhi = kwargs["fullPhi"]
            kwargs.pop("fullPhi")


        self.specialFirstLast = False
        if "specialFirstLast" in kwargs.keys():

            self.specialFirstLast = kwargs["specialFirstLast"]
            if self.specialFirstLast:
                if "specialFirstLast_func" not in kwargs.keys():
                    print("specialFirstLast_func is needed as keyword of pyPhiFunction if specialFirstLast is True")
                    raise RuntimeError("specialFirstLast_func is needed as keyword of pyPhiFunction if specialFirstLast is True")
                self.specialFirstLast_func = kwargs["specialFirstLast_func"]
                kwargs.pop("specialFirstLast_func")
            kwargs.pop("specialFirstLast")

        self.forceCF = False
        if "forceCF" in kwargs.keys():
            self.forceCF = kwargs["forceCF"]
            self.forceCF_eval = kwargs["forceCF_eval"]

            kwargs.pop("forceCF")
            kwargs.pop("forceCF_eval")

        self.order = self.name
        if "order" in kwargs.keys():
            self.order = kwargs["order"]
            kwargs.pop("order")


        if len(kwargs.keys()) != 0:
            print("unknown kewords:")
            print(kwargs.keys())

        if self.specialFirstLast and modelHalfAir:
            input("is it indentionally to modelhalfair and have special treatment of the first and the last iron sheet?")

        self.createMask()
        self.createLocX()
            
        if not (self.forceCF == False):      
            if self.inAir and self.inIron:      
                self.fun = [self.forceCF_eval]*2
                super().__init__(self.forceCF)
            elif self.inAir:
                self.fun = [lambda x:0, self.forceCF_eval]
                super().__init__(self.forceCF * IfPos(self.localX - self.dFe/2, 1, 0))
            elif self.inIron:
                self.fun = [self.forceCF_eval, lambda x:0]
                super().__init__(self.forceCF * IfPos(self.localX - self.dFe/2, 0, 1))
            else:
                self.fun = [lambda x:0, lambda x:0]
                super().__init__(CF(0))

        else:

            self.fun = fun
            super().__init__(self.createCFs())


    def createMask(self):
        x_CF = (x, y, z)[self.orientation]
        if self.modelHalfAir:
            self.mask = IfPos(-(x_CF - self.xStart), 0, IfPos(x_CF + self.xStart, 0, 1)) 
        else:
            self.mask = IfPos(-(x_CF - self.xStart - self.d0/2), 0, IfPos(x_CF + self.xStart + self.d0/2, 0, 1)) 


        if self.specialFirstLast:
            self.mask_first = IfPos(x_CF - self.xStart - self.d0/2, IfPos(x_CF - self.xStart - self.d0/2 - self.dFe, 0, 1), 0) 
            self.mask_last = IfPos(x_CF + self.xStart + self.d0/2 + self.dFe, IfPos(x_CF + self.xStart + self.d0/2, 0, 1), 0) 

    
    def createCFs(self):

        funFe = self.fun[0] if self.inIron else lambda x : CF(0)
        funAir = self.fun[1] if self.inAir else lambda x : CF(0)

        # dxfunFe = self.dxfun[0] if self.inIron else lambda x : CF(0)
        # dxfunAir = self.dxfun[1] if self.inAir else lambda x : CF(0)

        if self.specialFirstLast:

            # print(self.specialFirstLast_func[0])
            self.phi = IfPos(self.localX - self.dFe/2, funAir(self.localX), funFe(self.localX)) * (self.mask - self.mask_first - self.mask_last) +\
                        IfPos(self.localX - self.dFe/2,  0, self.specialFirstLast_func[0](self.localX)) * (self.mask_first) +\
                        IfPos(self.localX - self.dFe/2,  0, self.specialFirstLast_func[1](self.localX)) * (self.mask_last)
        else:
            self.phi = IfPos(self.localX - self.dFe/2, funAir(self.localX), funFe(self.localX)) * self.mask
             

        # self.dzphi = IfPos(self.localX - self.dFe/2, dxfunAir(self.localX), dxfunFe(self.localX)) * self.mask
        return self.phi #, self.dzphi



    def createLocX(self):
        # get coordinate
        x_CF = (x, y, z)[self.orientation]
        # start point
        xPos = self.xStart - self.d0/2 - self.dFe
        if not self.modelHalfAir:
            xPos = self.xStart - self.d0/2 - self.dFe
        istart = -1

        ret = CF(0)
        for i in range(istart, self.numSheets):

            sheetOrigin = x_CF - xPos - self.dFe/2

            # if self.specialFirstLast == True and i == 0:
            #     ret += IfPos(sheetOrigin  + self.dFe/2 , IfPos(sheetOrigin  - self.dFe/2 - self.d0, 0, IfPos(sheetOrigin - self.dFe/2, sheetOrigin, sheetOrigin/2 + self.dFe/4)), 0)                           
            # elif self.specialFirstLast == True and i == self.numSheets-1:
            #     ret += IfPos(sheetOrigin  + self.dFe/2 , IfPos(sheetOrigin  - self.dFe/2 - self.d0, 0, IfPos(sheetOrigin - self.dFe/2, sheetOrigin, sheetOrigin/2 - self.dFe/4)), 0)                           
            # else:
            ret += IfPos(sheetOrigin  + self.dFe/2 , IfPos(sheetOrigin  - self.dFe/2 - self.d0, 0, sheetOrigin), 0)                           
            xPos += self.d0 + self.dFe

        ret *= self.mask

        self.localX = ret

        return self.localX

    def isIron(self, x):
        x = self.map2sheet(x)

        return x >= -self.dFe/2 and x <= self.dFe/2

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

        if self.modelHalfAir and (x < self.xStart or x > -self.xStart):
            return 0
        elif not self.modelHalfAir and (x < self.xStart + self.d0/2 or x > -self.xStart - self.d0/2):
            return 0

        x_mapped = self.map2sheet(x)
        if self.specialFirstLast and x < self.xStart + self.d0/2 + self.dFe:
             return self.specialFirstLast_func[0](x_mapped) if self.inIron else 0
        if self.specialFirstLast and x > -self.xStart - self.d0/2 - self.dFe:    
             return self.specialFirstLast_func[1](x_mapped) if self.inIron else 0
        # iron or air
        if x_mapped <= self.dFe/2 :
            return self.fun[0](x_mapped) if self.inIron else 0
        else: 
            return self.fun[1](x_mapped) if self.inAir else 0

    def getDiff(self):

        from sympy import diff, Symbol, lambdify
        x_sym = Symbol("x")
    
        funFe = self.fun[0]
        fun_symFe = funFe(x_sym)
        dxfun_symFe = diff(fun_symFe, x_sym)
        dxfunFe = lambdify(x_sym, dxfun_symFe)
        
        funAir = self.fun[1]
        fun_symAir = funAir(x_sym)
        dxfun_symAir = diff(fun_symAir)
        dxfunAir = lambdify(x_sym, dxfun_symAir)

        # if type(coordinate) == type(None):
        #     coordinate = (x, y, z)[self.orientation]
            
        # dxfunFe = self.fun[0].Diff(coordinate) if self.inIron else lambda x : CF(0)
        # dxfunAir = self.fun[1].Diff(coordinate) if self.inAir else lambda x : CF(0)

        # return [dxfunFe, dxfunAir] 
        return pyPhiFunction(self.zstart, self.d, [dxfunFe, dxfunAir], self.numSheets, self.ff, self.orientation, self.inIron, self.inAir, self.inOuter, self.douter, self.modelHalfAir, "dx" + self.name)
    def getInt(self):

        from sympy import integrate, Symbol, lambdify
        x_sym = Symbol("x")
    
        dxfunFe = self.fun[0]
        dxfun_symFe = dxfunFe(x_sym)
        fun_symFe = integrate(dxfun_symFe, x_sym)
        funFe = lambdify(x_sym, fun_symFe- fun_symFe.subs(x_sym, self.dFe/2))
        
        dxfunAir = self.fun[1]
        dxfun_symAir = dxfunAir(x_sym)
        fun_symAir = integrate(dxfun_symAir)
        funAir = lambdify(x_sym, fun_symAir- fun_symAir.subs(x_sym, self.dFe/2)) # - constatnt s.t. it is a bubble


        # if type(coordinate) == type(None):
        #     coordinate = (x, y, z)[self.orientation]
            
        # dxfunFe = self.fun[0].Diff(coordinate) if self.inIron else lambda x : CF(0)
        # dxfunAir = self.fun[1].Diff(coordinate) if self.inAir else lambda x : CF(0)

        # return [dxfunFe, dxfunAir] 
        return pyPhiFunction(self.zstart, self.d, [funFe, funAir], self.numSheets, self.ff, self.orientation, self.inIron, self.inAir, self.inOuter, self.douter, self.modelHalfAir, "dx" + self.name)


class pyPhiConst(pyPhiFunction):
    def __init__(self, val=1, inAir=True, inIron=True): 
        self.val = val
        super().__init__(cl_Phi.getZStart(), cl_Phi.getD(), lambda x:val, cl_Phi.numSheets, cl_Phi.getFF(), cl_Phi.orientation, inAir=inAir, inIron=inIron, 
                        name=f"pyPhiConstant", forceCF=CF(val), forceCF_eval = lambda x: val)

class pyPhiZero(pyPhiConst):
    def __init__(self, inAir=True, inIron=True): 
        super().__init__(0, inAir=inAir, inIron=inIron)


class pydxLobatto(pyPhiFunction):
    def __init__(self, zstart, d, order, numSheets, ff, orientation, inIron=True, inAir=True, inOuter=False, douter=None, modelHalfAir=True, **kwargs): 

        dFe = d*ff
        d0 = d*(1-ff)
        xs = (d0/2 + dFe/2)
        self.order = order
        self.dxFunDict = {
            # [fun in Fe, fun in insulation]
            0:  [lambda x : 0, lambda x : 0],
            1:  [lambda x : 2.0/dFe , lambda x : -2.0/d0 ],
            2:  [lambda x : sqrt(3.0/8)*(8/(dFe**2) *x), lambda x : sqrt(3.0/8)*(8/(d0**2) *(x-dFe/2-d0/2))], 
            3:  [lambda x : sqrt(5.0/8)*(24/(dFe**3) *(x**2)- 2/dFe), lambda x : sqrt(5.0/8)*(24/(d0**3) *((x-dFe/2-d0/2)**2)- 2/d0)], 
            4:  [lambda x : sqrt(7.0/128)*(320/(dFe**4) *(x**3)- 48/(dFe**2)*x), lambda x : sqrt(7.0/128)*(320/(d0**4) *((x-dFe/2-d0/2)**3)- 48/(d0**2)*(x-dFe/2-d0/2))],

            6:  [lambda x: 1.0/16*sqrt(11/2)*((8064*(x-dFe/2-d0/2)**5)/(d0)**6-(2240*(x-dFe/2-d0/2)**3)/(d0)**4+(120*(x-dFe/2-d0/2))/(d0)**2),  lambda x : 1.0/16*sqrt(11/2)*((8064*(x)**5)/dFe**6-(2240*(x)**3)/dFe**4+(120*x)/dFe**2)], 
            -1:  [lambda x : -2.0/dFe , lambda x : 2.0/d0 ],
            -2:  [lambda x : -sqrt(3.0/8)*(8/(dFe**2) *x), lambda x : -sqrt(3.0/8)*(8/(d0**2) *(x-dFe/2-d0/2))], 
        }   

        super().__init__(zstart, d, self.dxFunDict[self.order], numSheets, ff, orientation, inIron=inIron, inAir=inAir, 
                        inOuter=inOuter, douter=douter, modelHalfAir=modelHalfAir, name=f"pydxLobatto({order})", order = self.order, **kwargs)

class pyLobatto(pyPhiFunction):
    def __init__(self, zstart, d, order, numSheets, ff, orientation, inIron=True, inAir=True, inOuter=False, douter=None, modelHalfAir=True, **kwargs): 
        dFe = d*ff
        d0 = d*(1-ff)
        xs = (d0/2 + dFe/2)
        self.order = order
        self.funDict = {
            # [fun in Fe, fun in insulation]
            0:  [lambda x : 1, lambda x : 1],
            1:  [lambda x :  2.0/dFe * x, lambda x : -2.0/d0 * (x - xs)],
            2:  [lambda x : sqrt(3.0/8) * (4/(dFe**2) *(x**2)- 1), lambda x : sqrt(3.0/8)*(4/(d0**2)*((x-xs)**2)- 1)],
            3:  [lambda x : sqrt(5.0/8)*(8/(dFe**3) *(x**3)- 2/dFe*x), lambda x : sqrt(5.0/8)*(8/(d0**3) *((x-xs)**3)- 2/d0*(x-xs))],
		    4:  [lambda x : sqrt(7.0/128)*(80/(dFe**4) * (x**4)- 24/(dFe**2)*(x**2)+1), lambda x : sqrt(7.0/128)*(80/(d0**4) *((x-xs)**4)- 24/(d0**2)*((x-xs)**2)+1)],

            6:  [lambda x: 1.0/16*sqrt(11/2)*((x*2/dFe)**2-1)*(21*(x*2/dFe)**4-14*(x*2/dFe)**2+1),  lambda x : 1.0/16*sqrt(11/2)*(((x-xs)*2/d0)**2-1)*(21*((x-xs)*2/d0)**4-14*((x-xs)*2/d0)**2+1)], 
            -1:  [lambda x : -( 2.0/dFe * x), lambda x : -(-2.0/d0 * (x - xs))],
            -2:  [lambda x : -(sqrt(3.0/8) * (4/(dFe**2) *(x**2)- 1)), lambda x : -(sqrt(3.0/8)*(4/(d0**2)*((x-xs)**2)- 1))],
            
        }

        super().__init__(zstart, d, self.funDict[self.order], numSheets, ff, orientation, inIron=inIron, inAir=inAir, 
                                inOuter=inOuter, douter=douter, modelHalfAir=modelHalfAir, name=f"pyLobatto({order})", order = self.order, **kwargs)

        



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
    zstart = None
    douter = np.nan
    mesh = None
    modelHalfAir = True
    phiFunction = pyLobatto
    dzPhiFunction = pydxLobatto
    orientation = 2
    useGradients = True
    useAbsolutes = True
    useCurls = True

    def getZStart():
        return  -cl_Phi.numSheets*(cl_Phi.dFe+cl_Phi.d0)/2
    def getD():
        return  (cl_Phi.dFe+cl_Phi.d0)
    def getFF():
        return cl_Phi.dFe/(cl_Phi.dFe+cl_Phi.d0)
        
    def __init__(self, phi_order, fes_order=0, inIron=True, inAir=True, inOuter=False, material=".*", dirichlet="", **kwargs):


        # ------------------------------------------------------------------------------
        # --- kwargs
        # ------------------------------------------------------------------------------
        self.numSheets = cl_Phi.numSheets
        if "numSheets" in kwargs.keys():
            self.numSheets = kwargs["numSheets"]
            kwargs.pop("numSheets")
        
        self.dFe = cl_Phi.dFe
        if "dFe" in kwargs.keys():
            self.dFe = kwargs["dFe"]
            kwargs.pop("dFe")

        self.d0 = cl_Phi.d0
        if "d0" in kwargs.keys():
            self.d0 = kwargs["d0"]
            kwargs.pop("d0")


        self.douter = cl_Phi.douter
        if "douter" in kwargs.keys():
            self.douter = kwargs["douter"]
            kwargs.pop("douter")

        self.orientation = cl_Phi.orientation
        if "orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]
            kwargs.pop("orientation")
        
        self.modelHalfAir = cl_Phi.modelHalfAir
        if "modelHalfAir" in kwargs.keys():
            self.modelHalfAir = kwargs["modelHalfAir"]
            kwargs.pop("modelHalfAir")

        self.phiFunction = cl_Phi.phiFunction
        if "phiFunction" in kwargs.keys():
            self.phiFunction = kwargs["phiFunction"]
            kwargs.pop("phiFunction")
        
        self.dzPhiFunction = cl_Phi.dzPhiFunction 
        if "dxPhiFunction" in kwargs.keys():
            self.dzPhiFunction = kwargs["dxPhiFunction"]
            kwargs.pop("dxPhiFunction")
        elif "dyPhiFunction" in kwargs.keys():
            self.dzPhiFunction =  kwargs["dyPhiFunction"]
            kwargs.pop("dyPhiFunction")
        elif "dzPhiFunction" in kwargs.keys():
            self.dzPhiFunction =  kwargs["dzPhiFunction"]           
            kwargs.pop("dzPhiFunction")

        self.useGradients=cl_Phi.useGradients
        if "useGradients" in kwargs.keys():
            self.useGradients = kwargs["useGradients"]
            kwargs.pop("useGradients")

        self.useAbsolutes = cl_Phi.useAbsolutes
        if "useAbsolutes" in kwargs.keys():
            self.useAbsolutes = kwargs["useAbsolutes"]
            kwargs.pop("useAbsolutes")

        self.useCurls = cl_Phi.useCurls
        if "useCurls" in kwargs.keys():
            self.useCurls = kwargs["useCurls"]
            kwargs.pop("useCurls")


        assert type(self.numSheets) != type(None)
        assert type(self.dFe) != type(None)
        assert type(self.d0) != type(None)
        
        self.zstart = -self.numSheets*(self.dFe+self.d0)/2
        if "zstart" in kwargs.keys():
            self.zstart = kwargs["zstart"]
            kwargs.pop("zstart")


        self.specialFirstLast = False
        if "specialFirstLast" in kwargs.keys():
            self.specialFirstLast = kwargs["specialFirstLast"]
            if "specialFirstLast_func" not in kwargs.keys() or "specialFirstLast_dzfunc" not in kwargs.keys():
                print("specialFirstLast_func and specialFirstLast_dzfunc is needed as keyword of pyPhiFunction if specialFirstLast is True")
                raise RuntimeError("specialFirstLast_func is needed as keyword of pyPhiFunction if specialFirstLast is True")
            self.specialFirstLast_func = kwargs["specialFirstLast_func"]
            self.specialFirstLast_dzfunc = kwargs["specialFirstLast_dzfunc"]
            kwargs.pop("specialFirstLast")
            kwargs.pop("specialFirstLast_func")
            kwargs.pop("specialFirstLast_dzfunc")

        self.nograds = False
        if "nograds" in kwargs.keys():
            self.nograds = kwargs["nograds"]
            kwargs.pop("nograds")

        # ------------------------------------------------------------------------------
        # --- end kwargs
        # ------------------------------------------------------------------------------
        if len(kwargs.keys()) != 0:
            print("unknown kewords:")
            print(kwargs.keys())


        checkValidBoundaryAndDomainNames = True
        if "checkValidBoundaryAndDomainNames" in kwargs.keys():
            checkValidBoundaryAndDomainNames = kwargs["checkValidBoundaryAndDomainNames"]
            kwargs.pop("checkValidBoundaryAndDomainNames")


        if self.numSheets == None or self.dFe == None or self.d0 == None:
            raise ValueError("set numSheets, dFe and d0 first -> cl_Phi.numSheets = N, cl_Phi.dFe = ... ")

        self.ff = self.dFe/(self.d0+self.dFe)
                      
        if isinstance(phi_order, (int, float)):
            if True:                
            # try:
                if self.specialFirstLast:
                    self.phi = self.phiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir, specialFirstLast=self.specialFirstLast, specialFirstLast_func=self.specialFirstLast_func)
                    self.dzphi = self.dzPhiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir, specialFirstLast=self.specialFirstLast, specialFirstLast_func=self.specialFirstLast_dzfunc)
               
                    # self.dzphi.phi = self.phi.Diff(modelHalfAir=self.modelHalfAir,(x, y, z)[self.orientation])
                    print("create dzphi functions with firstLast=True")
                else:
                    self.phi = self.phiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir, specialFirstLast=False)
                    self.dzphi = self.dzPhiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir, specialFirstLast=False)
                    

            else:
            # except:
                try:
                    self.phi = self.phiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir)
                    self.dzphi = self.dzPhiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir, inOuter=inOuter, douter=self.douter, modelHalfAir=self.modelHalfAir)
                    print("create phi functions without firstLast")
                
                except:
                    print("failed to create phi functions")
                    self.phi = self.phiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir)
                    self.dzphi = self.dzPhiFunction(self.zstart , self.dFe+self.d0, phi_order, self.numSheets, self.ff, self.orientation, inIron=inIron, inAir=inAir)
            self.name = self.phi.name

        else:
            self.phi = phi_order[0]
            self.dzphi = phi_order[1]

            
            if hasattr(self.phi, "coordinate"):
                self.orientation = self.phi.coordinate
            else:
                self.orientation = self.phi.orientation
            self.zstart = self.phi.xStart
            self.name = self.phi.name

            if self.phi.inAir != inAir:
                print("warning: cl_Phi parameter in Air is ignored. Set it in definition of pyPhiFunction")
            if self.phi.inIron != inIron:
                print("warning: cl_Phi parameter in Iron is ignored. Set it in definition of pyPhiFunction")
            if "modelHalfAir" in kwargs.keys() and self.phi.modelHalfAir != kwargs["modelHalfAir"]:
                print("warning: cl_Phi parameter modelHalfAir is ignored. Set it in definition of pyPhiFunction")

        self.fes_oder = fes_order
        self.inIron = self.phi.inIron
        self.inAir = self.phi.inAir
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

    def plotDirectEvaluated(orderPhi, xi= None, nFig=1):
        import matplotlib.pyplot as plt

        if len(orderPhi) == 0:
            return

        if type(xi) == type(None):
            xi = np.linspace(orderPhi[0].zstart, -orderPhi[0].zstart, 10000)
        plt.figure(nFig)
        plt.clf()
        
        plt.subplot(2, 1, 1)
        plt.title("phis for Phi")

        # plot sheets
        xpos = orderPhi[0].zstart + cl_Phi.d0/2
        d = orderPhi[0].d0 + orderPhi[0].dFe
        for i in range(orderPhi[0].numSheets):
            plt.fill([xpos, xpos+cl_Phi.dFe, xpos+cl_Phi.dFe, xpos], [-1, -1, 1, 1], c="lightgray")
            plt.plot([xpos, xpos], [-1, 1], "--k")
            plt.plot([xpos+cl_Phi.dFe, xpos+cl_Phi.dFe], [-1, 1], "--k")
            xpos += d

        # [plt.plot(xi, [o.phi.DirectEvaluate(xx) for xx in xi], label=o.phi.name) for o in orderPhi ]

        [plt.plot(xi, [o.phi.DirectEvaluate(xx) for xx in xi], label=o.phi.name if hasattr(o.phi, "name") else "") for o in orderPhi ]
        plt.legend()
        plt.subplot(2, 1, 2)
        xpos = orderPhi[0].zstart + cl_Phi.d0/2
        for i in range(orderPhi[0].numSheets):
            plt.fill([xpos, xpos+cl_Phi.dFe, xpos+cl_Phi.dFe, xpos], [-2/cl_Phi.dFe, -2/cl_Phi.dFe, 2/cl_Phi.dFe, 2/cl_Phi.dFe], c="lightgray")
            plt.plot([xpos, xpos], [-1, 1], "--k")
            plt.plot([xpos+cl_Phi.dFe, xpos+cl_Phi.dFe], [-1, 1], "--k")
            xpos += d

        # [plt.plot(xi, [o.dzphi.DirectEvaluate(x) for x in xi], label=o.phi.name) for o in orderPhi ]
        [plt.plot(xi, [o.dzphi.DirectEvaluate(xx) for xx in xi], label=o.dzphi.name if hasattr(o.dzphi, "name") else f"diff_{o.phi.name}") for o in orderPhi ]
        
        plt.legend()
        # [o.plot() for o in orderPhi]


        plt.show()

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

        # plot sheets
        xpos = orderPhi[0].zstart + cl_Phi.d0/2
        d = orderPhi[0].d0 + orderPhi[0].dFe
        for i in range(orderPhi[0].numSheets):
            plt.fill([xpos, xpos+cl_Phi.dFe, xpos+cl_Phi.dFe, xpos], [-1, -1, 1, 1], c="lightgray")
            plt.plot([xpos, xpos], [-1, 1], "--k")
            plt.plot([xpos+cl_Phi.dFe, xpos+cl_Phi.dFe], [-1, 1], "--k")
            xpos += d

        # [plt.plot(xi, [o.phi.DirectEvaluate(xx) for xx in xi], label=o.phi.name) for o in orderPhi ]

        if cl_Phi.orientation == 0:
            [plt.plot(xi, [o.phi(cl_Phi.mesh(xx, 0, 0)) for xx in xi], label=o.phi.name if hasattr(o.phi, "name") else "") for o in orderPhi ]
        if cl_Phi.orientation == 1:
            [plt.plot(xi, [o.phi(cl_Phi.mesh(0, yy, 0)) for yy in xi], label=o.phi.name if hasattr(o.phi, "name") else "") for o in orderPhi ]
        if cl_Phi.orientation == 2:
            [plt.plot(xi, [o.phi(cl_Phi.mesh(0, 0, zz)) for zz in xi], label=o.phi.name if hasattr(o.phi, "name") else "") for o in orderPhi ]
        plt.legend()
        plt.subplot(2, 1, 2)
        xpos = orderPhi[0].zstart + cl_Phi.d0/2
        for i in range(orderPhi[0].numSheets):
            plt.fill([xpos, xpos+cl_Phi.dFe, xpos+cl_Phi.dFe, xpos], [-2/cl_Phi.dFe, -2/cl_Phi.dFe, 2/cl_Phi.dFe, 2/cl_Phi.dFe], c="lightgray")
            plt.plot([xpos, xpos], [-1, 1], "--k")
            plt.plot([xpos+cl_Phi.dFe, xpos+cl_Phi.dFe], [-1, 1], "--k")
            xpos += d

        # [plt.plot(xi, [o.dzphi.DirectEvaluate(x) for x in xi], label=o.phi.name) for o in orderPhi ]
        if cl_Phi.orientation == 0:
            [plt.plot(xi, [o.dzphi(cl_Phi.mesh(xx, 0, 0)) for xx in xi], label=o.dzphi.name if hasattr(o.dzphi, "name") else f"diff_{o.phi.name}") for o in orderPhi ]
        if cl_Phi.orientation == 1:
            [plt.plot(xi, [o.dzphi(cl_Phi.mesh(0, yy, 0)) for yy in xi], label=o.dzphi.name if hasattr(o.dzphi, "name") else f"diff_{o.phi.name}") for o in orderPhi ]
        if cl_Phi.orientation == 2:
            [plt.plot(xi, [o.dzphi(cl_Phi.mesh(0, 0, zz)) for zz in xi], label=o.dzphi.name if hasattr(o.dzphi, "name") else f"diff_{o.phi.name}") for o in orderPhi ]
        plt.legend()
        # [o.plot() for o in orderPhi]


        plt.show()



    def getStrOfCl_PhiParameters():
        ret = ""
        ret += "numSheets\t" + str(cl_Phi.numSheets)+ "\n"
        ret += "dFe\t" + str(cl_Phi.dFe)+ "\n"
        ret += "d0\t" + str(cl_Phi.d0)+ "\n"
        ret += "zstart\t" + str(cl_Phi.zstart)+ "\n"
        ret += "douter\t" + str(cl_Phi.douter)+ "\n"
        ret += "mesh\t" + str(cl_Phi.mesh)+ "\n"
        ret += "modelHalfAir\t" + str(cl_Phi.modelHalfAir)+ "\n"
        ret += "phiFunction\t" + str(cl_Phi.phiFunction)+ "\n"
        ret += "dzPhiFunction\t" + str(cl_Phi.dzPhiFunction)+ "\n"
        ret += "orientation\t" + str(cl_Phi.orientation)+ "\n"
        ret += "useGradients\t" + str(cl_Phi.useGradients)+ "\n"
        ret += "useAbsolutes\t" + str(cl_Phi.useAbsolutes)+ "\n"
        ret += "useCurls\t" + str(cl_Phi.useCurls)+ "\n"
        return ret
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
        return checkValidBoundaryAndDomainNames([self], mesh)

class cl_MS():

    def __init__(self, orderPhi, sol, **kwargs):

        
        self.orderPhi = orderPhi
        self.dim = sol.space.mesh.dim
        self.sol = sol
        self.trials, self.tests = sol.space.TnT()

        if not hasattr(self.trials, "__iter__"):
            raise RuntimeError("Componed FE Space needed. Use fes = FESpace([fes])")
        
        # ------------------------------------------------------------------------------
        # --- kwargs
        # ------------------------------------------------------------------------------
        self.orientation = cl_Phi.orientation
        if "self.orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]
            kwargs.pop("orientation")

        self.istart = 0
        if "istart" in kwargs.keys():
            self.istart = kwargs["istart"]
            kwargs.pop("istart")

        self.ansatz  = ""
        if "ansatz" in kwargs.keys():
            self.ansatz = kwargs["ansatz"]
            kwargs.pop("ansatz")

        self.addPhi0Outer = True
        if "addPhi0Outer" in kwargs.keys():
            self.addPhi0Outer = kwargs["addPhi0Outer"]
            kwargs.pop("addPhi0Outer")


        # ------------------------------------------------------------------------------
        # --- end kwargs
        # ------------------------------------------------------------------------------
        if len(kwargs.keys()) != 0:
            print("cl_MS unknown kewords:")
            print(kwargs.keys())

        self.coupling_matrix = None
        self.sol_pack = []
        self.sol_trace_n_pack = []
        
        

        self.u_pack = []
        self.v_pack = []

        self.phi0 = pyPhiConst()


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

    
    def generateCouplingMatrix(self, terms_u, terms_v, lambdaFe, lambdaAir, force_num_int=False, **kwargs):

        self.lambdaAir = lambdaAir
        self.lambdaFe = lambdaFe
        self.coupling_matrix = {}#np.zeros([len(terms_u), len(terms_v)])
        for i in range(len(terms_u)):
            self.coupling_matrix[terms_u[i][1]] = {}
            for j in range(len(terms_v)):
                # print(terms_u[i][1].name, terms_v[j][1].name, __phiXphi__(terms_u[i][1], terms_v[j][1], lambdaFe, lambdaAir, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, )  )
                # print(terms_v[j][1].name, terms_u[i][1].name, __phiXphi__(terms_v[j][1], terms_u[i][1], lambdaFe, lambdaAir, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, )  )
                # print("..........")
                lambda_bar = __phiXphi__(terms_u[i][1], terms_v[j][1], self.lambdaFe, self.lambdaAir, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, **kwargs)  
                
                self.coupling_matrix[terms_u[i][1]][terms_v[j][1]] = lambda_bar
        
        return self.coupling_matrix
    def checkCouplingMatrxiSymmetric(self, terms_u, terms_v, eps=1e-5):
        if type(self.coupling_matrix) == type(None):
            print("generate the matrix first")
            return None
        ret = True
        for i in range(len(terms_u)):
            phi_i = terms_u[i][1]
            for j in range(len(terms_v)):
                phi_j = terms_v[j][1]

                if abs(self.coupling_matrix[phi_i][phi_j] - self.coupling_matrix[phi_j][phi_i])/abs(self.coupling_matrix[phi_i][phi_j] + 1e-15) > eps:
                    print(i, j, self.coupling_matrix[phi_i][phi_j], self.coupling_matrix[phi_j][phi_i])
                    ret = False
        return ret

    def setElementCouplingMatrix(self, terms_u, terms_v, r, c, val):
        self.coupling_matrix[terms_u[r][1]][terms_v[c][1]] = val
    def getElementCouplingMatrix(self, terms_u, terms_v, r, c):
        return self.coupling_matrix[terms_u[r][1]][terms_v[c][1]]

    def printCouplingMatrix(self, terms_u, terms_v, sparsity=False, round_digits=3):
        if type(self.coupling_matrix) == type(None):
            print("generate the matrix first")
            return None

        from tabulate import tabulate
        
        ret = [["", ""], ["", ""]]
        
        for i in range(len(terms_v)):
            ret[0] += [i]
        for i in range(len(terms_v)):
            ret[1] += [terms_v[i][1].name]

        for i in range(len(terms_u)):
            phi_i = terms_u[i][1]
            ret += [[i, phi_i.name]]
            for j in range(len(terms_v)):
                phi_j = terms_v[j][1]
                if not sparsity:
                    ret[i+2] += [round(float(self.coupling_matrix[phi_i][phi_j]), round_digits)]    
                else:
                    ret[i+2] += " " if float(self.coupling_matrix[phi_i][phi_j]) == 0  else "x"
                    
        

        print(tabulate(ret, headers="firstrow"))
        return tabulate(ret, headers="firstrow")


    def getIntegrand4BFI(self, terms_u, terms_v, lambdaFe=None, lambdaAir=None, force_num_int = False, force_recalc = False, **kwargs):
        if type(self.coupling_matrix) == type(None) or (lambdaFe != None and lambdaFe != self.lambdaFe) or (lambdaAir != None and lambdaAir != self.lambdaAir):

            self.generateCouplingMatrix(lambdaFe=lambdaFe, lambdaAir=lambdaAir, force_num_int=force_num_int, **kwargs)

        ret = CF(0)
        for i in range(len(terms_u)):
            phi_i = terms_u[i][1]
            for j in range(len(terms_v)):
                phi_j = terms_v[j][1]
                # recalculate value and store it
                if force_recalc or not phi_i in self.coupling_matrix.keys() or not phi_j in self.coupling_matrix[phi_i].keys():
                    if not force_recalc:
                        print("Warning: element "+ phi_i.name + " "  + phi_j.name + " not in coupling matrix -> adding")
                    # calculate it and 
                    lambda_bar = __phiXphi__(phi_i, phi_j, lambdaFe, lambdaAir, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, **kwargs)  
                    if phi_i not in self.coupling_matrix.keys():
                        # add a container
                        self.coupling_matrix[phi_i] = {}
                    # add it
                    self.coupling_matrix[phi_i][phi_j] = lambda_bar
                ret += InnerProduct(self.coupling_matrix[phi_i][phi_j] * terms_u[i][0], terms_v[j][0])

        return ret

    def getIntegrand4LFI(self, val, terms_v, lambdaFe=None, lambdaAir=None, force_num_int = False, force_recalc = False, **kwargs):
        return self.getIntegrand4BFI([[val, self.phi0]], terms_v, lambdaFe=lambdaFe, lambdaAir=lambdaAir, force_num_int=force_num_int, force_recalc=force_recalc, **kwargs)



class cl_curlcurlMS(cl_MS):
    def __init__(self, orderT, sol, eddy_inplane=False, **kwargs):

        # ------------------------------------------------------------------------------
        # --- kwargs
        # ------------------------------------------------------------------------------
        
        # ------------------------------------------------------------------------------
        # --- end kwargs
        # ------------------------------------------------------------------------------
    
        
        super().__init__(orderT, sol, **kwargs)
        

        # self.u_pack = []
        # self.v_pack = []

        self.tri_curlu = []
        self.tri_curlv = []      
        self.tri_curlsol = []


        self.u_trace_n_pack = []
        self.v_trace_n_pack = []


        if eddy_inplane:
            # H1 -> eddy curretns in z-direction
            self.fillContainers_2D_eddy_inPlane()
        else:
            self.fillContainers_2D_eddy_normal()




    def myCross(self, a, b):
        if self.dim == 3:
            # (ax ay az) x (bx by bz) = (ay*bz-az*by, az*bx-ax*bz, ax*by-ay-bx)
            return Cross(a, b)
        elif self.dim == 2:
            # (ax ay 0) x (bx by 0) = (0, 0, ax*by-ay-bx)
            # return (a[0]*b[1] - a[1]*b[0])
            return Cross(CF((a[0], a[1], 0)), CF((b[0], b[1], 0)))[2]
        else:
            # (ax 0 0) x (bx 0 0) = (0, 0, 0)
            return 0


    def fillContainers_2D_eddy_normal(self):
        n = specialcf.normal(self.dim)
        sol_c = self.sol.components
        i = self.istart

        # T in Hcurl -> J = (dx Ty - dy Tx) ez
        for phi_i  in self.orderPhi:
            phi = phi_i.phi
            dzphi = phi_i.dzphi


            # curl(u) * phi
            if phi_i.useCurls:
                self.tri_curlu += [[curl(self.trials[i]), 1, phi ]] # ok 
                self.tri_curlv += [[curl(self.tests[i]), 1, phi]]
                self.tri_curlsol += [[curl(sol_c[i]), 1, phi ]] # ok 

            
            if type(dzphi) != pyPhiZero and phi_i.useAbsolutes:
                # grad(phi) x u = e x u * dzphi
                self.tri_curlu += [[self.trials[i], self.e, dzphi]] # ok 
                self.tri_curlv += [[self.tests[i], self.e, dzphi]]
                self.tri_curlsol += [[sol_c[i], self.e, dzphi]]
            
            
            self.u_pack += [[self.trials[i], phi]] # ok
            self.v_pack += [[self.tests[i], phi]]
            # solution = H
            self.sol_pack += [[sol_c[i], phi]]
            # normal compenent of solution
            self.sol_trace_n_pack += [[sol_c[i].Trace() * n, phi]]
            self.u_trace_n_pack += [[self.trials[i].Trace() * n, phi]]
            self.v_trace_n_pack += [[self.tests[i].Trace() * n, phi]]
            # Ansatz
            self.ansatz += " + T" + str(i) + " * "+ phi.name

            i +=1

                            # curl u 1 phi                                                    # e x u * dzphi       
        self.curlu_pack = [[el[0] * el[1], el[2]] if el[0].derivname == ""              else [self.myCross(el[1], el[0]), el[2]] for el in self.tri_curlu]
        self.curlv_pack = [[el[0] * el[1], el[2]] if el[0].derivname == ""              else [self.myCross(el[1], el[0]), el[2]] for el in self.tri_curlv]
        self.curlsol_pack = [[el[0] * el[1], el[2]] if not hasattr(el[0], "derivname")  else [self.myCross(el[1], el[0]), el[2]] for el in self.tri_curlsol]

        # self.curlu_trace_pack = [[el[0].Trace() * el[1], el[2]] if el[0].derivname == "" else [self.myCross(el[1], el[0].Trace()), el[2]] for el in self.tri_curlu]
        # self.curlv_trace_pack = [[el[0].Trace() * el[1], el[2]] if el[0].derivname == "" else [self.myCross(el[1], el[0].Trace()), el[2]] for el in self.tri_curlv]



        # solution components
        self.sol_comp = [self.sol_pack[i][0] * self.sol_pack[i][1] for i in range(len(self.sol_pack))]
        self.curlsol_comp = [self.curlsol_pack[i][0] * self.curlsol_pack[i][1] for i in range(len(self.curlsol_pack))]



    def fillContainers_2D_eddy_inPlane(self):
        # T in H1 -> J = dy T ex - dx T ey
        raise ValueError("not implemented yet")

    
    def generateCouplingMatrix(self, lambdaFe, lambdaAir, force_num_int=False, **kwargs):
        super().generateCouplingMatrix(self.curlu_pack, self.curlv_pack, lambdaFe, lambdaAir, force_num_int=force_num_int, **kwargs)
    
    def checkCouplingMatrxiSymmetric(self, eps=1e-5):
        return super().checkCouplingMatrxiSymmetric(self.curlu_pack, self.curlv_pack, eps=eps)
    
    def setElementCouplingMatrix(self, r, c, val):
        super().setElementCouplingMatrix(self.curlu_pack, self.curlv_pack, r, c, val)
    
    def getElementCouplingMatrix(self, r, c):
        return super().getElementCouplingMatrix(self.curlu_pack, self.curlv_pack, r, c)
    
    def printCouplingMatrix(self, sparsity=False, round_digits=3):
        return super().printCouplingMatrix(self.curlu_pack, self.curlv_pack, sparsity=sparsity, round_digits=round_digits)


            



class cl_gradgradMS(cl_MS):
    def __init__(self, orderPhi, sol, **kwargs):
        # ------------------------------------------------------------------------------
        # --- kwargs
        # ------------------------------------------------------------------------------
        self.secondOrder = False
        if "secondOrder" in kwargs.keys():
            self.secondOrder = kwargs["secondOrder"]
            kwargs.pop("secondOrder")
        # ------------------------------------------------------------------------------
        # --- end kwargs
        # ------------------------------------------------------------------------------
        super().__init__(orderPhi, sol, **kwargs)

        
    


        self.tri_gradu = []
        self.tri_gradv = []      

        self.gradsol_comp = []
        self.gradsol_pack = []

        if self.secondOrder:
            self.tri_hesse_u = []
            self.tri_hesse_v = []

            # def jumpdn(u,v):
            #     return n*(grad(u)-v)
            def hesse(v):
                return v.Operator("hesse")
            # def hessenn(v):
            #     return InnerProduct(n, hesse(v)*n)

        self.fillContainers()
        self.gradgrad_matrix = None

    def fillContainers(self):
        n = specialcf.normal(self.dim)
        sol_c = self.sol.components
        i = self.istart

        # n = CF((0, 1))
        # clPhi0 = cl_Phi(0, 0, material=".*", modelHalfAir=cl_Phi.modelHalfAir) 
        phi0 = self.phi0
        if self.addPhi0Outer:

            # phi 0 outer
            self.tri_gradu += [[grad(self.trials[i]), -1, phi0]] # ok
            self.tri_gradv += [[grad(self.tests[i]), -1, phi0]]

            if self.secondOrder:
                self.tri_hesse_u += [[hesse(self.trials[i]), CF(1), phi0]]
                self.tri_hesse_v += [[hesse(self.tests[i]), CF(1), phi0]]

            # components
            self.u_pack += [[self.trials[i], phi0]]
            self.v_pack += [[self.tests[i], phi0]]

            #solution
            self.sol_pack += [[sol_c[i], phi0]]
            self.gradsol_pack += [[-grad(sol_c[i]), phi0]]

            self.ansatz += "-grad(u0_outer)"
            i+= 1

        
        # Phi: grad(Phi*phi) = grad(Phi) * phi + Phi * dy phi ey
        for phi_i  in self.orderPhi:
            phi = phi_i.phi
            dxphi = phi_i.dzphi

            if phi_i.useAbsolutes and type(dxphi) != pyPhiZero:
                # dont add order 0 diff because dxphi = 0
                self.tri_gradu += [[self.trials[i], -self.e,  dxphi]] # ok
                self.tri_gradv += [[self.tests[i], -self.e,  dxphi]]

                self.gradsol_pack += [[-sol_c[i]*self.e, dxphi]]

                if self.secondOrder:
                    self.tri_hesse_u += [[self.trials[i], OuterProduct(self.e, self.e), dxphi.getDiff()], [grad(self.trials[i]), self.e, dxphi.getDiff()]]
                    self.tri_hesse_v += [[self.tests[i], OuterProduct(self.e, self.e), dxphi.getDiff()], [grad(self.tests[i]), self.e, dxphi.getDiff()]]

            if phi_i.useGradients:
                self.tri_gradu += [[grad(self.trials[i]), -1,  phi]] # ok
                self.tri_gradv += [[grad(self.tests[i]), -1, phi]]

                self.gradsol_pack += [[-grad(sol_c[i]), phi]]

                if self.secondOrder:
                    self.tri_hesse_u += [[hesse(self.trials[i]), CF(1),  phi], [grad(self.trials[i]), self.e, dxphi]]
                    self.tri_hesse_v += [[hesse(self.tests[i]), CF(1), phi], [grad(self.tests[i]), self.e, dxphi]]



            self.u_pack += [[self.trials[i], phi]]
            self.v_pack += [[self.tests[i], phi]]

            #solution
            self.sol_pack += [[sol_c[i], phi]]
            
            
            if phi_i.useGradients and phi_i.useAbsolutes:
                self.ansatz += " - grad(u" + str(i) + " * "+ phi.name + ")"
            elif phi_i.useAbsolutes:
                self.ansatz += " - u" + str(i) + " * grad("+ phi.name + ")"
            elif phi_i.useGradients:
                self.ansatz += " - grad(u" + str(i) + ") * "+ phi.name 

        
            i +=1   

        # solution components
        self.sol_comp = [self.sol_pack[i][0] * self.sol_pack[i][1] for i in range(len(self.sol_pack))]
        self.gradsol_comp = [self.gradsol_pack[i][0] * self.gradsol_pack[i][1] for i in range(len(self.gradsol_pack))]

        # grad components
        self.gradu_pack = [[el[0] * el[1], el[2]] for el in self.tri_gradu]
        self.gradv_pack = [[el[0] * el[1], el[2]] for el in self.tri_gradv]

        # normal trace components
        self.gradu_trace_n_pack = [[InnerProduct(el[0].Trace() * el[1], n), el[2]] for el in self.tri_gradu]
        self.gradv_trace_n_pack = [[InnerProduct(el[0].Trace() * el[1], n), el[2]] for el in self.tri_gradv]

        if self.secondOrder:
            # normal normal components
            self.hesse_u_trace_nn = []
            for el in self.tri_hesse_u:
                if el[1].dim == 1:
                    self.hesse_u_trace_nn += [[InnerProduct(n, el[0].Trace()* el[1] * n), el[2]] for el in self.tri_hesse_u]
                
                elif el[1].dim == 4:
                    self.hesse_u_trace_nn += [[InnerProduct(n, el[0].Trace()* el[1] * n), el[2]] for el in self.tri_hesse_u]
                else:
                    a = InnerProduct(n, OuterProduct(el[0].Trace(), el[1]) * n)
                    self.hesse_u_trace_nn += [[a, el[2]] for el in self.tri_hesse_u]

            
            self.hesse_v_trace_nn = []
            for el in self.tri_hesse_v:
                if el[1].dim == 1:
                    self.hesse_v_trace_nn += [[InnerProduct(n, el[0].Trace()* el[1] * n), el[2]] for el in self.tri_hesse_v]
                elif el[1].dim == 4:
                    self.hesse_v_trace_nn += [[InnerProduct(n, el[0].Trace()* el[1] * n), el[2]] for el in self.tri_hesse_v]
                else:
                    a = InnerProduct(n, OuterProduct(el[0].Trace(), el[1]) * n)
                    self.hesse_v_trace_nn += [[a, el[2]] for el in self.tri_hesse_v]



    
    def generateCouplingMatrix(self, lambdaFe, lambdaAir, force_num_int=False, **kwargs):
        super().generateCouplingMatrix(self.gradu_pack, self.gradv_pack, lambdaFe, lambdaAir, force_num_int=force_num_int, **kwargs)
    
    def checkCouplingMatrxiSymmetric(self, eps=1e-5):
        return super().checkCouplingMatrxiSymmetric(self.gradu_pack, self.gradv_pack, eps=eps)
    
    def setElementCouplingMatrix(self, r, c, val):
        super().setElementCouplingMatrix(self.gradu_pack, self.gradv_pack, r, c, val)
    
    def getElementCouplingMatrix(self, r, c):
        return super().getElementCouplingMatrix(self.gradu_pack, self.gradv_pack, r, c)
    
    def printCouplingMatrix(self, sparsity = False, round_digits=3):
        return super().printCouplingMatrix(self.gradu_pack, self.gradv_pack, sparsity=sparsity, round_digits=round_digits)

    def generateGradGradMatrix(self):
        if type(self.coupling_matrix) == type(None):
            print("generate coupling matrix first!")
            return None

        n = sum([self.tri_gradu[i][0].dim for i in range(len(self.tri_gradu))])


        self.gradgrad_matrix = np.zeros([n, n])

        n = 1
        r = 0
        for i in range(len(self.gradu_pack)):    
            phi_i = self.gradu_pack[i][1]         
            for k in range(self.tri_gradu[i][0].dim):  
                c = 0
                for j in range(len(self.gradv_pack)):
                    phi_j = self.gradv_pack[j][1]
                    for l in range(self.tri_gradv[j][0].dim):
                        # grad u grad v
                        if self.tri_gradu[i][0].derivname == "" and self.tri_gradv[j][0].derivname == "":
                            if k == l  :
                                self.gradgrad_matrix[r][c] = self.coupling_matrix[phi_i][phi_j]#__phiXphi__(self.tri_gradu[i][2], self.tri_gradv[j][2], lambdaFe, lambda0, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, **kwargs)                              
                        # grad u v
                        elif self.tri_gradu[i][0].derivname == "" and self.tri_gradv[j][0].derivname != "":
                            if k == self.tri_gradu[i][2].orientation:
                                self.gradgrad_matrix[r][c] = self.coupling_matrix[phi_i][phi_j]#__phiXphi__(self.tri_gradu[i][2], self.tri_gradv[j][2], lambdaFe, lambda0, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, **kwargs)                              

                        # u grad v
                        elif self.tri_gradu[i][0].derivname != "" and self.tri_gradv[j][0].derivname == "":
                              if l == self.tri_gradv[j][2].orientation:
                                self.gradgrad_matrix[r][c] = self.coupling_matrix[phi_i][phi_j]#__phiXphi__(self.tri_gradu[i][2], self.tri_gradv[j][2], lambdaFe, lambda0, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, **kwargs)                              
                        # u v
                        else:    
                            self.gradgrad_matrix[r][c] = self.coupling_matrix[phi_i][phi_j]#__phiXphi__(self.tri_gradu[i][2], self.tri_gradv[j][2], lambdaFe, lambda0, cl_Phi.dFe, cl_Phi.d0, force_num_int=force_num_int, **kwargs)                              

                        n += 1    
                        c += 1
                r += 1
 
        return self.gradgrad_matrix

    def printGradGradMatrix(self):
        if type(self.gradgrad_matrix) == type(None):
            self.generateGradGradMatrix()
            
        from tabulate import tabulate
        ret = [[""]]

        orientations = ["x", "y", "z"]
        
        
        for j in range(len(self.gradv_pack)):
            FeAirSuffix = ("_Fe" if self.tri_gradv[j][2].inAir==False else "") + ("_ins" if self.tri_gradv[j][2].inIron==False else "")
            for l in range(self.tri_gradv[j][0].dim):
                if self.tri_gradv[j][0].derivname ==  "":
                    ret[0] += ["d" + orientations[l] + "_v_" + str(self.tri_gradv[j][2].order)+ FeAirSuffix]
                else:
                    ret[0] += ["v_" + str(self.tri_gradv[j][2].order) + FeAirSuffix]


        r = 0
        
        for i in range(len(self.gradu_pack)):
            FeAirSuffix = ("_Fe" if self.tri_gradu[i][2].inAir==False else "") + ("_ins" if self.tri_gradu[i][2].inIron==False else "")
            for k in range(self.tri_gradu[i][0].dim):
                # that gradients
                if self.tri_gradu[i][0].derivname == "":
                        ret += [["d" + orientations[k] + "_u_" + str(self.tri_gradu[i][2].order) + FeAirSuffix]]
                # thats absolutes
                else:
                    ret += [["u_" + str(self.tri_gradu[i][2].order) + FeAirSuffix ]]
                
                c = 0
                for j in range(len(self.gradv_pack)):
                    for l in range(self.tri_gradv[j][0].dim):
                        ret[-1] += [self.gradgrad_matrix[r][c]]

                        c += 1
                r += 1

        print(tabulate(ret, headers="firstrow"))
        return tabulate(ret, headers="firstrow")
    

    def addCurlCurlMS(self, curlcurlMS : cl_curlcurlMS):
        self.gradu_pack += curlcurlMS.u_pack
        self.gradv_pack += curlcurlMS.v_pack

        
        self.gradu_trace_n_pack += curlcurlMS.u_trace_n_pack
        self.gradv_trace_n_pack += curlcurlMS.v_trace_n_pack
        

        # update solution
        self.gradsol_pack += curlcurlMS.sol_pack
        self.gradsol_comp = [self.gradsol_pack[i][0] * self.gradsol_pack[i][1] for i in range(len(self.gradsol_pack))]

        self.ansatz += curlcurlMS.ansatz
        pass



        


        

        



        

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

    if isinstance(phi, pyLobatto) or isinstance(phi, pyPhiConst):
        ret_code += "phi"
    elif isinstance(phi, pydxLobatto) or isinstance(phi, pyPhiFunction):
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

def num_int(a, b,  fun1, fun2=lambda x: 1, int_order=5, int_order_max=50, eps=1e-15):
    ret_old = 0
    ret = 0
    int_order0 = int_order
    
    for i in range(int_order, int_order_max):
    # while (int_order <= int_order0+1 or (ret_old != 0 and abs((ret - ret_old))/abs(ret_old)> 1e-5)) and int_order <= int_order_max:
        ret_old  = ret
        xi, wi = np.polynomial.legendre.leggauss(i)
        ret = (b-a)/2 * sum([w * fun1((b-a)/2*x+(a+b)/2) * fun2((b-a)/2*x+(a+b)/2) for x, w in zip(xi, wi)])     


        if i > int_order and (abs(ret- ret_old) < eps):
            # print(i, "done")
            break

    # from numpy import trapz, linspace
    # ret_old = 0
    # ret = 0
    
    # int_order = 20
    # N = int_order
    # for i in range(int_order, int_order_max):
    #     ret_old = ret
    #     xi = linspace(a, b, i)

    #     y_fun1 = np.array([fun1(x) for x in xi])
    #     y_fun2 = np.array([fun2(x) for x in xi])
    #     ret = trapz(x=xi, y= y_fun1*y_fun2)

    #     if i > int_order and (abs(ret- ret_old)/abs(ret) < eps):
    #         # print(i, "done")

    #         break

    return round(ret   , 11)

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
    # elif hasattr(phi1, "modelHalfAir") and hasattr(phi1, "modelHalfAir"):
    #     modelHalfAir = phi1.modelHalfAir or phi2.modelHalfAir
    else:
        modelHalfAir = cl_Phi.modelHalfAir

    # print("modelHalfAir", modelHalfAir)
    # print("force_full_Phi", force_full_Phi)
    # print("force_num_int", force_num_int)




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


    if (phi1.inAir == False and phi1.inIron == False) or (phi2.inAir == False and phi2.inIron == False):
        return 0
    
    # if True:
    if code2 == False or code1 == False or force_num_int or force_full_Phi:
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
        if phi1.fullPhi == False and phi2.fullPhi == False and force_full_Phi==False and modelHalfAir and phi1.modelHalfAir == True and phi2.modelHalfAir == True:
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
                        
                    # print("tmp1", n, ret)
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
        
        return round(ret, 15)

    
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
                              
                ret += InnerProduct((lambda_bar  * trialList[i][0]),testList[j][0]) 

                # ret += (lambda_bar  * trialList[i][0]) * testList[j][0]
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
    ret = True
    for i in order:
        # check boundaries
        bnds = i.dirichlet.split("|")
        for bnd in bnds:
            if bnd not in mesh.GetBoundaries() and bnd !="":
                print(bnd)
                ret = False
                input(f"boundary name {bnd} unknown")
                

        # check domains
        mats = i.material.split("|")
        for mat in mats:
            if mat not in mesh.GetMaterials() and mat != ".*":
                print(mat)
                ret = False
                input(f"material name {mat} unkown >>")
    return ret


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




    
