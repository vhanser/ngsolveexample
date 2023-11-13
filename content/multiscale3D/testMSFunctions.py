from netgen.occ import *
from ngsolve import *
from netgen.meshing import IdentificationType
#from MS_helper_functions import *
from netgen.webgui import Draw as DrawGeo
# from ngsolve.webgui import Draw
# Draw = lambda *args, **kwargs : None

import matplotlib.pyplot as plt

plt.ioff()

from myPackage import evalOnLine, __sep__

import numpy as np
import stepLapMeshGenerator as mg


import importlib

importlib.reload(mg)


from myPackage import myBreak
myBreak(locals(), globals(), file=__file__.split('/')[-1])
sheetsMeshCenter = mg.sheetsMeshCenter



def run():

    d = 0.5e-2
    ff = 0.7

    dFe = d*ff
    d0 = d*(1-ff)

    numSheets = 3
    numPeriods = 1

        

    specialTB1 = [1, 1, 1,1]
    specialTB1 = False
    specialTB2 = [1, 1, 1,1]
    specialTB2 = False
    domainName_insulation = "air"
    halfAir =True
    fullProblemX = True
    fullProblemY = True
    fullProblemZ = True


    x0 = __sep__(5*d, [4, 3, 2, 1, 0.1])

    x1 = __sep__(2.5*d, 2)

    x1_i = __sep__(2.5*d, 3)
    x2 = np.flipud(x0)


    y0 = __sep__(3*d, [6, 4, 2, 1.5, 0.5, 0.25])
    y1 = __sep__(3*d, [0.25, 0.5, 2, 0.5, 0.25])
    y2 = list(np.flipud(y0))

    z0=__sep__(3*d, [10, 10, 10])
    z1_iron = __sep__(dFe, [1, 3, 5, 3, 1])
    z1_air=[d0]
    z2 = list(np.flipud(z0))

    z1_iron_MS_sep = [1, 4, 4, 1]

    cMeshMS = sheetsMeshCenter("test.vol", x0, x1, x2, y0, y1, y2, z0, z1_iron, z1_air, z2, 
        x1_i=x1_i,
        numSheets=numSheets, numPeriods=1,
        multiScale=True, 
        z1_iron_MS_sep=z1_iron_MS_sep, 
        specialTB1=False, specialTB2=False, # speTB1 for seperated first sheet, speTB2 for spec domain in first sheet
        domainName_Gap = "gap",
        domainName_insulation="insulation", 
        halfAir=True, 
        specialBoundariesForGapDomain=False, 
        specialBoundariesForInnerOuter=False,
        fullProblemX=True,
        fullProblemY=True, 
        fullProblemZ=True, seperateIronDomains=False)

    cMeshRef = sheetsMeshCenter("test.vol", x0, x1, x2, y0, y1, y2, z0, z1_iron, z1_air, z2, 
        x1_i=x1_i,
        numSheets=numSheets, numPeriods=1,
        multiScale=False, 
        z1_iron_MS_sep=False, 
        specialTB1=False, specialTB2=False, # speTB1 for seperated first sheet, speTB2 for spec domain in first sheet
        domainName_Gap = "gap",
        domainName_insulation="insulation", 
        halfAir=True, 
        specialBoundariesForGapDomain=False, 
        specialBoundariesForInnerOuter=False,
        fullProblemX=True,
        fullProblemY=True, 
        fullProblemZ=True, seperateIronDomains=False)


    from myPackage import drawBndAll
    

    meshMS=cMeshMS.Mesh
    meshRef=cMeshRef.Mesh
    print(meshMS.GetMaterials())
    print(meshMS.GetBoundaries())



    # drawBndAll(meshMS)

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


    from myPackage import myBreak
    myBreak(locals(), globals(), file=__file__.split('/')[-1])

    Draw(cl_Phi(1, inIron=False).phi, meshRef, "phi1")
    cl_Phi.numSheets
    cl_Phi.getZStart()
    
    Draw(meshMS.MaterialCF({"iron":1, "gap":2,}, default=0), meshMS, "domains")
    from myPackage import myBreak
    myBreak(locals(), globals(), file=__file__.split('/')[-1])

    # Draw(IfPos(z-cl_Phi.getZStart(), 1, 0)*IfPos(z+cl_Phi.getZStart(), 0, 1), meshMS, "test")
    from myPackage import assert_almost
    
    # assert_almost(Integrate(1, meshMS, definedon=meshMS.Materials("iron")), (sum(x0)+sum(x2))*(sum(y1))*(sum(z1_iron)+sum(z1_air))*numSheets, 1e-9)
    # assert_almost(Integrate(cl_Phi(0, inAir=False).phi, meshRef), (sum(x0)+sum(x1_g)+sum(x2))*(sum(y0)+sum(y1)+sum(y2))*(sum(z1_iron))*numSheets, 1e-5)
    # next one is not good, since mehs is not defined
    # assert_almost(Integrate(cl_Phi(0, inIron=False).phi, meshMS), (sum(x0)+sum(x1_g)+sum(x2))*(sum(y0)+sum(y1)+sum(y2))*(sum(z1_air))*numSheets, 1e-5)
    # assert_almost(Integrate(cl_Phi(0).phi, meshMS), (sum(x0)+sum(x1_g)+sum(x2))*(sum(y0)+sum(y1)+sum(y2))*(sum(z1_iron) + sum(z1_air))*numSheets, 1e-5)
    

    from myPackage import myBreak
    myBreak(locals(), globals(), file=__file__.split('/')[-1])



    

if __name__ == "__main__":
    run()