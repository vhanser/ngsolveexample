#from ngsolve.internal import *


def assert_almost(a, b, eps = 1e-3, message=""):
    if a < 1e-10 and b < 1e-10 :
        return
    # assertion used to be eps*100
    assert abs((a-b)/a) < eps, message+f"\n a:\t{a}\n b:\t{b}\n eps:\t{eps}\n result:\t{abs((a-b)/a)*100}"
    

def rotationCF(mesh, xshift, yshift, angle, dimension=2, domains=["rotor"]):
    from ngsolve import CF, cos, sin, x, y, z, Id
    
    if dimension == 2:
        rotmat = CF( (cos(angle), -sin(angle), sin(angle), cos(angle)), dims=(2,2))
        shift = CF( (xshift, yshift) )
        pos = CF( (x,y) )
        trans_CF = (rotmat - Id(2))*(pos - shift) - shift
        return CF( [trans_CF if mat in domains else (0, 0)for mat in mesh.GetMaterials()])



    else:

    
        rotmat = CF( (cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1), dims=(3,3))
        shift = CF( (xshift, yshift, 0) )
        pos = CF( (x,y, z) )
        trans_CF = (rotmat - Id(3))*(pos - shift) - shift
        return CF( [trans_CF if mat in domains else (0, 0, 0) for mat in mesh.GetMaterials()])


class TextColor:
    PURPLE = '\033[1;35;48m'
    CYAN = '\033[1;36;48m'
    BOLD = '\033[1;37;48m'
    BLUE = '\033[1;34;48m'
    GREEN = '\033[1;32;48m'
    YELLOW = '\033[1;33;48m'
    RED = '\033[1;31;48m'
    BLACK = '\033[1;30;48m'
    UNDERLINE = '\033[4;37;48m'
    END = '\033[1;37;0m'

def colorprint(text, color=TextColor.END):
    print(color + text + TextColor.END)


class L2Draw:
    drawFunc = None
    orientation = 0
    def __init__(self, ain, mesh, name="test", diff=False, order=2, **kwargs):
        """ Map a and b to L2 Space or VectorL2

        :param ain: CF a
        :param mesh: mesh to Draw on
        :param name: name to be used for Draw, defaults to "test"
        :param diff: True: draw a - b, False: darw a on one sied and b on other, defaults to False
        :param order: order of L2 space, defaults to 2
        :param orientation: orientation of a and b jump, defaults to 0
        :raises ValueError: a and b must have same dimension
        """
        from ngsolve import L2, VectorL2, IfPos, x, y, z, TaskManager
        
        self.drawFunc = L2Draw.drawFunc
        if "drawFunc" in kwargs.keys():
            self.drawFunc = kwargs["drawFunc"]
            kwargs.pop("drawFunc")

        if self.drawFunc == None:
            from ngsolve import Draw
            self.drawFunc = Draw

        self.orientation = L2Draw.orientation
        if "orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]
            kwargs.pop("orientation")

        if type(self.drawFunc) == type(None):
            raise ValueError("drawFunc not Set. Either set L2Draw.drawFunc = Draw, or use the function kward drawFunc=Draw")

        
        if hasattr(ain, "__iter__"):
            a = ain[0]
            b = ain[1]
        else:
            a = ain
            b = ain
        
        if a.dim != b.dim:
            raise ValueError("a an b have to have the same dimension")
        
        if a.is_complex != b.is_complex:
            raise ValueError("a an b have to have the same is_complex")
        



        if a.dim == 1:
            spaceFoo = L2
        else:
            spaceFoo = VectorL2
        tmp = GridFunction(spaceFoo(mesh, order=order, complex=a.is_complex))
        with TaskManager():
            if not diff:
                tmp.Set(IfPos((x, y, z)[self.orientation], b, a))
            else:
                tmp.Set(a- b)
        self.a = a
        self.b = b
        self.mesh = mesh
        self.name = name
        self.CF = tmp
        self.scene = self.drawFunc(tmp, mesh, name, **kwargs)

def getConvexConcavePoint(H_KL, B_KL, order=1, H_KL_pos=None, realDiff=False):
    from cempy import basicClasses

    ind = (H_KL >= 0)
    H_KL = H_KL[ind]
    B_KL = B_KL[ind]
    H_KL[0] = 0
    B_KL[0] = 0

    KL = basicClasses.KL(H_KL, B_KL, order=order)
    
    if type(H_KL_pos) == type(None):
        H_KL_pos = getLines(H_KL, 1000)
    
    B_KL_pos = [KL(x) for x in H_KL_pos]
    if realDiff:
        dB = [KL.Differentiate(x) for x in H_KL_pos]
    else:
        dB = B_KL_pos/(H_KL_pos+1e-15)
    
    ind = np.argmax(dB)
    H_convexconcav = H_KL_pos[ind]
    B_convexconcav = B_KL_pos[ind]

    return H_convexconcav, B_convexconcav


def boundedCF(val, upper_limit, lower_limit=0):
    from ngsolve import IfPos, Norm

    return IfPos(val - upper_limit, upper_limit, IfPos(-val+lower_limit, lower_limit, val))


def getLinearFakePreisach(mesh, intrule, H, ev, dist, field="H", mask = None):
    from ngsolve import CF, L2, MatrixValued, VectorValued
    class classB(GridFunction):
        def __init__(self, fes, mu, H):
            super().__init__(fes)
            self.mu = mu
            self.H = H
            self.fes = fes
            
            super().Set(self.mu * H)

        def copy(self):
            tmp = GridFunction(self.fes)
            tmp.vec.data = self.vec
            return tmp

        def Set(self, x):
            super().Set(x)

    class classH(classB):
        def __init__(self, fes, H):
            super().__init__(fes, 1, H)





    class FakeLinearPreisach:
        def __init__(self, mesh, intrule, H, ev, dist, field="H"):
            self.name="FakePreisach"
            self.usePerfectDemag = True
            self.usePreviousForMuDiffOnly = False
            self.ev = ev
            self.mu = self.ev.maxB/self.ev.maxH

            self.H = H
            if H.dim > 1:
                self.fesB = VectorValued(L2(mesh))
            else:
                self.fesB = L2(mesh)
            self.B = classB(self.fesB, self.mu, H)
            self.Hout = classH(self.fesB, H)
            self.fesMu = MatrixValued(L2(mesh))
            self.mesh = mesh

        def Update(self):
            self.B.Set(self.mu * self.H)
            

        def UpdatePast(self):
            self.B.Set(self.mu * self.H)
            


        
          
          
        def GetMatDiff(self):
            matDiff = GridFunction(self.fesMu)
            matDiff.Set(diagCF(self.ev.maxB/self.ev.maxH, self.H.dim))
            # return diagCF(self.ev.maxB/self.ev.maxH, self.H.dim)
            return matDiff
        def GetMat(self):
            return self.GetMatDiff()

        def GetMu(self):
            return self.GetMat()
        def GetMuDiff(self):
            return self.GetMatDiff()

        def GetB(self):
            # return self.ev.maxB/self.ev.maxH * self.H
            
            return self.B
        
        def GetH(self):
            # return self.ev.maxB/self.ev.maxH * self.H
            return self.Hout

        def Demagnetise(*kwargs):
            None
        def GetEnergyDensity(*kwargs):
            return 0
        def GetHystereticLossesLoop(*kwargs):
            return 0

        def UpdateHystereticLossesLoop(*kwargs):
            None 

    return FakeLinearPreisach(mesh, intrule, H, ev, dist, field="H")


class selfFillingList:
    # [[string key, int index, int appearence]]
    def __init__(self, startvalue=1):
        self.data = []
        self.current_index = startvalue
        
        
    def __getFullRow__(self, key):
        
        isIt = [key == line[0] for line in self.data]
        isIn = sum(isIt)
        if len(self.data) == 0 or not isIn:
            self.data += [[key, self.current_index, 1]]
            self.current_index += 1 
            ind = -1
        else:
            ind = isIt.index(True)
            self.data[ind][2] += 1
            
        return self.data[ind]
        
    def __getitem__(self, key):
        return self.__getFullRow__(key)[1]
    def __setitem__(self, key, val):
        self.__getFullRow__(key)[1] = val
    
    def isused(self, key):
        return self.__getFullRow__(key)[2] > 0

        
    def keys(self):
        return [line[0] for line in self.data]
    def values(self):
        return [line[1] for line in self.data]
    def numberofcalls(self):
        return [line[2] for line in self.data]

    def __setCurrentIndex__(self, value):
        self.current_index = value

    def __str__(self):
        s = ""
        for line in self.data:
            s += "\"" + str(line[0]) + "\"\t" + str(line[1]) + ", \t" + str(line[2]) +"\n\r"

        return s

def __sep__(x, N, scale=1):
    import numpy as np
    if hasattr(x, "__iter__"):
        x = sum(x)
    
    if isinstance(N, int):
        return list(np.ones(N)*1/N*x*scale)
    elif isinstance(N, bool):
        print("auto-replacee bool value to 1")
        input()
        return list(np.ones(N)*x*scale)
    else:
        return list(x* np.array(N)/np.sum(N)*scale)
        
def halfArray(a, secondHalf=True, inPlace=False):
    import numpy as np
    b = np.array(a, dtype="float")

    if len(b) == 1:
        b[0] /= 2
        if not inPlace:
            return list(b)
        else:
            a[0] /= 2


    if secondHalf:
        b = b[int(len(b)/2):]
    else:
        b = b[:int((len(b)+1)/2)]


    if len(a) % 2== 1 and secondHalf:
        b[0] /= 2 
    elif len(a) % 2== 1 and not secondHalf:
        b[-1] /= 2 

    if sum(a) != 0 and abs((sum(a)/2 - sum(b))/sum(b)) > 1e-5:
        print(a)
        print(b)

        myBreak(locals(), globals(), file=__file__.split('/')[-1])
        raise ValueError(f"sum(a)/2 not sum(b) in halfArray  {sum(a)/2} {sum(b)}")


    if not inPlace:
        if isinstance(b, np.ndarray):
            b = list(b)
        return b
    
    elif isinstance(a, list):
        a.clear()
        a += list(b)
    elif isinstance(a, np.ndarray):
        a.resize(len(b), refcheck=False)
        a[:] = b[:]
      
def loadView(N=1, filename="./viewOpt.txt"):
    try:
        for i in range(N):
            exec(open(filename).read())
    except Exception as e:
        print(str(e))
            
    
def addExec(var_loc, var_glo={}, filename="./addExec.txt"):
    try:
        exec(open(filename).read(), var_glo, var_loc)
    except Exception as e:
        print(str(e))         
    

def myBreak(var_loc, var_glo={}, file="breakpoint"):
    
    from inspect import currentframe
    cf = currentframe().f_back.f_lineno
    return cmdInput(var_loc, var_glo, file+"::"+str(cf)+" >> ")

    
    
def cmdInput(var_loc, var_glo={}, text="input your command:\n>> "):
    try:
        open('.inputrc', 'a').close()       # manually create the necesasry file
    except Exception as e2:
        print(str(e1) +"\n" + str(e2))
    import readline 
    import rlcompleter    

    #usage : locals, globals = cmdInput(locals(), globals())
    vars = var_glo
    vars.update(var_loc)
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")

    

    if 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    
    cmd = "start"
    while  cmd != "" and cmd != "exit" and cmd != "quit": 
        cmd = input(text)
        
        if cmd == "exit" or cmd=="exit()":
            exit()
        text = ">> "
        try:
            returnvalue = eval(cmd, var_glo, var_loc)
            print(returnvalue)
        except Exception as e1:
            try:
                exec(cmd, var_glo, var_loc)
            except Exception as e2:
                print(str(e1) +"\n" + str(e2))
                
    return cmd, var_loc, var_glo
                

def evalOnLine(myCF, mesh, pnt1, pnt2=[0, 0, 0], N=100, plot=False, show=True, complex="real", clear=True, yoff=0, x_start_fig = 0, title="measurement", fig=None, **kwargs):
    """ evaluate myCF in the mesh from pnt1 to pnt2 in N steps

    :param myCF: CF
    :param mesh: mesh
    :param pnt1: point1 (from)
    :param pnt2: point2 (to) , defaults to [0, 0, 0]
    :param N: number of evaluation points or evalpoints in [0, |pnt2 - pnt1|], defaults to 100
    :param plot: plot the results, defaults to False
    :param complex: use real/imag/norm part , defaults to "real"
    :param clear: clear figure, defaults to True
    :param yoff: y offset in figure, defaults to 0
    :param title: title of fiugre , defaults to "measurement"
    :return: xi, yi, fig
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if pnt2 == [0, 0, 0]:
        pnt2 = pnt1
        pnt1 = [0, 0, 0]

    pnt1 = np.array(pnt1)
    pnt2 = np.array(pnt2)

    v = pnt2 - pnt1
    normv = np.linalg.norm(v)

    if normv == 0:
        return [0]
    ev = v/normv

    if isinstance(N, int):
        xi = np.linspace(0, normv, N)    
    else:
        xi = N


    if complex == "imag":
        yi = [myCF.imag(mesh(pnt1[0] + ev[0]*x, pnt1[1] + ev[1]*x, pnt1[2] + ev[2]*x)) for x in xi]
    elif complex == "real":
        yi = [myCF.real(mesh(pnt1[0] + ev[0]*x, pnt1[1] + ev[1]*x, pnt1[2] + ev[2]*x)) for x in xi]
    else:
        # yi = [((myCF.imag**2+myCF.real**2)**0.5)(mesh(pnt1[0] + ev[0]*x, pnt1[1] + ev[1]*x, pnt1[2] + ev[2]*x)) for x in xi]
        yi = [((myCF.imag**2+myCF.real**2)**0.5)(mesh(pnt1[0] + ev[0]*x, pnt1[1] + ev[1]*x)) for x in xi]


    fig = None
    if plot != False:
        fig = plt.figure(plot)

        if clear:
            plt.clf()

        
    
        plt.plot(xi  + x_start_fig,np.array(yi) + yoff, **kwargs)
        plt.title(title)
        if len(np.shape(yi)) > 1:
            plt.legend([str(i+1) for i in range(np.shape(yi)[1])])

        if show==True:
            plt.pause(0.1)

    return xi, yi, fig








# get a step function for reduced PreisachModels
# var.. x, y, or z of ngsolve
def getMappingCF(var, x0, steps, symmetric=True):
    from ngsolve import IfPos
    CF = 0
    x = x0
    for i in range(len(steps)):
        if not symmetric or i < len(steps)/2:
            CF += IfPos(var - x, 1, 0)

        elif i> len(steps)/2:
            CF -= IfPos(var - x, 1, 0)
        x+=steps[i]

    # last element
    if not symmetric:
        CF += IfPos(var - x, 1, 0)
    else:
        CF -= IfPos(var - x, 1, 0)
    return CF

# colour the boundaries of a mesh. (Set the according boundarie to a value 2)
# use :
# [drawBnd(mesh, x) for x in ["bottom", "right", "top", "left", "ibot", "itop", "interface", "natural", "iright", "ileft"]]
# to colour all boundaries in a mesh


def drawDomainsAll(mesh, drawFunc=None, block=True):
    drawDomains(mesh, name=set(mesh.GetMaterials()), drawFunc=drawFunc, block=block) 
def drawDomains(mesh, name=None, drawFunc=None, block=True):
    from ngsolve import CF, Integrate

    if drawFunc == None:
        from ngsolve import Draw
        drawFunc = Draw

    if name==None:
        vals = dict(enumerate(mesh.GetMaterials()))
        vals = {v: k for k, v in vals.items()}
        drawFunc(mesh.MaterialCF(vals), mesh, "domains")
    else:
        if not hasattr(name, "__iter__") or type(name) == str:
            name = [name]
        
        for n in name:

            domCF = mesh.MaterialCF({n:1}, default=0)
            print(f"---- {n} ----- {Integrate(domCF,mesh)} ")
            drawFunc(domCF, mesh, "domains")
            
            if block:
                cmdInput(locals(), globals())
            
def drawBndAll(mesh, block=True, old=None, drawFunc=None, useBBnd=False):

    if old == None:
        old = True if mesh.dim <= 2 else False
    [drawBnd(mesh, x, block, old, drawFunc=drawFunc, useBBnd=useBBnd) for x in set(mesh.GetBoundaries() if not useBBnd else mesh.GetBBoundaries())]

def drawBnd(mesh, name="bottom|right|top|left|ibot|itop|interface|ileft|iright|natural", block=False, old=None, drawFunc=None, useBBnd=False):
    from ngsolve import CF, H1, GridFunction, BND, BBND, Integrate

    if old == None:
        old = True if mesh.dim <= 2 else False
    if drawFunc == None:
        from ngsolve import Draw
        drawFunc = Draw


    val = {bnd:1 for bnd in name.split("|")}


    if old:
        if useBBnd:
            fes = H1(mesh, dirichlet_bbnd=name)
            sol = GridFunction(fes, "bbnd")
            sol.Set(CF([val[bbnd] if bbnd in val.keys() else 0 for bbnd in mesh.GetBBoundaries()]), VOL_or_BND=BBND)
        else:
            fes = H1(mesh, dirichlet=name)
            sol = GridFunction(fes, "bnd")
            sol.Set(CF([val[bnd] if bnd in val.keys() else 0 for bnd in mesh.GetBoundaries()]), VOL_or_BND=BND)
        drawFunc(sol)
        print("-----", name, sum(sol.vec))

    else:
        bnd_CF = CF([val[bnd] if bnd in val.keys() else 0 for bnd in (mesh.GetBoundaries() if not useBBnd else mesh.GetBBoundaries())])
        drawFunc(bnd_CF, mesh, "bnd", draw_vol=False)
        print("-----", name, Integrate(bnd_CF, mesh, VOL_or_BND=(BND if not useBBnd else BBND)))

    if block:
        cmdInput(locals(), globals())


# for preisach testing
def getCircleExcitation(H0, Nperiods, NperPeriod, periodsToAmplitude):

    import numpy as np
    Hout = []

    if int(periodsToAmplitude * NperPeriod) >0:
        phii_init = np.linspace(0, periodsToAmplitude*np.pi*2, int(periodsToAmplitude * NperPeriod))
        ri_init = np.linspace(0, H0, len(phii_init))
        [Hout.append((ri_init[i] * np.cos(phii_init[i]), ri_init[i] * np.sin(phii_init[i]))) for i in range(len(ri_init))]
    else:
        phii_init = [0]

    if int((Nperiods-periodsToAmplitude)* NperPeriod) >0:
        phii = np.linspace(phii_init[-1], phii_init[-1] + np.pi*2, int((Nperiods-periodsToAmplitude)* NperPeriod))
        [Hout.append((H0 * np.cos(phii[i]), H0 * np.sin(phii[i]))) for i in range(len(phii))]

    return Hout

def getLines(val, N):
    import numpy as np
    ret = np.array([])

    for i in range(0, len(val)-1):
        ret = np.hstack([ret, np.linspace(val[i], val[i+1], N, endpoint=(i == len(val) - 2))])


    return ret
		

def plotDist(dist, fig = 0, weights = False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    o = dist.getPoints()

    sp = np.zeros([len(o), 4])
    for i in range(len(o)):
        sp[i, :] = np.array([o[i].first, o[i].second, o[i].third, i]) 
        if weights:
            sp[i, 0:3] *= o[i].fourth

    fig = plt.figure(fig)#, figsize=plt.figaspect(2))
    plt.clf()
    ax = plt.axes(projection='3d')
    # ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(2, 1, 2, projection='3d')

    plt.ion()
    ax.scatter( sp[:, 0], sp[:, 1], sp[:, 2])#, c=1, vmin=-1, vmax=1,s=50)

    for i in range(np.shape(sp)[0]):
        ax.text(sp[i, 0], sp[i, 1], sp[i, 2], int(sp[i, 3]))

    #pB = ax.scatter( spherePoints[0,:], y, z, s= 10, c='r')
    
    if weights:
        tmp = np.max(np.max(np.abs(sp[:, 0:3])))
        plt.xlim(-tmp,tmp)
        plt.ylim(-tmp,tmp)
        ax.set_zlim3d(-tmp, tmp)
    else:
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        ax.set_zlim3d(-1, 1)

    plt.draw()


    if dist.halfSphere:
        plt.title("halfSphere " +dist.name )        
    else:
        plt.title(dist.name)
    plt.show()
    plt.pause(0.1)

def plotSPTriag(p, ax=-1, text=True, show=False):
    """plot the preisach Plane of a preisachModel

    :param p: preisach Model
    :param ax: axis to be plot on, defaults to -1
    :param text: print input and outputvalue, defaults to True
    :param show: show figure in the end, defaults to False

    :return: axis
    """


    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    import numpy as np
    
    points = p.getPointList()
    maxH = p.getEV().maxH
    maxHin = p.maxHin


    outer = [ [-maxH, -maxH], [-maxH, maxH], [maxH, maxH], [-maxH, -maxH]]
    #perfect Demag
    perfDemag = [[-maxH, maxH], [-maxHin, maxHin]]

    inner = [[-maxH, -maxH], [-maxH, maxH], [-maxHin, maxHin]]
    for i in range(len(points)):
        inner.append([inner[-1][0], points[i].first])
        inner.append([points[i].second, points[i].first])
    inner.append([-maxH, -maxH])

    inner = np.array(inner)
    outer = np.array(outer)
    perfDemag = np.array(perfDemag)

    if ax == -1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.clear()
    ax.set_aspect('equal')
    ax.fill(inner[:, 0], inner[:, 1], edgecolor="black")    
    ax.plot(outer[:, 0], outer[:, 1], 'b')
    ax.plot(perfDemag[:, 0], perfDemag[:, 1], 'r')
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")

    if text:
        ax.text(-maxH*1/5, -1*maxH/3, "B = " + str(np.round(p.getB_interpolated(), 3)))
        ax.text(-maxH*1/5, -1.7*maxH/3, "H = " + str(np.round(p.getH_interpolated(), 3)))
    

    if show:
        plt.show()


    return ax
    


def plotPV(pV, nFig = 0, field="HB", numbers=False, label=True, file=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    B = []
    H = []

    dist = pV.getDist()
    for i in range(pV.size):
        pS = pV.getPreisachScalar(i)
        xyzW = pS.getXYZW

        xyz = np.array([xyzW.first, xyzW.second, xyzW.third])
        B.append(pS.getB_interpolated()*1/pV.getEV().maxB*xyz)
        H.append(pS.getH_interpolated()*1/pV.getEV().maxH*xyz)


    B = np.array(B)
    H = np.array(H)

    fig = plt.figure(nFig)#, figsize=plt.figaspect(2))
    plt.clf()
    if dist.dim == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.gca()

    # ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(2, 1, 2, projection='3d')

    B_val = np.array([pV.getB().first, pV.getB().second, pV.getB().third])
    H_val = np.array([pV.getH().first, pV.getH().second, pV.getH().third])
    plt.ion()

    if dist.dim == 3:
        ax.set_zlim3d(-1, 1)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    #ax.scatter( sp[0,:], sp[1,:], sp[2,:])#, c=1, vmin=-1, vmax=1,s=50)
    
    if field == "HB":
        if dist.dim==3:
            pB = ax.scatter( B[:,0], B[:,1], B[:,2], s= 10, c='r', 
                label = '         %.2lf\nB = [ %.2lf ]\n         %.2lf'%(B_val[0],B_val[1],B_val[2]))
            pH = ax.scatter( H[:,0], H[:,1], H[:,2], s= 10, c='g', 
                label = '         %.2lf\nH = [ %.2lf ]\n         %.2lf'%(H_val[0],H_val[1],H_val[2]))
        else: 
            pB = ax.scatter( B[:,0], B[:,1], s= 10, c='r', 
                label = '         %.2lf\nB = [ %.2lf ]\n         %.2lf'%(B_val[0],B_val[1],B_val[2]))
            pH = ax.scatter( H[:,0], H[:,1], s= 10, c='g', 
                label = '         %.2lf\nH = [ %.2lf ]\n         %.2lf'%(H_val[0],H_val[1],H_val[2]))
    elif field == "H":
        if dist.dim ==3:
            pH = ax.scatter( H[:,0], H[:,1], H[:,2], s= 10, c='g', 
                label = '         %.2lf\nH = [ %.2lf ]\n         %.2lf'%(H_val[0],H_val[1],H_val[2]))
        else:
            pH = ax.scatter( H[:,0], H[:,1], s= 10, c='g', 
                label = '         %.2lf\nH = [ %.2lf ]\n         %.2lf'%(H_val[0],H_val[1],H_val[2]))

    elif field == "B":
        if dist.dim==3:
            pB = ax.scatter( B[:,0], B[:,1], B[:,2], s= 10, c='r', 
                label = '         %.2lf\nB = [ %.2lf ]\n         %.2lf'%(B_val[0],B_val[1],B_val[2]))
        else:
            pB = ax.scatter( B[:,0], B[:,1], s= 10, c='r', 
            label = '         %.2lf\nB = [ %.2lf ]\n         %.2lf'%(B_val[0],B_val[1],B_val[2]))
    else:
        print("use field = H, B or HB")
        exit()


    #ax.set_title("Unit Cube")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if dist.dim==3:
        ax.set_zlabel("z")
    ax.set_title("Distribution of Scalar Preisach Models")
    #plt.colorbar()
    plt.ioff()
    plt.legend(
        scatterpoints=1,
        loc='lower left',
        ncol=2,
        fontsize=12)
    
    if numbers:
        for i in range(P.size):
            ax.text(sp[i, 0], sp[i, 1], sp[i, 2], int(sp[i, 3]))

    if dist.halfSphere:
        plt.title("halfSphere " +dist.name )        
    else:
        plt.title(dist.name)
    plt.pause(0.1)

    if file != None:
        plt.savefig(file)

    import numpy as np
    Hout = []
    
def plotEVF(ev, nFig = 0, title=None, show=True, vec= None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    if type(vec) == type(None):
        vec = ev.GetDiscInputValues()

    X, Y = np.meshgrid(vec, vec)
    #Z = np.nan * np.zeros(np.shape(X))
    Z = np.zeros(np.shape(X))
    for i in range(len(vec)):
        for j in range(len(vec) ):
            if( j > i):
                continue

            Z[i, j] = ev.GetValue(Y[i, j], X[i, j])
            if np.isnan(Z[i, j]):
                print(X[i, j], Y[i, j], "\t->\t", Z[i, j])

    fig = plt.figure(nFig)#, figsize=plt.figaspect(2))
    plt.clf()    
    ax = plt.axes(projection='3d')
    #ax = fig.add_subplot(2, 1, 2, projection='3d')

    # plt.ion()
    ax.plot_surface( X, Y, Z)#, c=1, vmin=-1, vmax=1,s=50)
    # ax.scatter( X, Y, Z)#, c=1, vmin=-1, vmax=1,s=50)


    #pB = ax.scatter( spherePoints[0,:], y, z, s= 10, c='r')
    # ax.set_zlim3d(-1, 1)
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)

    #plt.xticks(vec)
    #plt.yticks(vec)
    Z = np.flipud(Z)
    
    if type(title) == type(None):
        plt.title(ev.name + " , dim=" + str(ev.dim))
    else:
        plt.title(title)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_zlabel(r"$E(\alpha, \beta)$")


    fig =plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(Z)

    if show:
        plt.show()

def KLMinMax(H_KL, B_KL, inverse=False, diff=True, ignore_monotony=False):
    import numpy as np


    if not diff:
        H_KL = np.array(H_KL)
        B_KL = np.array(B_KL)

        ind = H_KL != 0

        mu = B_KL[ind]/H_KL[ind]

        return np.min(mu), np.max(mu)

    mu_diff_min = np.nan
    mu_diff_max = np.nan

    if not ignore_monotony:
        for i in range(len(H_KL)-1):
            if(H_KL[i+1] <= H_KL[i] or B_KL[i+1] <= B_KL[i]):
                print("KL is wrong monotony error! - myPackage.py- , set ignore_monotony=True to ignore that error")
                print(H_KL[i+1], H_KL[i])
                print(B_KL[i+1], B_KL[i])
                continue
            #            raise ValueError("invalid KL")


            if not inverse:
                mu_diff = (B_KL[i+1] - B_KL[i])/(H_KL[i+1] - H_KL[i]); 
            else:
                mu_diff = (H_KL[i+1] - H_KL[i])/(B_KL[i+1] - B_KL[i]); 
                
            if(np.isnan(mu_diff_min) or mu_diff < mu_diff_min):
                mu_diff_min = mu_diff


            if(np.isnan(mu_diff_max) or mu_diff > mu_diff_max):
                mu_diff_max = mu_diff

    return mu_diff_min, mu_diff_max

def diagCF(val=1, dim=3):
    from ngsolve import CF
    if dim == 3:
        return CF((val, 0, 0, 0, val, 0, 0, 0, val), dims=(3,3))
    if dim == 2:
        return CF((val, 0, 0, val), dims=(2,2))
    if dim == 1:
        return CF((val), dims=(1,1))


        
# ------------------------------------------------------------------------------------    
# measurement/evaluating of CF and (Integration without ngsolve) on defined area
# ------------------------------------------------------------------------------------  
                
def getScalarFromVec(vec, dim="norm"):  
    if dim == "norm":
        out = np.linalg.norm(vec)
    elif dim == "x": 
        out = vec[0]
    elif dim == "y":
        out = vec[1] 
    elif dim == "z":
        out = vec[2] 
    else:
        print("wrong parameter in getScalarFromVec()")
        exit() 
    return out

def measure_top(mesh, B_CF, variable2show="norm", granularity_xy=[2,2]):    
    return measure(mesh, B_CF, point_lb=[0, 96+39, 93.533/2+3], point_rt=[47+13, 96+39+40, 93.533/2+3], variable2show=variable2show, granularity_xy=granularity_xy)
    
def measure_front(mesh, B_CF, variable2show="norm", granularity_xy=[2,2]):
    return measure(mesh, B_CF, point_lb=[47+13, 96+39, 93.533/2+3-60], point_rt=[47+13, 96+39+40, 93.533/2+3], variable2show=variable2show, granularity_xy=granularity_xy)
    
    
def measure_back(mesh, B_CF, variable2show="norm",  granularity_xy=[2,2]):
    return measure(mesh, B_CF, point_lb=[-47-13, 96+39, 93.533/2+3-60], point_rt=[-47-13, 96+39+40, 93.533/2+3], variable2show=variable2show, granularity_xy=granularity_xy)
    
def measure(mesh, B_CF, point_lb=[0, 96+39, 93.533/2+3], point_rt=[47+13, 96+39+40, 93.533/2+3], variable2show="norm",  granularity_xy=[2,2], scale=1e-3):
    #default for topmsm
    tmp = np.array(point_lb) == np.array(point_rt)
    #check which layerpoint_lb
    if np.all(tmp == 0):
        print("not a vaild input data - check coordinates, only ortho layers are allowed")
        return -1, -1
        
    if tmp[0] == 1:
        # yz layer
        yi = np.arange(point_lb[1], point_rt[1] + granularity_xy[0], granularity_xy[0], dtype='float')    
        zi = np.arange(point_lb[2], point_rt[2] + granularity_xy[1], granularity_xy[1], dtype='float') 
        Y,Z = np.meshgrid(yi, zi)
        X = np.ones(np.shape(Y), dtype='float')*point_lb[0]
    elif tmp[1] == 1:
        # xz layer
        xi = np.arange(point_lb[0], point_rt[0] + granularity_xy[0], granularity_xy[0], dtype='float')    
        zi = np.arange(point_lb[2], point_rt[2] + granularity_xy[1], granularity_xy[1], dtype='float') 
        X,Z = np.meshgrid(xi, zi)
        Y = np.ones(np.shape(X), dtype='float')*point_lb[1]
    elif tmp[2] == 1:
        # xy layer        
        xi = np.arange(point_lb[0], point_rt[0] + granularity_xy[0], granularity_xy[0], dtype='float')    
        yi = np.arange(point_lb[1], point_rt[1] + granularity_xy[1], granularity_xy[1], dtype='float') 
        X,Y = np.meshgrid(xi, yi)
        Z = np.ones(np.shape(Y), dtype='float')*point_lb[2]
    
    B = np.zeros(np.shape(X))   
    
    X *= scale
    Y *= scale
    Z *= scale
    useAbs=0
    # evaluate
    for i in range(0, np.size(B, 0)):
        for j in range(0, np.size(B, 1)):
            try:
                if useAbs == 0:
                    val = B_CF(mesh(X[i,j], Y[i,j], Z[i,j])) # !!! abs - caused by eighth problem !!! otherwise error
                else:
                    val = B_CF(mesh(abs(X[i,j]), abs(Y[i,j]), abs(Z[i,j])))
                val = np.array(val)
                B[i, j] = getScalarFromVec(val, dim=variable2show)
            except:
                print("invalid point -> trying abs(X,Y,Z)")
                try:
                    val = B_CF(mesh(abs(X[i,j]), abs(Y[i,j]), abs(Z[i,j]))) # !!! abs - caused by eighth problem !!! otherwise error
                    useAbs=1
                    val = np.array(val)
                    B[i, j] = getScalarFromVec(val, dim=variable2show)
                except:
                    print("no, not working")
                    return -1, 1
                    


    #integrate B to Phi
    if tmp[0] == 1:
        # yz layer
        Phi_vec = np.trapz(B, x=yi*scale, axis=1)
        Phi = np.trapz(Phi_vec, x=zi*scale)
    elif tmp[1] == 1:
        # xz layer
        Phi_vec = np.trapz(B, x=xi*scale, axis=1)
        Phi = np.trapz(Phi_vec, x=zi*scale)
    elif tmp[2] == 1:
        # xy layer
        Phi_vec = np.trapz(B, x=xi*scale, axis=1)
        Phi = np.trapz(Phi_vec, x=yi*scale)

    
    return B, Phi


def getOrthoBaseR3(v1):
    
    
    import numpy as np
    #projection of v into u
    def projection(v, u):
        scal = np.inner(v, u)
        if scal != 0:
            ret = np.inner(v, u)/np.inner(u, u) * u
        else:
            ret = np.array([0, 0, 0])
        return ret
    
    v1 = np.array(v1)
    v2 = np.roll(v1, 1)
    v3 = np.roll(v2, 1)
    
    # Gram Schwidt
    u1 = v1
    u2 = v2 - projection(v2, u1)
    u3 = v3 - projection(v3, u2) - projection(v3, u1)
    

    assert np.inner(u1, u2) <= 1e-9
    assert np.inner(u2, u3) <= 1e-9
    assert np.inner(u1, u3) <= 1e-9
    
    return u1/np.linalg.norm(u1), u2/np.linalg.norm(u2), u3/np.linalg.norm(u3)
# ------------------------------------------------------------------------------------    
# MultiScale micorshape functions
# ------------------------------------------------------------------------------------  
def createPhiCF_independent(x_CF, y_CF, z_CF, size_core=[20, 20, 20], num_sheets=15, ratio_medium2total=0.9, multi_scale_combination=[1, 5]):
    from numpy import array
    from ngsolve import IfPos
    scale = 1e-3
    size_core = array(size_core)*scale

    # heights of medium and air
    height_core_medium = size_core[2]/num_sheets * ratio_medium2total
    height_core_air = size_core[2]/(num_sheets-1) * (1-ratio_medium2total)
    height_core_single = height_core_medium + height_core_air

    # total mask
    x_mask_min = IfPos(x_CF + size_core[0]/2, 1, 0)
    x_mask_max = IfPos(x_CF - size_core[0]/2, 0, 1)

    y_mask_min = IfPos(y_CF + size_core[1]/2, 1, 0)
    y_mask_max = IfPos(y_CF - size_core[1]/2, 0, 1)

    z_mask_min = IfPos(z_CF + size_core[2]/2, 1, 0)
    z_mask_max = IfPos(z_CF - size_core[2]/2, 0, 1)


    # dz
    dz_medium = 2/height_core_medium
    dz_air = -2/height_core_air

    # start point
    z_min_coordinate = -size_core[2]/2
    

    Phi_CF = 0
    dzPhi_CF = 0
    #   Medium_CF = 0
    
    for i in range(0, num_sheets):
        # for air
        z_max = IfPos(z_CF - z_min_coordinate - (i+1) * height_core_single , 0, 1)                      # air
        z_mid_max = IfPos(z_CF - z_min_coordinate - i * height_core_single - height_core_medium, 1, 0)
        # air =1 -> x_max*x_mid_max

        # for medium
        z_mid_min = IfPos(z_CF - z_min_coordinate - i * height_core_single - height_core_medium, 0, 1)    # medium
        z_min = IfPos(z_CF - z_min_coordinate - i * height_core_single, 1, 0)

        
        # medium zick zack
        Phi_CF += (dz_medium*(z_CF - z_min_coordinate - i * height_core_single) - 1)*(z_min*z_mid_min)       
        dzPhi_CF += dz_medium * (z_min*z_mid_min)    
        #   Medium_CF += x_min*x_mid_min 
        # air zickzack
        if i != num_sheets -1:
            Phi_CF += (dz_air*(z_CF - z_min_coordinate - i * height_core_single - height_core_medium) + 1)*(z_max*z_mid_max) 
            dzPhi_CF += dz_air * (z_max*z_mid_max)
    
    # mask zick zack to medium
    Phi_CF *= x_mask_min * x_mask_max * y_mask_min * y_mask_max* z_mask_min * z_mask_max
    dzPhi_CF *= x_mask_min * x_mask_max * y_mask_min * y_mask_max* z_mask_min * z_mask_max
    return Phi_CF, dzPhi_CF
    
def createPhiCF_domains(mesh, x_CF, y_CF, z_CF, size_core=[20, 20, 20], num_sheets=15, ratio_medium2total=0.9):
    scale = 1e-3

    size_core = np.array(size_core)*scale
    # heights of medium and air
    height_core_medium = size_core[2]/num_sheets * ratio_medium2total
    height_core_air = size_core[2]/(num_sheets-1) * (1-ratio_medium2total)
    height_core_single = height_core_medium + height_core_air
    
    
    # dz
    dz_medium = 2/height_core_medium
    dz_gap = -2/height_core_air
    
    # start point
    z_min_coordinate = -size_core[2]/2
    
    Phi_CF_iron = 0
    Phi_CF_gap = 0
    for i in range(0, num_sheets):
        # for air
        z_max = IfPos(z_CF - z_min_coordinate - (i+1) * height_core_single , 0, 1)                      # air
        z_mid_max = IfPos(z_CF - z_min_coordinate - i * height_core_single - height_core_medium, 1, 0)
        # air =1 -> x_max*x_mid_max

        # for medium
        z_mid_min = IfPos(z_CF - z_min_coordinate - i * height_core_single - height_core_medium, 0, 1)    # medium
        z_min = IfPos(z_CF - z_min_coordinate - i * height_core_single, 1, 0)
        
        # medium zick zack
        Phi_CF_iron += (dz_medium*(z_CF - z_min_coordinate - i * height_core_single) - 1)*(z_min*z_mid_min)       
  
        #   Medium_CF += x_min*x_mid_min 
        # air zickzack
        if i != num_sheets -1:
            Phi_CF_gap += (dz_gap*(z_CF - z_min_coordinate - i * height_core_single - height_core_medium) + 1)*(z_max*z_mid_max) 

    val = {"air":0, "iron":Phi_CF_iron, "gap":Phi_CF_gap}
    Phi_CF = CoefficientFunction([val[mat] for mat in mesh.GetMaterials()])    




    val = {"air":0, "iron":dz_medium, "gap":dz_gap}
    dzPhi_CF = CoefficientFunction([val[mat] for mat in mesh.GetMaterials()])
    
    return Phi_CF, dzPhi_CF
  
 
# 1d Draw
from ngsolve import Mesh, GridFunction
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
class Draw1d:
    def __init__(self, CF, mesh=None, name=None, N = 30, fig=None, ax=None, draw_mesh=False, new_frame = False, show=False, clear = True, **kwargs):

    	# draw a mesh Draw(mesh)
        if isinstance(CF, Mesh):
            self.mesh = CF
            self.name = "mesh"
            self.CF = None
        # draw a gridfunction  Draw(GF)
        elif isinstance(CF, GridFunction) and (mesh == None and name == None):
            self.mesh = CF.space.mesh
            self.name = CF.name
            self.CF = CF

        else: # draw a CF Draw(CF, mesh, "name")
            self.CF = CF
            self.mesh = mesh
            self.name = name


        
        assert self.mesh.dim == 1, "use this Draw class only for 1D meshes"
            
        self.pnts = np.array([i[0] for i in self.mesh.ngmesh.Points()])
        # evaluation points on ref element
        self.eval_pnts = np.linspace(0, 1, N)
        # all points in element
        self.eval_pnts[0] += 1e-3
        self.eval_pnts[-1] -= 1e-3
        self.fig = fig
        self.ax = ax
        self.draw_mesh = draw_mesh
        self.show = show
        self.clear = clear
        self.Redraw(new_frame=new_frame, **kwargs)
            
        
    def Redraw(self, new_frame=False, **kwargs ): 
        if "c" not in kwargs.keys():
            kwargs.update({"c":"b"})

        if self.fig == None:
            self.fig = plt.gcf()
            
        if self.fig == None or new_frame:
            self.fig = plt.figure(figsize=[16, 5]) 
        plt.figure(self.fig)


        if self.clear:
            if self.ax != None:
                ax = plt.subplot(self.ax[0], self.ax[1], self.ax[2])
                ax.clear()
            else:
                plt.clf()                    
        

        

        # draw mesh
        if type(self.CF)==type(None):
            # vertices
            
            # domains
            doms = [int(e[-1]) for e in self.mesh.ngmesh.Elements1D().__str__().split("\n") if len(e) > 0]  
            colourmap = cm.get_cmap('Set1', len(set(doms)))

            # [plt.plot([self.pnts[i], self.pnts[i+1]], [0, 0], c=colourmap(doms[i]-1), lw=10 ) for i in range(len(doms))]
            h = (self.pnts[-1]-self.pnts[0])/20
            [plt.fill([self.pnts[i], self.pnts[i+1], self.pnts[i+1], self.pnts[i]], [-h, -h, h, h], fc=colourmap(doms[i]-1), ec=None ) for i in range(len(doms))]
            plt.ylim([-30*h, 30*h])
            plt.yticks([])

            plt.plot(self.pnts, np.zeros(np.shape(self.pnts)), "ok", label="mesh")
            plt.title("Mesh")
            
             
            # dom_nam = self.mesh.GetMaterials()
            # ,label=dom_nam[doms[i]-1]
            # plt.legend()

        else:
            # vertices
            roots = np.zeros(len(self.eval_pnts)*(len(self.pnts)-1))
            values = np.zeros([len(roots), self.CF.dim])

            

            for i in range(len(self.pnts) - 1):
                dx = (self.pnts[i+1]-self.pnts[i])
                for j in range(len(self.eval_pnts)):
                    roots[i*len(self.eval_pnts)+j] = self.pnts[i] + self.eval_pnts[j]* dx
                    values[i*len(self.eval_pnts)+j] = self.CF(self.mesh(roots[i*len(self.eval_pnts)+j], 0, 0))

            # plot scalar
            if self.CF.dim == 1:
                
                ind = lambda i: slice(i*len(self.eval_pnts),(i+1)*len(self.eval_pnts))
                [plt.plot(roots[ind(i)], values[ind(i)], label =self.name, **kwargs) for i in range(len(self.pnts))]
                plt.title(self.name)
            # plot vec
            else:            
                plt.quiver(roots, np.zeros(np.shape(roots)), values[:, 0], values[:,1], np.linalg.norm(values, axis=1), label =self.name)
                plt.title(self.name)

            if self.draw_mesh:
                plt.plot(self.pnts, np.zeros(np.shape(self.pnts)), "ok", label="mesh")
            
        if self.show:
            plt.pause(0.1)    
