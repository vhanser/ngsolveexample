from netgen.occ import *
from ngsolve import Mesh, Integrate

def assert_almost(a, b, eps, txt):
    if a == b:
        return 
    if a == 0:
        assert False, txt + f" {a}, {b}"
    assert abs((a-b)/a) <= eps, txt + f" {a}, {b}, error: {abs((a-b)/a)} > {eps}"


class mesh2DLaminates():
    def __init__(self, D, d, ff, numSheets, multiscale=False, modelHalfAir=False, domainNameHalfAir="multiscale", 
            onlySmooth=False, onlyRough=False, maxh_edges=None, maxh = None, rotated=False, fullProblemX = True, fullProblemY = True, 
            modelGap=False, gapDomainName="gap", quad_dominated=False):

        self.D = D
        self.d = d
        self.ff = ff
        self.numSheets = numSheets
        self.multiscale = multiscale
        self.modelHalfAir = modelHalfAir
        self.onlySmooth = onlySmooth
        self.onlyRough = onlyRough
        self.maxh_edges = maxh_edges
        self.maxh = maxh
        self.dFe = d/numSheets * ff
        self.d0 = d/numSheets * (1-ff)
        self.domainNameHalfAir = domainNameHalfAir

        self.rotated = rotated
        self.fullProblemX = fullProblemX
        self.fullProblemY = fullProblemY
        self.quad_dominated = quad_dominated

        if type(maxh) == type(None):
            maxh = self.d

        
        if type(maxh_edges) == type(None):
            maxh_edges = self.d0 * 2
        if not hasattr(maxh_edges, "__iter__"):
            # rough, smooth
            maxh_edges = [maxh_edges , maxh_edges ]

        print("maxh_edges", maxh_edges)

        assert not(onlyRough * onlySmooth)

        if onlySmooth:
            modelGap = False


        wp = WorkPlane()
        if onlySmooth:
            outer = wp.RectangleC(D, d).Face()
        elif onlyRough:
            outer = wp.RectangleC(d, D).Face()
        else:
            outer = wp.RectangleC(D, D).Face()

        if modelGap == False:
            gapDomainName="outer"

        left = "left"
        right = "right"
        bottom = "bottom"
        top = "top"

        if rotated:

            left = "bottom"
            right = "top"
            bottom = "right"
            top = "left"

        outer.name="outer"
        outer.edges.Min(X).name = left
        outer.edges.Max(X).name = right
        outer.edges.Min(Y).name = bottom
        outer.edges.Max(Y).name = top


        xstart = -d/2 + self.d0/2
        rect_sheets = []
        if not multiscale:
            #half insulation
            if modelHalfAir:
                wp.MoveTo(xstart - self.d0/2, -d/2)
                rect_sheets.append(wp.Rectangle(self.d0/2, d).Face())
                rect_sheets[-1].name = "insulation"
                rect_sheets[-1].col = (0.1, 0, 1, 1)
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(Y).name = bottom
                    rect_sheets[-1].edges.Max(Y).name = top

            for i in range(numSheets):



                wp.MoveTo(xstart, -d/2)
                #iron
                rect_sheets.append(wp.Rectangle(self.dFe, d).Face())
                rect_sheets[-1].name = "inner"
                rect_sheets[-1].col = (1, 0, 0, 1)
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(X).name = "i"+left
                    rect_sheets[-1].edges.Max(X).name = "i"+right
                    rect_sheets[-1].edges.Min(Y).name = bottom
                    rect_sheets[-1].edges.Max(Y).name = top
                else:
                    rect_sheets[-1].edges.Min(X).name = "i"+left
                    rect_sheets[-1].edges.Max(X).name = "i"+right
                    rect_sheets[-1].edges.Min(Y).name = "i"+bottom
                    rect_sheets[-1].edges.Max(Y).name = "i"+top
                xstart += self.dFe 

                # insulation
                if i != numSheets - 1:
                    wp.MoveTo(xstart, -d/2)
                    rect_sheets.append(wp.Rectangle(self.d0, d).Face())
                    rect_sheets[-1].name = "insulation"
                    rect_sheets[-1].col = (0.1, 0, 1, 1)
                    if self.onlySmooth:
                        rect_sheets[-1].edges.Min(Y).name = bottom
                        rect_sheets[-1].edges.Max(Y).name = top
                    xstart += self.d0

                
            if modelHalfAir:
                wp.MoveTo(xstart , -d/2)
                rect_sheets.append(wp.Rectangle(self.d0/2, d).Face())
                rect_sheets[-1].name = "insulation"
                rect_sheets[-1].col = (0.1, 0, 1, 1)
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(Y).name = bottom
                    rect_sheets[-1].edges.Max(Y).name = top
                xstart += self.d0


            if modelGap:
                if modelHalfAir:
                    wp.MoveTo(-d/2, -D/2)
                    rect_sheets.append(wp.Rectangle(d, (D-d)/2).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                    wp.MoveTo(-d/2, d/2)
                    rect_sheets.append(wp.Rectangle(d, (D-d)/2).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                else:
                    wp.MoveTo(-d/2 + self.d0/2, -D/2 )
                    rect_sheets.append(wp.Rectangle(d-self.d0, (D-d)/2 ).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                    wp.MoveTo(-d/2 + self.d0/2, d/2)
                    rect_sheets.append(wp.Rectangle(d-self.d0, (D-d)/2).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                

                    

        # ------------------------------------------------------------------------------
        # --- Multiscale
        # ------------------------------------------------------------------------------


        else: 
            xstart = -d/2
            # frame
            if modelHalfAir and domainNameHalfAir != "multiscale":
                wp.MoveTo(xstart, -d/2)
                rect_sheets.append(wp.Rectangle(self.d0/2, d).Face())
                rect_sheets[-1].name = domainNameHalfAir
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(Y).name = bottom
                    rect_sheets[-1].edges.Max(Y).name = top
                else:
                    rect_sheets[-1].edges.Min(Y).name = "i"+bottom
                    rect_sheets[-1].edges.Max(Y).name = "i"+top

                if self.onlyRough:
                    rect_sheets[-1].edges.Min(X).name = left
                    rect_sheets[-1].edges.Max(X).name = right
                else:
                    rect_sheets[-1].edges.Min(X).name = "i" + left + "_outer"
                    rect_sheets[-1].edges.Max(X).name = "i" + right
                rect_sheets[-1].col = (0, 0, 1, 1)


            # frameless multiscale
            if modelHalfAir and domainNameHalfAir == "multiscale":
                wp.MoveTo(xstart, -d/2)
                rect_sheets.append(wp.Rectangle(d , d).Face())
            else:
                xstart += self.d0/2
                wp.MoveTo(xstart, -d/2)
                rect_sheets.append(wp.Rectangle(d - self.d0, d).Face())
            rect_sheets[-1].name = "multiscale"
            if self.onlySmooth:
                rect_sheets[-1].edges.Min(Y).name = bottom
                rect_sheets[-1].edges.Max(Y).name = top
            else:
                rect_sheets[-1].edges.Min(Y).name = "i"+bottom
                rect_sheets[-1].edges.Max(Y).name = "i"+top

            if self.onlyRough:
                rect_sheets[-1].edges.Min(X).name = left
                rect_sheets[-1].edges.Max(X).name = right
            else:
                rect_sheets[-1].edges.Min(X).name = "i" + left
                rect_sheets[-1].edges.Max(X).name = "i" + right

            rect_sheets[-1].col = (1, 1, 0, 1)

            # frame
            if modelHalfAir and domainNameHalfAir != "multiscale":
                wp.MoveTo(-xstart, -d/2)
                rect_sheets.append(wp.Rectangle(self.d0/2, d).Face())
                rect_sheets[-1].name = domainNameHalfAir
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(Y).name = bottom
                    rect_sheets[-1].edges.Max(Y).name = top
                else:
                    rect_sheets[-1].edges.Min(Y).name = "i"+bottom
                    rect_sheets[-1].edges.Max(Y).name = "i"+top

                if self.onlyRough:
                    rect_sheets[-1].edges.Min(X).name = left
                    rect_sheets[-1].edges.Max(X).name = right
                else:
                    rect_sheets[-1].edges.Min(X).name = "i" + left
                    rect_sheets[-1].edges.Max(X).name = "i" + right + "_outer"
                
                rect_sheets[-1].col = (0, 0, 1, 1)

            if modelGap:

                if modelHalfAir:
                    # lower gap
                    wp.MoveTo(-d/2, -D/2)
                    rect_sheets.append(wp.Rectangle(d, (D-d)/2).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                    rect_sheets[-1].edges.Max(X).name = right
                    rect_sheets[-1].edges.Min(X).name = left
                    rect_sheets[-1].edges.Min(Y).name = bottom
                    

                    # upper gap
                    wp.MoveTo(-d/2, d/2)
                    rect_sheets.append(wp.Rectangle(d, (D-d)/2).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                    rect_sheets[-1].edges.Max(X).name = right
                    rect_sheets[-1].edges.Min(X).name = left
                    rect_sheets[-1].edges.Max(Y).name = top
                else:
                    # if domainNameHalfAir != "multiscale":
                    #     raise RuntimeError("model gap and different domain halfAir not implemented yet")

                    wp.MoveTo(-d/2 + self.d0/2, -D/2 )
                    rect_sheets.append(wp.Rectangle(d-self.d0, (D-d)/2 ).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                    rect_sheets[-1].edges.Max(X).name = right
                    rect_sheets[-1].edges.Min(X).name = left
                    rect_sheets[-1].edges.Min(Y).name = bottom

                    wp.MoveTo(-d/2 + self.d0/2, d/2)
                    rect_sheets.append(wp.Rectangle(d-self.d0, (D-d)/2).Face())
                    rect_sheets[-1].name = gapDomainName
                    rect_sheets[-1].col = (1, 0.5, 0.5, 1)
                    rect_sheets[-1].edges.Max(X).name = right
                    rect_sheets[-1].edges.Min(X).name = left
                    rect_sheets[-1].edges.Max(Y).name = top
                    
        for r in rect_sheets:   
            if r.name == "multiscale" or r.name == "inner":

                r.edges.Min(X if rotated else Y).maxh = maxh_edges[0]
                r.edges.Max(X if rotated else Y).maxh = maxh_edges[0]

                r.edges.Min(Y if rotated else X).maxh = maxh_edges[1]
                r.edges.Max(Y if rotated else X).maxh = maxh_edges[1]

        self.geo = Glue([outer - sum(rect_sheets)] + rect_sheets)

        if rotated: 
            self.geo = self.geo.Rotate(Axis(Pnt(0, 0, 0), Vec(0, 0, 1)), 90)


        # symmetry
        if not fullProblemX and not fullProblemY:
            cutting = wp.MoveTo(0, 0).Rectangle(2*D, 2*D).Face()
            cutting.edges.Min(X).name = "symLeft"
            cutting.edges.Min(Y).name = "symBottom"

            self.geo *= cutting
        elif not fullProblemX:
            cutting = wp.MoveTo(0, -2*D).Rectangle(2*D, 4*D).Face()
            cutting.edges.Min(X).name = "symLeft"
            self.geo *= cutting

        elif not fullProblemY:
            cutting = wp.MoveTo(-2*D, 0).Rectangle(4*D, 2*D).Face()
            cutting.edges.Min(Y).name = "symBottom"
            self.geo *= cutting



        self.mesh = Mesh(OCCGeometry(self.geo, dim=2).GenerateMesh(maxh=maxh, quad_dominated=self.quad_dominated))
        
        coef = 1
        coef *= (1 if fullProblemX else 2)
        coef *= (1 if fullProblemY else 2)

        width = self.D if not onlySmooth else self.d
        height = self.D if not onlyRough else self.d

        assert_almost(Integrate(1, self.mesh) * coef, width * height, 1e-6, f"volume is wrong")

        if multiscale:
            if not (modelGap and gapDomainName == "multiscale"):
                if modelHalfAir and domainNameHalfAir == "multiscale":
                    assert_almost(Integrate(1, self.mesh, definedon=self.mesh.Materials("multiscale")) * coef, self.d * self.d, 1e-6, "multiscale area is wrong")
                else:
                    if modelHalfAir:
                        assert_almost(Integrate(1, self.mesh, definedon=self.mesh.Materials(self.domainNameHalfAir)) * coef, self.d * (self.d0), 1e-6, "multiscale area is wrong")
                    assert_almost(Integrate(1, self.mesh, definedon=self.mesh.Materials("multiscale")) * coef, self.d * (self.d - self.d0), 1e-6, "multiscale area is wrong")
            else:
                if modelHalfAir and domainNameHalfAir == "multiscale":
                    assert_almost(Integrate(1, self.mesh, definedon=self.mesh.Materials("multiscale")) * coef, self.D * self.d, 1e-6, "multiscale area is wrong")
                else:
                    assert_almost(Integrate(1, self.mesh, definedon=self.mesh.Materials("multiscale")) * coef, self.D * (self.d - self.d0), 1e-6, "multiscale area is wrong")
        else:
            assert_almost(Integrate(1, self.mesh, definedon=self.mesh.Materials("inner")) * coef, (self.dFe * self.d) * self.numSheets, 1e-6, "sheets area is wrong")


        



def run():
    from ngsolve import Draw, CF
    from myPackage import drawBndAll
    D = 0.04
    d = 0.02

    ff = 0.9


    numSheets = 10

    maxh_edges = d/numSheets*1/2 * 0.5


    onlySmooth = False
    onlyRough = True

    cMeshRef = mesh2DLaminates(D, d, ff, numSheets, multiscale=True, onlySmooth=onlySmooth, onlyRough=onlyRough,
        maxh_edges=maxh_edges, rotated=True, fullProblemX=True, fullProblemY=True,  modelGap=True,
        modelHalfAir=True)
    meshRef = cMeshRef.mesh

    print(meshRef.GetMaterials())
    Draw(meshRef.MaterialCF({"inner":1, "outer":2, "gap":3, "insulation":4, "multiscale":1}, default=10), meshRef, "test")

    drawBndAll(meshRef)

if __name__ == "__main__":

    run()
