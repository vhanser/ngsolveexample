from netgen.occ import *
from ngsolve import Mesh



class mesh2DLaminates():
    def __init__(self, D, d, ff, numSheets, multiscale=False, modelHalfAir=False, domainNameHalfAir="multiscale", onlySmooth=False, maxh_edges=None, maxh = None):

        self.D = D
        self.d = d
        self.ff = ff
        self.numSheets = numSheets
        self.multiscale = multiscale
        self.modelHalfAir = modelHalfAir
        self.onlySmooth = onlySmooth
        self.maxh_edges = maxh_edges
        self.maxh = maxh
        self.dFe = d/numSheets * ff
        self.d0 = d/numSheets * (1-ff)
        self.domainNameHalfAir = domainNameHalfAir

        if type(maxh) == type(None):
            maxh = self.d

        if type(maxh_edges) == type(None):
            maxh_edges = self.d0 * 2



        wp = WorkPlane()
        if onlySmooth:
            outer = wp.RectangleC(D, d).Face()
        else:
            outer = wp.RectangleC(D, D).Face()

        outer.name="outer"
        outer.edges.Min(X).name = "left"
        outer.edges.Max(X).name = "right"
        outer.edges.Min(Y).name = "bottom"
        outer.edges.Max(Y).name = "top"


        xstart = -d/2 + self.d0/2
        rect_sheets = []
        if not multiscale:

            for i in range(numSheets):
                wp.MoveTo(xstart, -d/2)
                #iron
                rect_sheets.append(wp.Rectangle(self.dFe, d).Face())
                rect_sheets[-1].edges.maxh = maxh_edges
                rect_sheets[-1].name = "inner"
                rect_sheets[-1].col = (1, 0, 0, 1)
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(Y).name = "bottom"
                    rect_sheets[-1].edges.Max(Y).name = "top"
                rect_sheets[-1].edges.Min(X).name = "ileft"
                rect_sheets[-1].edges.Max(X).name = "iright"
                xstart += self.dFe 

                # gap
                wp.MoveTo(xstart, -d/2)
                rect_sheets.append(wp.Rectangle(self.d0, d).Face())
                rect_sheets[-1].name = "gap"
                rect_sheets[-1].col = (1, 0, 1, 1)
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(Y).name = "bottom"
                    rect_sheets[-1].edges.Max(Y).name = "top"
                xstart += self.d0



        else: 
            if modelHalfAir and domainNameHalfAir=="multiscale":
                wp.MoveTo(xstart-self.d0/2, -d/2)
                rect_sheets.append(wp.Rectangle(d, d).Face())
                rect_sheets[-1].edges.maxh = maxh_edges
                rect_sheets[-1].name = "multiscale"
                rect_sheets[-1].col = (1, 1, 0, 1)
                rect_sheets[-1].edges.Min(X).name = "ileft"
                rect_sheets[-1].edges.Max(X).name = "iright"

            else:
                if modelHalfAir and domainNameHalfAir != "multiscale":
                    wp.MoveTo(xstart, -d/2)
                    rect_sheets.append(wp.Rectangle(self.d0/2, d).Face())
                    rect_sheets[-1].edges.maxh = maxh_edges
                    rect_sheets[-1].name = domainNameHalfAir
                    if self.onlySmooth:
                        rect_sheets[-1].edges.Min(Y).name = "bottom"
                        rect_sheets[-1].edges.Max(Y).name = "top"
                    rect_sheets[-1].edges.Min(X).name = "ileft_outer"
                    rect_sheets[-1].edges.Max(X).name = "ileft"
                    rect_sheets[-1].col = (0, 0, 1, 1)
                # frameless multiscale
                wp.MoveTo(xstart, -d/2)
                rect_sheets.append(wp.Rectangle(d - self.d0, d).Face())
                rect_sheets[-1].edges.maxh = maxh_edges
                rect_sheets[-1].name = "multiscale"
                if self.onlySmooth:
                    rect_sheets[-1].edges.Min(Y).name = "bottom"
                    rect_sheets[-1].edges.Max(Y).name = "top"
                rect_sheets[-1].edges.Min(X).name = "ileft"
                rect_sheets[-1].edges.Max(X).name = "iright"
                rect_sheets[-1].col = (1, 1, 0, 1)

                if modelHalfAir and domainNameHalfAir != "multiscale":
                    wp.MoveTo(-xstart, -d/2)
                    rect_sheets.append(wp.Rectangle(self.d0/2, d).Face())
                    rect_sheets[-1].edges.maxh = maxh_edges
                    rect_sheets[-1].name = domainNameHalfAir
                    if self.onlySmooth:
                        rect_sheets[-1].edges.Min(Y).name = "bottom"
                        rect_sheets[-1].edges.Max(Y).name = "top"
                    rect_sheets[-1].edges.Min(X).name = "iright"
                    rect_sheets[-1].edges.Max(X).name = "iright_outer"
                    rect_sheets[-1].col = (0, 0, 1, 1)
                    


        self.geo = Glue([outer - sum(rect_sheets)] + rect_sheets)
        self.mesh = Mesh(OCCGeometry(self.geo, dim=2).GenerateMesh(maxh=maxh))


