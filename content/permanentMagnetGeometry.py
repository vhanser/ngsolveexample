from netgen.occ import *
from ngsolve import *
from netgen.meshing import IdentificationType

from netgen.webgui import Draw as DrawGeo
from ngsolve.webgui import Draw
import numpy as np

class simpleGeoMagnetOnIron:
#                   B
#                                 
#       ---------------------------
#      |                          |     r4-r3
#      |        ------------      |
#      |        |          |      |      r3-r2
#      |--------------------------| 
#      |                          |        r2-r1
#     WP --------------------------  
#
#               |    BM    |
#
#                                           r1
#
#                    0                   
#
#
#

    def __init__(self, r1, r2, r3, r4, Bm=4e-3, B=8e-3, maxh=1, maxhRotor=1, maxhMagnet=1, maxhAir=1, periodic=False, maxhEdges = 1):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.Bm = Bm
        self.B = B
        self.maxh = maxh
        self.periodic = periodic
        self.maxhMagnet = maxhMagnet
        self.maxhRotor = maxhRotor
        self.maxhAir = maxhAir


        
          

        wp = WorkPlane()
        wp.MoveTo(-B/2,r1)
        rotor = wp.Rectangle(B,r2 - r1).Face()
        rotor.name = "rotor"
        rotor.edges.Min(X).name = "left_iron" if not periodic else "periodic"
        rotor.edges.Max(X).name = "right_iron" if not periodic else "periodic"
        if B != Bm:
            rotor.edges.Max(Y).name = "interface"
        rotor.edges.Min(Y).name = "inner"
        rotor.edges.Max(Y).maxh = maxhEdges
        rotor.maxh = maxhRotor

        wp.MoveTo(-B/2 + (B-Bm)/2,r2)
        magnet = wp.Rectangle(Bm,r3-r2).Face()
        magnet.name = "magnet"
        
        if B == Bm:
            magnet.edges.Min(X).name = "left_magnet" if not periodic else "periodic"
            magnet.edges.Max(X).name = "right_magnet" if not periodic else "periodic"    
            magnet.edges.Max(Y).name = "interface"
        else:
            magnet.edges.name = "interface"
        magnet.edges.maxh = maxhEdges
        magnet.maxh = maxhMagnet

        magnet.edges.Min(Y).name = "mag_rot"

        
        
        if B == Bm:
            wp.MoveTo(-B/2,r3)
            air = wp.Rectangle(B,r4-r3).Face()
        else:
            wp.MoveTo(-B/2,r2)
            air = wp.Rectangle(B,r4-r2).Face()
        air.name = "air"
        air.edges.Min(X).name = "left_air" if not periodic else "periodic"
        air.edges.Max(X).name = "right_air" if not periodic else "periodic"
        air.edges.Max(Y).name = "outer"
        air.maxh = maxhAir

        if periodic:
            self.geo = Glue([rotor, magnet, air-magnet])

            edge_air_left = self.geo.edges.Nearest(Pnt(-B/2, r3+(r4-r3)/2 , 0))
            edge_rotor_left = self.geo.edges.Nearest(Pnt(-B/2, r1+(r2-r1)/2 , 0))

            edge_air_right = self.geo.edges.Nearest(Pnt(B/2, r3+(r4-r3)/2 , 0))
            edge_rotor_right = self.geo.edges.Nearest(Pnt(B/2, r1+(r2-r1)/2 , 0))
            
            edge_air_left.name= "periodic_air"
            edge_rotor_left.name= "periodic_rotor"
            edge_air_right.name= "periodic_air"
            edge_rotor_right.name= "periodic_rotor"

            edge_air_right.Identify(edge_air_left, "periodic_air", IdentificationType.PERIODIC)
            edge_rotor_right.Identify(edge_rotor_left, "periodic_rotor", IdentificationType.PERIODIC)

            if B == Bm:
                edge_magnet_right = self.geo.edges.Nearest(Pnt(B/2, r2+(r3-r2)/2 , 0))
                edge_magnet_left = self.geo.edges.Nearest(Pnt(-B/2, r2+(r3-r2)/2 , 0))
                edge_magnet_left.name= "periodic_magnet"
                edge_magnet_right.name= "periodic_magnet"
                edge_magnet_right.Identify(edge_magnet_left, "periodic_magnet", IdentificationType.PERIODIC)




        
        else:
            # self.geo = Glue([rotor, magnet, air-magnet])
            self.geo = Glue([rotor, magnet, air-magnet])

        
        self.mesh = Mesh(OCCGeometry(self.geo, dim=2).GenerateMesh(maxh=maxh))
        
class fullGeoMagnet:

    def __init__(self, r1, r2, r3, r4, phi_deg=45, maxh=1, periodic=False, maxhEdges = 1):
        
        self.maxh = maxh
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4

        self.phi_deg = phi_deg

        self.periodic = periodic


        inner = WorkPlane().Circle(r1).Face()
        inner.edges.name="inner"

        rotor = WorkPlane().Circle(r2).Face()
        rotor.name="rotor"
        rotor.col=(0.8, 0.5, 0)
        
        
        # magnet
        magnet = WorkPlane(Axes((0,0,0), n=Z, h=X))
        
        magnet.MoveTo(0, r2)
        magnet.Arc(r2, -phi_deg).Rotate(90)
        magnet.Line(r3-r2).Rotate(90)
        magnet.Arc(r3, 2*phi_deg).Rotate(90)
        magnet.Line(r3-r2).Rotate(90)
        magnet.Arc(r2, -phi_deg).Rotate(90)
        magnet_top = magnet.Face()
        magnet_top.name="magnet_top"
        magnet_top.edges.name = "interface"
        magnet_top.edges.Nearest(Pnt(r1/10, r2)).name = "mag_rot"
        magnet_top.edges.Nearest(Pnt(-r1/10, r2)).name = "mag_rot"
        magnet_top.col=(1, 0, 0)

        magnet_top.edges.maxh = maxhEdges
        

        magnet = WorkPlane(Axes((0,0,0), n=Z, h=-X))
        
        magnet.MoveTo(0, r2)
        magnet.Arc(r2, -phi_deg).Rotate(90)
        magnet.Line(r3-r2).Rotate(90)
        magnet.Arc(r3, 2*phi_deg).Rotate(90)
        magnet.Line(r3-r2).Rotate(90)
        magnet.Arc(r2, -phi_deg).Rotate(90)
        magnet_bottom = magnet.Face()
        magnet_bottom.name="magnet_bottom"
        magnet_bottom.edges.name = "interface"
        magnet_bottom.edges.Nearest(Pnt(r1/10, -r2)).name = "mag_rot"
        magnet_bottom.edges.Nearest(Pnt(-r1/10, -r2)).name = "mag_rot"
        magnet_bottom.col=(0, 0, 1)

        magnet_bottom.edges.maxh = maxhEdges
        
        outer = WorkPlane().Circle(r4).Face()
        outer.name="air"
        outer.edges.name="outer"
        
        outer.col=(0.7, 0.7, 0.7)
        
        
        if periodic:
            
            angle = 0
            ez   = Axis((0, 0, 0), Z)

            symy_cutting_right = WorkPlane(Axes((-2*r4,0,0), n=Z, h=X)).Rectangle(4*r4, 2*r4).Face().Rotate(ez, -angle/2)
            symy_cutting_left = WorkPlane(Axes((-2*r4,0,0), n=Z, h=X)).Rectangle(4*r4, 2*r4).Face().Rotate(ez, angle/2)
            
            outer, rotor, inner, magnet_top, magnet_bottom = [_*(symy_cutting_left+symy_cutting_right) for _ in (outer, rotor, inner, magnet_top, magnet_bottom)]
            
            air = outer-rotor-magnet_top - magnet_bottom
            rotor.edges.maxh = maxhEdges
            rotor.edges.name = "interface"
            rotor = rotor - inner 

           
            # edge_rotor_right.name= "periodic_right"

            self.geo = Glue([air, rotor ,  magnet_top, magnet_bottom])

            edge_air_left = self.geo.edges.Nearest(Pnt(-(r4+r3)/2 * np.cos(angle/2*np.pi/180), -(r4+r3)/2 * np.sin(angle/2*np.pi/180), 0))
            edge_rotor_left = self.geo.edges.Nearest(Pnt(-(r2+r1)/2 * np.cos(angle/2*np.pi/180), -(r2+r1)/2 * np.sin(angle/2*np.pi/180)))

            edge_air_right = self.geo.edges.Nearest(Pnt((r4+r3)/2 * np.cos(-angle/2*np.pi/180), (r4+r3)/2 * np.sin(-angle/2*np.pi/180), 0))
            edge_rotor_right = self.geo.edges.Nearest(Pnt((r2+r1)/2 * np.cos(-angle/2*np.pi/180), (r2+r1)/2 * np.sin(-angle/2*np.pi/180)))


            edge_air_left.name= "periodic_air"
            edge_rotor_left.name= "periodic_rotor"
            edge_air_right.name= "periodic_air"
            edge_rotor_right.name= "periodic_rotor"


            trafo = Rotation( ez, 180+angle)
            edge_air_right.Identify(edge_air_left, "periodic_air", IdentificationType.PERIODIC, trafo)
            edge_rotor_right.Identify(edge_rotor_left, "periodic_rotor", IdentificationType.PERIODIC, trafo)

        else:
            rotor.edges.maxh = maxhEdges
            rotor.edges.name = "interface"
            

            self.geo = Glue([outer- rotor- magnet_top - magnet_bottom, rotor-inner ,  magnet_top, magnet_bottom])
        
        self.mesh = Mesh(OCCGeometry(self.geo, dim=2).GenerateMesh(maxh=maxh))
        self.mesh.Curve(3)
        

if __name__ == "__main__":
    import netgen.gui
    if False:
        geo_D = 1e-2
        geo_tau = 0.8

        geo_B = geo_D*np.pi/2
        cMesh = simpleGeoMagnetOnIron(B = geo_B , Bm = geo_B*geo_tau)
        mesh = cMesh.mesh
        
    else:
        
        cMesh = fullGeoMagnet(r1 = 0.1 , r2 = 0.2, r3 = 0.3, r4 = 0.4, periodic=False )
        mesh = cMesh.mesh
        mesh.Curve(3)

        Draw(cMesh.mesh)

