"""
FreeCAD stub using pythonocc (OCC) as backend.
Covers all App/Part/Mesh attributes used in the CADDreamer codebase.
"""
import numpy as np

# ── Vector ────────────────────────────────────────────────────────────────────
class _Vector:
    def __init__(self, x=0, y=0, z=0):
        if hasattr(x, '__len__'):
            self.x, self.y, self.z = float(x[0]), float(x[1]), float(x[2])
        else:
            self.x, self.y, self.z = float(x), float(y), float(z)
    @staticmethod
    def _xyz(o):
        # Accept _Vector, numpy arrays, lists, tuples interchangeably
        if hasattr(o, 'x'):
            return float(o.x), float(o.y), float(o.z)
        return float(o[0]), float(o[1]), float(o[2])
    def __add__(self, o):
        ox, oy, oz = _Vector._xyz(o)
        return _Vector(self.x+ox, self.y+oy, self.z+oz)
    def __radd__(self, o):  return self.__add__(o)
    def __sub__(self, o):
        ox, oy, oz = _Vector._xyz(o)
        return _Vector(self.x-ox, self.y-oy, self.z-oz)
    def __rsub__(self, o):
        ox, oy, oz = _Vector._xyz(o)
        return _Vector(ox-self.x, oy-self.y, oz-self.z)
    def __mul__(self, s):  return _Vector(self.x*s, self.y*s, self.z*s)
    def __rmul__(self, s): return self.__mul__(s)
    def __truediv__(self, s): return _Vector(self.x/s, self.y/s, self.z/s)
    def __iadd__(self, o):
        ox, oy, oz = _Vector._xyz(o)
        self.x+=ox; self.y+=oy; self.z+=oz; return self
    def __iter__(self):    return iter([self.x, self.y, self.z])
    def __len__(self):     return 3
    def __getitem__(self, i): return [self.x, self.y, self.z][i]
    def __repr__(self):    return f"Vector({self.x},{self.y},{self.z})"
    def to_np(self):       return np.array([self.x, self.y, self.z])

# ── Surface base with ALL attributes used in the codebase ────────────────────
class _BaseSurf:
    TypeId      = 'Part::GeomUnknown'
    # geometry attributes
    Axis        = [0.0, 0.0, 1.0]
    Center      = [0.0, 0.0, 0.0]
    Position    = [0.0, 0.0, 0.0]
    Radius      = 1.0
    SemiAngle   = 0.0
    MajorRadius = 1.0
    MinorRadius = 0.3

    def projectPoint(self, pt, *a, **kw):
        # Real projection onto the underlying OCC surface when available
        # (surfaces obtained via _OCCShape.Surface carry _occ_face).
        if hasattr(pt, '__len__'):
            p = [float(pt[0]), float(pt[1]), float(pt[2])]
        elif hasattr(pt, 'x'):
            p = [float(pt.x), float(pt.y), float(pt.z)]
        else:
            return [0.0, 0.0, 0.0]
        f = getattr(self, '_occ_face', None)
        if f is not None:
            try:
                from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.gp import gp_Pnt as _P
                gs = BRep_Tool.Surface(f)
                proj = GeomAPI_ProjectPointOnSurf(_P(p[0], p[1], p[2]), gs)
                if proj.NbPoints() > 0:
                    q = proj.NearestPoint()
                    return [q.X(), q.Y(), q.Z()]
            except Exception:
                pass
        return p

class _CylinderSurf(_BaseSurf): TypeId = 'Part::GeomCylinder'
class _SphereSurf(_BaseSurf):   TypeId = 'Part::GeomSphere'
class _ConeSurf(_BaseSurf):     TypeId = 'Part::GeomCone'
class _PlaneSurf(_BaseSurf):    TypeId = 'Part::GeomPlane'
class _ToroidSurf(_BaseSurf):   TypeId = 'Part::GeomToroid'
class _BSplineSurf(_BaseSurf):  TypeId = 'Part::GeomBSplineSurface'
class _UnknownSurf(_BaseSurf):  TypeId = 'Part::GeomUnknown'

_SURF_MAP = {
    'Cylinder':     _CylinderSurf,
    'Sphere':       _SphereSurf,
    'Cone':         _ConeSurf,
    'Plane':        _PlaneSurf,
    'Toroid':       _ToroidSurf,
    'Torus':        _ToroidSurf,
    'BSplineSurface': _BSplineSurf,
}

# ── OCC imports ───────────────────────────────────────────────────────────────
try:
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere,
                                      BRepPrimAPI_MakeCone, BRepPrimAPI_MakeTorus)
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (GeomAbs_Cylinder, GeomAbs_Sphere, GeomAbs_Cone,
                                   GeomAbs_Plane, GeomAbs_Torus)
    OCC_AVAILABLE = True
except Exception:
    OCC_AVAILABLE = False

def _to_pnt(v):
    if v is None: return gp_Pnt(0,0,0)
    return gp_Pnt(float(v.x), float(v.y), float(v.z))

def _to_dir(v):
    if v is None: return gp_Dir(0,0,1)
    arr = np.array([v.x, v.y, v.z], dtype=float)
    n = np.linalg.norm(arr)
    if n < 1e-10: arr = np.array([0.,0.,1.])
    else: arr /= n
    return gp_Dir(float(arr[0]), float(arr[1]), float(arr[2]))

# ── Edge / Vertex wrappers ────────────────────────────────────────────────────
class _OCCVertex:
    def __init__(self, v):
        self._v = v
        self._shape = v
    @property
    def Point(self):
        try:
            from OCC.Core.BRep import BRep_Tool
            p = BRep_Tool.Pnt(self._v)
            return np.array([p.X(), p.Y(), p.Z()])
        except Exception:
            return np.zeros(3)

class _OCCEdge:
    def __init__(self, e):
        self._e = e
        self._shape = e
    @property
    def Vertexes(self):
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX
            from OCC.Core.TopoDS import topods
            out, exp = [], TopExp_Explorer(self._e, TopAbs_VERTEX)
            while exp.More():
                out.append(_OCCVertex(topods.Vertex(exp.Current())))
                exp.Next()
            return out
        except Exception:
            return []
    def isSame(self, other):
        try:
            return self._e.IsSame(getattr(other, '_e', other))
        except Exception:
            return False
    def IsSame(self, other):
        return self.isSame(other)

# ── _OCCShape ─────────────────────────────────────────────────────────────────
class _OCCShape:
    def __init__(self, occ_shape=None, surf_type=None):
        self._shape     = occ_shape
        self._surf_type = surf_type

    @property
    def Surface(self):
        """Surface object populated with REAL geometry read back from the OCC shape.

        Critical: freecad2newtongeom() reconstructs newton primitives by reading
        .Axis/.Center/.Position/.Radius/... off this object. Returning defaults
        here collapses every fitted plane to normal (0,0,1) and breaks all
        downstream intersections.
        """
        surf_cls = _SURF_MAP.get(self._surf_type or '', _UnknownSurf)
        surf = surf_cls()
        if not OCC_AVAILABLE or self._shape is None:
            return surf
        try:
            from OCC.Core.TopoDS import topods
            sh = self._shape
            if sh.ShapeType() != TopAbs_FACE:
                exp = TopExp_Explorer(sh, TopAbs_FACE)
                if not exp.More():
                    return surf
                sh = exp.Current()
            face = topods.Face(sh)
            ad = BRepAdaptor_Surface(face)
            t = ad.GetType()
            if t == GeomAbs_Plane:
                g = ad.Plane(); d = g.Axis().Direction(); l = g.Location()
                surf.Axis = [d.X(), d.Y(), d.Z()]
                surf.Position = [l.X(), l.Y(), l.Z()]
                surf.Center = list(surf.Position)
            elif t == GeomAbs_Cylinder:
                g = ad.Cylinder(); d = g.Axis().Direction(); l = g.Location()
                surf.Axis = [d.X(), d.Y(), d.Z()]
                surf.Center = [l.X(), l.Y(), l.Z()]
                surf.Position = list(surf.Center)
                surf.Radius = g.Radius()
            elif t == GeomAbs_Sphere:
                g = ad.Sphere(); l = g.Location()
                surf.Center = [l.X(), l.Y(), l.Z()]
                surf.Position = list(surf.Center)
                surf.Radius = g.Radius()
            elif t == GeomAbs_Cone:
                g = ad.Cone(); d = g.Axis().Direction(); l = g.Location()
                surf.Axis = [d.X(), d.Y(), d.Z()]
                surf.Center = [l.X(), l.Y(), l.Z()]
                surf.Position = list(surf.Center)
                surf.SemiAngle = g.SemiAngle()
            elif t == GeomAbs_Torus:
                g = ad.Torus(); d = g.Axis().Direction(); l = g.Location()
                surf.Axis = [d.X(), d.Y(), d.Z()]
                surf.Center = [l.X(), l.Y(), l.Z()]
                surf.Position = list(surf.Center)
                surf.MajorRadius = g.MajorRadius()
                surf.MinorRadius = g.MinorRadius()
            surf._occ_face = face  # enables real projectPoint()
        except Exception:
            pass
        return surf

    @property
    def Faces(self):
        # Always explore the real OCC faces first: a solid (e.g. makeCylinder)
        # contains caps + lateral faces, and callers filter by Surface type.
        if OCC_AVAILABLE and self._shape is not None:
            try:
                _t2s = {GeomAbs_Cylinder:'Cylinder', GeomAbs_Sphere:'Sphere',
                        GeomAbs_Cone:'Cone', GeomAbs_Plane:'Plane', GeomAbs_Torus:'Toroid'}
                exp   = TopExp_Explorer(self._shape, TopAbs_FACE)
                faces = []
                while exp.More():
                    face = exp.Current()
                    t    = BRepAdaptor_Surface(face).GetType()
                    faces.append(_OCCShape(face, _t2s.get(t, 'Unknown')))
                    exp.Next()
                if faces:
                    return faces
            except Exception:
                pass
        if self._surf_type in _SURF_MAP:
            return [self]
        return []

    @property
    def Edges(self):
        if not OCC_AVAILABLE or self._shape is None:
            return []
        try:
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopoDS import topods
            out, exp = [], TopExp_Explorer(self._shape, TopAbs_EDGE)
            while exp.More():
                out.append(_OCCEdge(topods.Edge(exp.Current())))
                exp.Next()
            return out
        except Exception:
            return []

    @property
    def Vertexes(self):
        if not OCC_AVAILABLE or self._shape is None:
            return []
        try:
            from OCC.Core.TopAbs import TopAbs_VERTEX
            from OCC.Core.TopoDS import topods
            out, exp = [], TopExp_Explorer(self._shape, TopAbs_VERTEX)
            while exp.More():
                out.append(_OCCVertex(topods.Vertex(exp.Current())))
                exp.Next()
            return out
        except Exception:
            return []
    @property
    def Shape(self):    return self
    @property
    def Object(self):   return self

    def tessellate(self, tol=0.1):
        if not OCC_AVAILABLE or self._shape is None:
            return ([], [])
        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRep import BRep_Tool
            BRepMesh_IncrementalMesh(self._shape, float(tol))
            exp = TopExp_Explorer(self._shape, TopAbs_FACE)
            verts, tris = [], []
            offset = 0
            while exp.More():
                face = exp.Current()
                loc = face.Location()
                tri_data = BRep_Tool.Triangulation(face, loc)
                if tri_data is None:
                    exp.Next(); continue
                n = tri_data.NbNodes()
                for i in range(1, n+1):
                    p = tri_data.Node(i)
                    verts.append((p.X(), p.Y(), p.Z()))
                nt = tri_data.NbTriangles()
                for i in range(1, nt+1):
                    a, b, c = tri_data.Triangle(i).Get()
                    tris.append((a-1+offset, b-1+offset, c-1+offset))
                offset += n
                exp.Next()
            return (verts, tris)
        except Exception:
            return ([], [])
    def translate(self, vec):
        return self.translated(vec)

    def translated(self, vec):
        if not OCC_AVAILABLE or self._shape is None:
            return self
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.gp import gp_Trsf, gp_Vec
            if hasattr(vec, 'x'):
                v = gp_Vec(float(vec.x), float(vec.y), float(vec.z))
            else:
                v = gp_Vec(float(vec[0]), float(vec[1]), float(vec[2]))
            trsf = gp_Trsf()
            trsf.SetTranslation(v)
            moved = BRepBuilderAPI_Transform(self._shape, trsf, True).Shape()
            return _OCCShape(moved, self._surf_type)
        except Exception as e:
            return self
    def _boolean(self, other, api):
        o = getattr(other, '_shape', None)
        if not OCC_AVAILABLE or self._shape is None or o is None:
            return _EmptyCompound()
        try:
            t = api(self._shape, o)
            t.Build()
            if not t.IsDone():
                return _EmptyCompound()
            return _OCCShape(t.Shape(), None)
        except Exception:
            return _EmptyCompound()

    def cut(self, other):
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        return self._boolean(other, BRepAlgoAPI_Cut)

    def common(self, other):
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        return self._boolean(other, BRepAlgoAPI_Common)

    def section(self, other):
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
        return self._boolean(other, BRepAlgoAPI_Section)

    def distToShape(self, other):
        o = getattr(other, '_shape', None)
        if not OCC_AVAILABLE or self._shape is None or o is None:
            return (1e9, [], [])
        try:
            from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
            d = BRepExtrema_DistShapeShape(self._shape, o)
            if d.IsDone():
                return (d.Value(), [], [])
        except Exception:
            pass
        return (1e9, [], [])

    def isNull(self):                return self._shape is None
    def IsSame(self, other):
        try:
            return self._shape.IsSame(getattr(other, '_shape', other))
        except Exception:
            return False


class _EmptyCompound:
    Faces    = []
    Edges    = []
    Vertexes = []
    Shape    = None
    def tessellate(self, tol=0.1):
        if not OCC_AVAILABLE or self._shape is None:
            return ([], [])
        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRep import BRep_Tool
            BRepMesh_IncrementalMesh(self._shape, float(tol))
            exp = TopExp_Explorer(self._shape, TopAbs_FACE)
            verts, tris = [], []
            offset = 0
            while exp.More():
                face = exp.Current()
                loc = face.Location()
                tri_data = BRep_Tool.Triangulation(face, loc)
                if tri_data is None:
                    exp.Next(); continue
                n = tri_data.NbNodes()
                for i in range(1, n+1):
                    p = tri_data.Node(i)
                    verts.append((p.X(), p.Y(), p.Z()))
                nt = tri_data.NbTriangles()
                for i in range(1, nt+1):
                    a, b, c = tri_data.Triangle(i).Get()
                    tris.append((a-1+offset, b-1+offset, c-1+offset))
                offset += n
                exp.Next()
            return (verts, tris)
        except Exception:
            return ([], [])
    def cut(self, *a):              return _EmptyCompound()
    def common(self, *a):           return _EmptyCompound()
    def section(self, *a):          return _EmptyCompound()
    def distToShape(self, *a):      return (1e9, [], [])
    def isNull(self):               return True

# ── Part ──────────────────────────────────────────────────────────────────────
class _PartMeta(type):
    """Metaclass so Part.AnyUndefinedAttr returns a safe stub instead of AttributeError."""
    def __getattr__(cls, name):
        # Return a dummy type for type() comparisons, or a no-op callable
        dummy = type(name, (_BaseSurf,), {'TypeId': f'Part::Geom{name}'})
        setattr(cls, name, dummy)   # cache it
        return dummy

class Part(metaclass=_PartMeta):
    Cylinder       = _CylinderSurf
    Sphere         = _SphereSurf
    Cone           = _ConeSurf
    Plane          = _PlaneSurf
    Toroid         = _ToroidSurf
    BSplineSurface = _BSplineSurf
    Face           = _OCCShape
    Shape          = _OCCShape

    @staticmethod
    def makeCylinder(radius, height, pnt=None, direction=None, angle=360):
        if not OCC_AVAILABLE: return _OCCShape(None, 'Cylinder')
        try:
            ax = gp_Ax2(_to_pnt(pnt), _to_dir(direction))
            return _OCCShape(BRepPrimAPI_MakeCylinder(ax, float(radius), float(height)).Shape(), 'Cylinder')
        except Exception: return _OCCShape(None, 'Cylinder')

    @staticmethod
    def makeSphere(radius, pnt=None):
        if not OCC_AVAILABLE: return _OCCShape(None, 'Sphere')
        try:
            ax = gp_Ax2(_to_pnt(pnt), gp_Dir(0,0,1))
            return _OCCShape(BRepPrimAPI_MakeSphere(ax, float(radius)).Shape(), 'Sphere')
        except Exception: return _OCCShape(None, 'Sphere')

    @staticmethod
    def makeCone(r1, r2, height, pnt=None, direction=None):
        if not OCC_AVAILABLE: return _OCCShape(None, 'Cone')
        try:
            ax = gp_Ax2(_to_pnt(pnt), _to_dir(direction))
            r1, r2 = float(r1), float(r2)
            # OCC rejects identical radii (degenerate cone → DomainError); nudge them apart.
            if abs(r1 - r2) < 1e-4:
                r2 = r1 + 1e-3
            return _OCCShape(BRepPrimAPI_MakeCone(ax, r1, r2, float(height)).Shape(), 'Cone')
        except Exception: return _OCCShape(None, 'Cone')

    @staticmethod
    def makeTorus(r1, r2, pnt=None, direction=None, angle1=0, angle2=360, angle=360):
        if not OCC_AVAILABLE: return _OCCShape(None, 'Toroid')
        try:
            ax = gp_Ax2(_to_pnt(pnt), _to_dir(direction))
            return _OCCShape(BRepPrimAPI_MakeTorus(ax, float(r1), float(r2)).Shape(), 'Toroid')
        except Exception: return _OCCShape(None, 'Toroid')

    @staticmethod
    def makePlane(w, h, pnt=None, direction=None):
        if not OCC_AVAILABLE:
            return _OCCShape(None, 'Plane')
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.gp import gp_Pln
            p = _to_pnt(pnt) if pnt else gp_Pnt(0, 0, 0)
            d = _to_dir(direction) if direction else gp_Dir(0, 0, 1)
            pln = gp_Pln(p, d)
            half_w, half_h = float(w) / 2, float(h) / 2
            face = BRepBuilderAPI_MakeFace(pln, -half_w, half_w, -half_h, half_h).Face()
            return _OCCShape(face, 'Plane')
        except Exception as e:
            return _OCCShape(None, 'Plane')

    @staticmethod
    def Compound(shapes):
        if not OCC_AVAILABLE:
            return _EmptyCompound()
        try:
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.BRep import BRep_Builder
            comp = TopoDS_Compound()
            b = BRep_Builder()
            b.MakeCompound(comp)
            added = 0
            for s in shapes:
                sh = getattr(s, '_shape', None)
                if sh is not None:
                    b.Add(comp, sh)
                    added += 1
            if added == 0:
                return _EmptyCompound()
            return _OCCShape(comp, None)
        except Exception:
            return _EmptyCompound()

    @staticmethod
    def makeCompound(shapes):
        return Part.Compound(shapes)

    @staticmethod
    def __fromPythonOCC__(shape):
        return _OCCShape(shape)

# ── App ───────────────────────────────────────────────────────────────────────
class _ObjStub:
    Shape = _OCCShape(None)
    def __getattr__(self, name): return None

class _Document:
    def addObject(self, *a, **kw): return _ObjStub()
    def recompute(self): pass

class App:
    Vector         = _Vector
    ActiveDocument = _Document()

    @staticmethod
    def newDocument(name="Unnamed"): return _Document()

    @staticmethod
    def activeDocument(): return _Document()

# ── Mesh ──────────────────────────────────────────────────────────────────────
class Mesh:
    pass
