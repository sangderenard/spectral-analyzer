"""Authored axial rod-bundle partition geometry (CPU reference).

Angular bisectors follow the generator's curved centreline and twist. Only
actual occupied rods compete; no blending or insertion-order material priority.
This module does not enable a transport mode. Integration must intersect the
new internal interfaces, not merely select an owner at old rod surfaces.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class AxialRodBundle:
    regions: tuple[int,...]
    eye_length: float
    eye_curve: float
    twist: float
    phase: float = 0.

    def __post_init__(self):
        if not 2<=len(self.regions)<=32 or len(set(self.regions))!=len(self.regions) or any(type(i) is not int or i<1 for i in self.regions):
            raise ValueError('bundle needs distinct authored rod region identities')
        if not all(math.isfinite(x) for x in (self.eye_length,self.eye_curve,self.twist,self.phase)) or self.eye_length<=0:
            raise ValueError('invalid axial bundle dimensions')

    def radial(self, point):
        x,y,z=map(float,point);t=y/self.eye_length
        return np.array([x-self.eye_curve*(1-t*t),z]),t

    def owner(self, point, active):
        """Unique spatial ownership; exact ties use stable authored rod identity.

        Tie breaking only defines a zero-area shared interface, never a finite
        overlap-volume precedence. Same-material rods retain distinct IDs.
        """
        candidates=set(active)&set(self.regions)
        if not candidates:return None
        radial,t=self.radial(point)
        scores=[]
        for index,region in enumerate(self.regions):
            if region in candidates:
                a=self.phase+self.twist*t+math.pi*index/len(self.regions)
                score=abs(radial[0]*math.cos(a)+radial[1]*math.sin(a))
                scores.append((score,-region,region))
        return max(scores)[2]

    def divider_angles(self):
        """All pairwise bisectors, including fallback where nearest rod is absent."""
        result=set();n=len(self.regions)
        for i in range(n):
            for j in range(i):
                for perpendicular in (0.,math.pi/2):
                    result.add(round(((i+j)*math.pi/(2*n)+perpendicular)%math.pi,14))
        return tuple(sorted(result))

    def divider_value_gradient(self, point, angle):
        """Implicit zero surface and analytic local gradient, for kernel reuse.

        F = Z cos(a(Y)) - (X-centre(Y)) sin(a(Y)). True cut interfaces
        are only those portions inside >=2 competing rods with distinct owners
        on opposite sides. Bubbles mask them; injected dye does not use them.
        """
        radial,t=self.radial(point);a=self.phase+self.twist*t+angle
        x,z=radial;c,s=math.cos(a),math.sin(a)
        da=self.twist/self.eye_length
        dxdy=2*self.eye_curve*float(point[1])/self.eye_length**2
        value=z*c-x*s
        return value,np.array([-s,-z*s*da-dxdy*s-x*c*da,c])

    def divider_intersections(self, origin, direction, angle, t_min, t_max,
                              *, tolerance=1e-10, max_intervals=10000):
        """Isolate crossing roots with analytic derivative bounds (CPU gate).

        Caller supplies a finite geometric interval in bundle-local units.
        No fixed sample grid can silently skip a narrow crossing. Ambiguous
        grazing/coincident roots and exhausted work bounds fail explicitly.
        Not yet a GPU kernel or a promise about float32 traversal.
        """
        from scipy.optimize import brentq
        ro=np.asarray(origin,np.float64);rd=np.asarray(direction,np.float64)
        if ro.shape!=(3,) or rd.shape!=(3,) or not np.isfinite([*ro,*rd,t_min,t_max,tolerance]).all() or t_min<0 or t_max<=t_min or tolerance<=0:
            raise ValueError('invalid finite divider intersection interval')
        if np.linalg.norm(rd)==0:raise ValueError('zero ray direction')
        da=self.twist*rd[1]/self.eye_length
        xsecond=2*self.eye_curve*rd[1]**2/self.eye_length**2
        def value(t):return self.divider_value_gradient(ro+rd*t,angle)[0]
        def bounds(lo,hi):
            mid=(lo+hi)/2;h=(hi-lo)/2
            p=ro+rd*mid;radial,_=self.radial(p)
            xp=rd[0]+2*self.eye_curve*p[1]*rd[1]/self.eye_length**2
            xmax=abs(radial[0])+abs(xp)*h+abs(xsecond)*h*h/2
            zmax=abs(p[2])+abs(rd[2])*h
            xpmax=abs(xp)+abs(xsecond)*h
            second=2*abs(rd[2]*da)+zmax*da*da+abs(xsecond)+2*xpmax*abs(da)+xmax*da*da
            f,g=self.divider_value_gradient(p,angle);first=float(g@rd)
            return f,first,second
        roots=[];todo=[(float(t_min),float(t_max))];work=0
        while todo:
            lo,hi=todo.pop();work+=1
            if work>max_intervals:raise ValueError('divider root isolation exceeded work bound')
            mid=(lo+hi)/2;h=(hi-lo)/2;f,df,dd=bounds(lo,hi)
            if abs(f)>(abs(df)+dd*h)*h:continue
            fl,fh=value(lo),value(hi)
            if abs(df)>dd*h:
                if fl==0:roots.append(lo)
                elif fh==0:roots.append(hi)
                elif np.signbit(fl)!=np.signbit(fh):roots.append(brentq(value,lo,hi,xtol=tolerance,rtol=4*np.finfo(float).eps))
                continue
            if hi-lo<=tolerance:
                raise ValueError('unresolved grazing or coincident divider root')
            todo.extend(((lo,mid),(mid,hi)))
        result=[]
        for t in sorted(roots):
            if result and abs(t-result[-1][0])<=2*tolerance:continue
            point=ro+rd*t;_,gradient=self.divider_value_gradient(point,angle)
            result.append((t,point,gradient/np.linalg.norm(gradient)))
        return result
