"""Plot recorded events only; no ray generation, interpolation or photo edits."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main():
    base=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    history=json.loads((base/'history-42-63/receipt.json').read_text())
    forward=np.load(base/'glass_rod-live-seed147/events.npz')['non_roi']
    camera=np.load(base/'camera-glass_rod/camera-adjoint/events.npz')['roi']
    f=forward.view('<f4');c=camera.view('<f4')
    receipt=json.loads((base/'glass_rod-live-seed147/receipt.json').read_text());roi=receipt['roi']
    counts=np.asarray(history['bounce_segments'][:6]);terminal=forward[:,10]!=0
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,8.5),facecolor='#f6f7fa');grid=fig.add_gridspec(2,2,width_ratios=[1,1.1],wspace=.28,hspace=.45)
    ax=fig.add_subplot(grid[0,0]);ax.set_facecolor('#f6f7fa')
    ax.bar(np.arange(6),counts,color='#225eab',width=.65);ax.set_yscale('log');ax.set_ylim(5,3e7)
    for i,n in enumerate(counts):ax.text(i,n*1.2,f'{n:,}',ha='center',fontsize=10)
    ax.set_xticks(range(6),[str(i+1) for i in range(6)]);ax.set_xlabel('Actual segment number on the path');ax.set_ylabel('Recorded segments (log scale)')
    ax.set_title('Forward energy: all 22 saved populations',loc='left',fontweight='bold',pad=14)
    ax.text(.01,.96,'2,054,632 segments · 0 crossings of the subject ROI',transform=ax.transAxes,va='top',fontsize=11)
    ax=fig.add_subplot(grid[1,0]);ax.set_facecolor('#f6f7fa')
    values=f[terminal,31];assert len(values)==65536 and np.all(values>0)
    ax.hist(values,bins=np.logspace(-11,-2,65),color='#cf7547',alpha=.9)
    ax.set_ylim(0,ax.get_ylim()[1]*1.25)
    ax.set_xscale('log');ax.axvline(.002,color='#9b2030',linestyle='--',lw=1.5)
    ax.text(.002,ax.get_ylim()[1]*.93,' 0.002 cutoff',color='#9b2030',va='top')
    ax.set_xlabel('Recorded final packet throughput');ax.set_ylabel('Terminated paths')
    ax.set_title('Latest population: every path cut off above zero',loc='left',fontweight='bold',pad=14)
    ax.text(.01,.96,'Source seed 63 / bounce RNG 147 · 65,536 positive-valued terminations',transform=ax.transAxes,va='top',fontsize=10)
    ax=fig.add_subplot(grid[:,1]);ax.set_facecolor('#eef1f6')
    a=c[:,12:15]+c[:,20:23]*c[:,23,None];b=c[:,12:15]+c[:,20:23]*c[:,24,None]
    # Exact recorded ROI intervals on existing segments, projected onto X/Z.
    center=np.array([0,0,.18]);segments=np.stack((a,b),axis=1)[:,:, [0,2]]*1000-center[[0,2]]*1000
    glass=camera[:,6].view('<i4')>=0
    for mask,color,label in [(~glass,'#2987ac','Air segments'),(glass,'#8646ad','Dielectric segments')]:
        lines=LineCollection(segments[mask],colors=color,linewidths=.3,alpha=.035,rasterized=True)
        ax.add_collection(lines);ax.plot([],[],color=color,lw=2,label=f'{label}: {int(mask.sum()):,}')
    lo=np.array(roi['bounds_min_m'])[[0,2]]*1000-center[[0,2]]*1000;hi=np.array(roi['bounds_max_m'])[[0,2]]*1000-center[[0,2]]*1000
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
    ax.set_xlabel('X relative to subject center (mm)');ax.set_ylabel('Z relative to subject center (mm)')
    ax.set_title('Camera adjoint paths: 61,102 actual ROI segments',loc='left',fontweight='bold',pad=14)
    ax.legend(loc='lower left',framealpha=.95,fontsize=10)
    ax.text(.02,.98,'X/Z projection of every retained ROI segment\nAdjoint throughput is not emitted forward energy',transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':6})
    fig.suptitle('What the tracers recorded in the studied scene',x=.07,y=.98,ha='left',fontsize=23,fontweight='bold')
    fig.text(.07,.025,'Forward: exact source history, 0 dropped records. Camera: separate 128 × 72 run, 3 sweeps + 16 rows. No NEE connections are shown.',fontsize=10,color='#49515c')
    fig.subplots_adjust(top=.88,bottom=.11,left=.07,right=.96)
    fig.savefig(out/'ray-arrival-diagnostic.png',dpi=160,facecolor=fig.get_facecolor())
    (out/'provenance.json').write_text(json.dumps(dict(forward_history=str(base/'history-42-63/receipt.json'),latest_forward=str(base/'glass_rod-live-seed147/events.npz'),camera=str(base/'camera-glass_rod/camera-adjoint/events.npz'),plot_semantics='All actual camera ROI segments projected X/Z; no invented or continued paths; zero forward intersections is a recorder result, not proof of zero physical illumination.'),indent=2))


if __name__=='__main__':main()
