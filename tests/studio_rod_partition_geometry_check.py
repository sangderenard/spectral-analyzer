"""CPU geometry experiment, NOT transport or a chosen partition policy.

Shows why equal total rod area and equal shares of each shared zone differ.
Uses the exact axial generator's superellipse cross-section equation.
"""
import json
import numpy as np


def measure(count, angles=2880, radial_bins=128):
    angle=(np.arange(angles)+.37)*2*np.pi/angles
    radius=np.sqrt((np.arange(radial_bins)+.5)/radial_bins)
    x=(radius[:,None]*np.cos(angle)).ravel()
    z=(radius[:,None]*np.sin(angle)).ravel()
    axes=np.arange(count)*np.pi/count
    u=x[:,None]*np.cos(axes)+z[:,None]*np.sin(axes)
    v=-x[:,None]*np.sin(axes)+z[:,None]*np.cos(axes)
    active=(u**4+(v/.25)**4)<=1
    owner=np.argmax(np.where(active,np.abs(u),-np.inf),axis=1)
    owner[~active.any(axis=1)]=-1
    totals=np.bincount(owner[owner>=0],minlength=count)
    codes=active@(1<<np.arange(count))
    worst=0.;example=None
    for mask in np.unique(codes):
        ids=np.flatnonzero(mask&(1<<np.arange(count)))
        if len(ids)<2:continue
        votes=np.bincount(owner[codes==mask],minlength=count)[ids]
        if votes.sum()<=100:continue
        imbalance=(votes.max()-votes.min())/(votes.sum()/len(ids))
        if imbalance>worst:
            worst=float(imbalance);example={'participants':ids.tolist(),'assigned_samples':votes.tolist()}
    return {'rods':count,'total_samples_per_rod':totals.tolist(),
            'worst_shared_zone_range_over_mean':worst,'example':example}


def run():
    rows=[measure(n) for n in (2,3,4,5,6)]
    for row in rows:
        assert max(row['total_samples_per_rod'])==min(row['total_samples_per_rod'])
    assert rows[0]['worst_shared_zone_range_over_mean']==0
    assert rows[2]['worst_shared_zone_range_over_mean']>1
    print(json.dumps(rows,indent=2))

if __name__=='__main__':run()
