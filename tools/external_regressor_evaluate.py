"""Receptor regressors on unseen chemistry: predicted pKi against measured pChEMBL.

For A2A, D2, HT2A and SERT the panel predicts a potency directly rather than a probability, so
the comparison needs no threshold and no label rule at all. This is the most direct external
test available: a number the model predicts against the same number measured in a laboratory,
on compounds absent from both the training index and the applicability-domain reference.
"""
import csv, math, random, sys, warnings
from collections import defaultdict
sys.path.insert(0, "."); sys.path.insert(0, "src/sai_cns")
warnings.filterwarnings("ignore")
import app as A

def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
    while i < len(o):
        j=i
        while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j+=1
        a=(i+j)/2.0+1.0
        for k in range(i,j+1): r[o[k]]=a
        i=j+1
    return r

def spearman(x,y):
    rx,ry=ranks(x),ranks(y); n=len(x); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx=sum((a-mx)**2 for a in rx)**.5; dy=sum((b-my)**2 for b in ry)**.5
    return num/(dx*dy) if dx and dy else float("nan")

def ci(x,y,n=1000,seed=0):
    rng=random.Random(seed); m=len(x); v=[]
    for _ in range(n):
        idx=[rng.randrange(m) for _ in range(m)]
        s=spearman([x[i] for i in idx],[y[i] for i in idx])
        if s==s: v.append(s)
    v.sort(); return (v[int(.025*len(v))], v[int(.975*len(v))]) if v else (float('nan'),)*2

rows=list(csv.DictReader(open("results/tables/chembl_holdout_set.csv",encoding="utf-8")))
by=defaultdict(list)
for r in rows: by[r["endpoint"]].append(r)
m=A.load_models.__wrapped__() if hasattr(A.load_models,"__wrapped__") else A.load_models()

print(f"  {'endpoint':8s} {'n':>5s} {'Spearman':>9s} {'95% CI':>20s} {'MAE':>7s} {'bias':>7s}")
print("  "+"-"*62)
out=[]
for ep in ("A2A","D2","HT2A","SERT"):
    rs=by.get(ep,[])
    if not rs: continue
    res=A.predict_all_batch([r["smiles"] for r in rs], m)
    pred,meas=[],[]
    for r,d in zip(rs,res):
        if d is None: continue
        v=(d.get("receptors") or {}).get(ep)
        if v is None: continue
        pred.append(float(v)); meas.append(float(r["pchembl"]))
    if len(pred)<30:
        print(f"  {ep:8s} {len(pred):>5d}  too few"); continue
    rho=spearman(pred,meas); lo,hi=ci(pred,meas)
    mae=sum(abs(a-b) for a,b in zip(pred,meas))/len(pred)
    bias=sum(a-b for a,b in zip(pred,meas))/len(pred)
    print(f"  {ep:8s} {len(pred):>5d} {rho:>+9.3f}  [{lo:+.3f}, {hi:+.3f}] {mae:>7.2f} {bias:>+7.2f}")
    out.append((ep,len(pred),rho,lo,hi,mae,bias))
with open("results/tables/chembl_holdout_regressors.csv","w",newline="",encoding="utf-8") as fh:
    w=csv.writer(fh); w.writerow(["endpoint","n_unseen","spearman_rho","ci_low","ci_high","mae_log_units","mean_bias"])
    for r in out: w.writerow([r[0],r[1],round(r[2],4),round(r[3],4),round(r[4],4),round(r[5],3),round(r[6],3)])
print("\n  MAE and bias are in log units. Positive bias = the panel predicts more potent than measured.")
