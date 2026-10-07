"""Annexe F.6 : exp39 (G=16, 8 tâches × 16) vs contrôles, et vs curriculums. Légende sous la figure."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d39=json.load(open('docs/rapport/figures/exp39_control.json')); cur=json.load(open('docs/rapport/figures/curricula.json'))
INK,MUTED="#0b0b0b","#52514e"
def roll(x,y,w=47,xmax=1300):
    x,y=np.array(x,float),np.array(y,float); g=np.arange(w/2,xmax,10); return g,np.array([y[(x>=c-w/2)&(x<c+w/2)].mean() if ((x>=c-w/2)&(x<c+w/2)).sum()>=3 else np.nan for c in g])
plt.rcParams.update({"font.size":8.5,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.titlesize":9})
def make(series, out, ymax):
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.9),constrained_layout=True)
    for name,(col,ls,xt,yt,xs,ent,kl) in series.items():
        m=np.array(xt)<=1300; axes[0].plot(np.array(xt)[m],np.array(yt)[m],color=col,ls=ls,lw=1.3,marker=".",ms=3,label=name)
        g,y=roll(xs,ent); axes[1].plot(g,y,color=col,ls=ls,lw=1.3)
        g,y=roll(xs,kl); axes[2].plot(g,np.clip(y,1e-4,1e3),color=col,ls=ls,lw=1.3)
    axes[0].set_ylim(0,ymax); axes[0].set_ylabel("pass@1 test (%)"); axes[0].set_title("(a) pass@1 test",loc="left")
    axes[1].set_ylim(0,1.3); axes[1].set_ylabel("entropie (moy. glissante)"); axes[1].set_title("(b) entropie",loc="left")
    axes[2].set_yscale("log"); axes[2].set_ylim(3e-4,1e3); axes[2].set_ylabel("KL (moy. glissante)"); axes[2].set_title("(c) divergence KL",loc="left")
    for ax in axes:
        ax.set_xlim(0,1300); ax.set_xlabel("mises à jour"); ax.grid(axis="y",color="#e6e5e1",lw=0.6); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=len(l),frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,-0.07))
    fig.savefig(out+'.pdf',bbox_inches="tight"); fig.savefig(out+'.png',dpi=200,bbox_inches="tight")
# figure 1 : contrôles
s1={}
for name,r in d39.items():
    k=name.split()[0]; col={"exp39":"#2a78d6","exp36":"#e87ba4","exp25":MUTED}[k]; ls="-" if k!="exp25" else (0,(4,2))
    te=np.array(r["test"]); tr=np.array(r["train"])
    label={"exp39":"G=16, 8 tâches × 16 (exp39)","exp36":"G=16, 4 tâches × 16 (contrôle)","exp25":"G=8, 8 tâches × 8 (référence)"}[k]
    s1[label]=(col,ls,te[:,0],te[:,2],tr[:,0],tr[:,4],tr[:,3])
make(s1,'docs/rapport/figures/fig_exp39_control',70)
# figure 2 : curriculums
k39=[k for k in d39 if k.startswith("exp39")][0]; tr=np.array(d39[k39]["train"]); te=np.array(d39[k39]["test"])
s2={"G=16, 8 tâches × 16, sans curriculum (exp39)":("#2a78d6","-",te[:,0],te[:,2],tr[:,0],tr[:,4],tr[:,3])}
for key,label,col in [("horizon","Horizon","#eb6834"),("budget","Budget","#1baf7a"),("prof_auto","Profondeur","#eda100")]:
    ev=cur[key]["evals"]; t=np.array(cur[key]["train"]); spe=cur[key]["spe"]
    s2[label]=(col,"-",[e["step"] for e in ev],[e["p1"] for e in ev],t[:,0]*spe,t[:,3],t[:,2])
make(s2,'docs/rapport/figures/fig_exp39_vs_curricula',80)
print("ok")
