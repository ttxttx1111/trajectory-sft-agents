"""Regenerate the published result figure from saved analysis (requires matplotlib)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
analysis = json.loads((ROOT/'results/reference/analysis.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), sharey=True)
fig.patch.set_facecolor('#f7f9fc')
for ax, split, label in zip(axes, ['dev','test'], ['Development','Test']):
    data=analysis['splits'][split]
    counts=[data[a]['summary']['successes'] for a in ['base','tuned']]
    rates=[v/40*100 for v in counts]
    ax.set_facecolor('#f7f9fc')
    ax.bar([0,1], rates, width=.55, color=['#98a8bb','#306ce7'], zorder=3)
    for x,rate,count in zip([0,1],rates,counts):
        ax.text(x,rate+3,f'{rate:.1f}%',ha='center',va='bottom',weight='bold',fontsize=19,color='#16243a')
        ax.text(x,rate+15,f'{count}/40 tasks',ha='center',fontsize=11,color='#59697f')
    ax.set_xticks([0,1],['Qwen3-1.7B','+ E017 LoRA'])
    ax.set_ylim(0,100)
    ax.set_xlim(-.6,1.6)
    ax.set_yticks([0,25,50,75,100])
    ax.grid(axis='y',alpha=.16,zorder=0)
    ax.set_title(f'{label}  |  +{data["paired"]["gain_percentage_points"]:g} pp',loc='left',pad=17,fontsize=16,weight='bold',color='#16243a')
    for side in ['top','right','left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#c9d1df')
    ax.tick_params(axis='both',length=0,pad=9)
axes[0].set_ylabel('Whole-task success (%)',labelpad=12)
fig.suptitle('Trajectory SFT for multi-turn tool use',x=.065,y=.97,ha='left',fontsize=23,weight='bold',color='#16243a')
fig.text(.065,.884,'120 training trajectories  /  1,129 targets  /  ~70 min on one RTX 5090',fontsize=12,color='#536479')
fig.text(.065,.038,'BFCL multi_turn_base · fixed in-domain 120/40/40 split · E017, one seed',fontsize=10.5,color='#536479')
fig.subplots_adjust(left=.08,right=.975,top=.75,bottom=.17,wspace=.19)
out=ROOT/'assets'
out.mkdir(exist_ok=True)
fig.savefig(out/'e017-results.png',dpi=180,facecolor=fig.get_facecolor())
fig.savefig(out/'e017-results.svg',facecolor=fig.get_facecolor(),metadata={'Date':None})
print('Wrote assets/e017-results.png and .svg')
