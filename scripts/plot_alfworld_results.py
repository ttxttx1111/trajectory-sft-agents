"""Plot the recorded, matched ALFWorld development outcomes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
scores = json.loads((ROOT / 'experiments/alfworld/releases/v0.3.0/summary.json').read_text(encoding='utf-8'))['scores']
arms = ['base', 'sft', 'current']
values = [scores[arm]['success_rate_percent'] for arm in arms]
fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
fig.patch.set_facecolor('#fafbfe')
ax.set_facecolor('#fafbfe')
ax.bar(['Base', 'Trajectory SFT', 'SFT + GRPO'], values,
       color=['#9da9bd', '#578be5', '#29a188'], width=0.56, zorder=3)
for index, arm in enumerate(arms):
    ax.text(index, values[index] + 3, f"{values[index]:.2f}%", ha='center',
            fontsize=16, fontweight='bold', color='#243348')
    ax.text(index, max(values[index] / 2, 1.9), f"{scores[arm]['successes']}/{scores[arm]['games']}",
            ha='center', va='center', color='white', fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_ylabel('Complete-task success (%)', fontsize=11)
ax.set_title('ALFWorld: small-model agent post-training', loc='left', fontsize=17, fontweight='bold', pad=23)
ax.text(0, 1.02, 'Qwen2.5-1.5B  |  600 verified expert trajectories  |  Single GPU',
        transform=ax.transAxes, fontsize=10, color='#566275')
ax.grid(axis='y', alpha=0.18, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#b8c0ca')
ax.tick_params(axis='both', labelsize=11, length=0, pad=8)
fig.text(0.98, -0.02, 'Matched development evaluation; one complete episode per game.',
         ha='right', fontsize=9, color='#566275')
for extension in ['png', 'svg']:
    fig.savefig(ROOT / f'assets/alfworld-sft-grpo.{extension}', dpi=160, bbox_inches='tight')
svg = ROOT / 'assets/alfworld-sft-grpo.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
               encoding='utf-8', newline='\n')
