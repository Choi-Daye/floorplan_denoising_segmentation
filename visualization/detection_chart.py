import matplotlib.pyplot as plt
import numpy as np

# 데이터 (clean(+1px)_clean 추가)
datasets = ["raw_raw", "raw_clean", "clean_clean", "gt_clean", "clean_clean_+1px"]
mAP50 = [0.209, 0.1032, 0.0598, 0.1141, 0.0925]
door = [0.2789, 0.1744, 0.1094, 0.1810, 0.2132]
window = [0.2396, 0.0826, 0.0420, 0.0964, 0.0322]
wall = [0.1085, 0.0528, 0.0279, 0.0649, 0.0320]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 5개 데이터셋에 맞춰 figsize 증가

# 왼쪽: mAP50 비교
x = np.arange(len(datasets))
width = 0.6

bars1 = ax1.bar(x, mAP50, width, color=['#F2CB61']*len(datasets), 
                edgecolor='black', linewidth=1)
ax1.set_ylabel('mAP50', fontsize=12, color='black')
ax1.set_title('mAP50 Comparison', fontsize=14, fontweight='bold', color='black')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=10, rotation=45, ha='right')  # 라벨 회전으로 가독성 향상
ax1.set_ylim(0, 0.25)
ax1.grid(axis='y', alpha=0.3, color='black')

# 값 라벨 추가
for bar, val in zip(bars1, mAP50):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 오른쪽: 클래스별 AP50 (5개 데이터셋용 width 조정)
width2 = 0.22
bars2_door = ax2.bar(x - width2*1.5, door, width2, label='Door', color='#CC3D3D', edgecolor='black')
bars2_win = ax2.bar(x - width2/2, window, width2, label='Window', color='#6B9900', edgecolor='black')
bars2_wall = ax2.bar(x + width2/2, wall, width2, label='Wall', color='#4374D9', edgecolor='black')

ax2.set_ylabel('AP50', fontsize=12, color='black')
ax2.set_title('Per-Class AP50', fontsize=14, fontweight='bold', color='black')
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=10, rotation=45, ha='right')
ax2.set_ylim(0, 0.30)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, color='black')

# 값 라벨 추가 (폰트 크기 조정)
for bars, vals in [(bars2_door, door), (bars2_win, window), (bars2_wall, wall)]:
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 스타일 설정
plt.style.use('default')
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')
ax2.set_facecolor('white')
plt.tight_layout(pad=2.0)

plt.show()
