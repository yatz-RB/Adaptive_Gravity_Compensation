import numpy as np
import matplotlib.pyplot as plt

# 读取 dx.txt 数据
# dm_values = np.loadtxt("data/dm.txt")
# dx_values = np.loadtxt("data/dx.txt")
# dy_values = np.loadtxt("data/dy.txt")
# dz_values = np.loadtxt("data/dz.txt")

j0_v = np.loadtxt("data/1j0.txt")[:2000]
j1_v = np.loadtxt("data/1j1.txt")[:2000]
j2_v = np.loadtxt("data/1j2.txt")[:2000]
j3_v = np.loadtxt("data/1j3.txt")[:2000]
j4_v = np.loadtxt("data/1j4.txt")[:2000]
j5_v = np.loadtxt("data/1j5.txt")[:2000]


# 读取期望值
# dm_expected = 1.0929584003946624
# dx_expected = 0.08294498400995304
# dy_expected = 0.09120529248231629
# dz_expected = 0.04908612551777656

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(j0_v, color = '#FF0000', label="joint1")
plt.plot(j1_v, color = '#FFFF00', label="joint2")
plt.plot(j2_v, color = '#00FF00', label="joint3")
plt.plot(j3_v, color = '#00FFFF', label="joint4")
plt.plot(j4_v, color = '#0000FF', label="joint5")
plt.plot(j5_v, color = '#FF00FF', label="joint6")
# plt.plot(dm_values, color = '#0000FF', label="joint5")

plt.axhline(y=0, color='black', linestyle='--', label="0",alpha = 0.7)

# 添加标签和图例
plt.xlabel("数据点")
plt.ylabel("关节力矩差值")
plt.title("变负载建模计算关节力矩与实际关节力矩的差值")
plt.legend()
plt.grid()

plt.savefig("changed_comparison.png", dpi=300, bbox_inches='tight')  # 保存为 PNG，300 DPI
# 显示图像
plt.show()
