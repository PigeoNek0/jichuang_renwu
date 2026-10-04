"""
电路① RC 低通滤波电路 —— 理论计算 + PySpice 仿真 + 对比表
================================================================
跑完这个脚本会得到：
  images/rc_transient.png   方波输入/输出瞬态波形（含 tau 标注）
  images/rc_bode.png        波特图（幅频 + 相频，含 -3dB 标注）
  终端打印的「手算 vs 仿真」对比表  -> 直接粘进 README

运行：
    python rc_filter.py
"""
import os
import sys

# ---------------------------------------------------------------- 环境
# 如果在 DSH 沙箱环境里跑，取消下面两行的注释（必须在 import PySpice 之前）
# sys.path.insert(0, r"D:\worktable\日常\_pyspice_env")
# import bootstrap_pyspice

import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Simulation import CircuitSimulator
from PySpice.Unit import *

try:
    import matplotlib
    matplotlib.use("Agg")            # 无界面环境也能存图
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("缺少 matplotlib,请先安装:pip install matplotlib")

# ---------------------------------------------------------------- 参数（自定）
R = 1e3          # ohm
C = 100e-9       # F
V_PULSE = 5.0    # V
F_SQUARE = 250.0 # Hz  方波频率（周期 4 ms = 40 tau，能看完整充放电）
V_AMP = 1.0      # V   AC 分析用小信号幅度

TAU = R * C
FC = 1.0 / (2 * np.pi * R * C)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

ng = NgSpiceShared.new_instance()

print("=" * 62)
print("电路① RC 低通滤波")
print("=" * 62)
print(f"参数        : R = {R:g} ohm, C = {C*1e9:g} nF")
print(f"理论 tau    : R*C = {TAU*1e6:.2f} us")
print(f"理论 fc     : 1/(2*pi*R*C) = {FC:.2f} Hz")
print()

# ================================================================ 1. 瞬态
period = 1.0 / F_SQUARE
c1 = Circuit("RC low pass transient")
c1.PulseVoltageSource("in", "src", c1.gnd,
                      initial_value=0 @ u_V, pulsed_value=V_PULSE @ u_V,
                      delay_time=(period / 4) @ u_s, rise_time=1 @ u_us, fall_time=1 @ u_us,
                      pulse_width=(period / 2) @ u_s, period=period @ u_s)
c1.R(1, "src", "out", R @ u_Ohm)
c1.C(1, "out", c1.gnd, C @ u_F)

sim1 = CircuitSimulator.factory(c1, simulator="ngspice-shared", ngspice_shared=ng)
an1 = sim1.transient(step_time=(TAU / 100) @ u_s, end_time=(2 * period) @ u_s)
t = np.asarray(an1.time)
v_out = np.asarray(an1.nodes["out"])
v_src = np.asarray(an1.nodes["src"])

# ---- 定位充电区间
# ⚠️ 坑：不要用「相邻采样点跳变量」去找阶跃沿。ngspice 会在上升沿内部自动插入很多小步长
#    （实测 0V -> 0.44 -> 0.75 -> 1.38 -> 2.63 -> 5.0V），相邻点跳变永远到不了阈值，判据必然失效。
#    网表里的 delay_time 是我们自己设的、精确已知，直接用它作为充电区间起点最可靠。
t_step = period / 4.0          # = PulseVoltageSource 的 delay_time
mask = (t >= t_step) & (t < t_step + period / 2.0)
seg_t = t[mask] - t_step
seg_v = v_out[mask]
if len(seg_t) < 10:
    sys.exit("充电区间采样点太少，检查脉冲源参数与 step_time")

# ---- 方法 A：63.2% 读数法
target = 0.632 * V_PULSE
k = int(np.argmax(seg_v >= target))
tau_a = seg_t[k]

# ---- 方法 B：最小二乘拟合 ln(1 - v/V) = -t/tau
m = (seg_v > 0.01 * V_PULSE) & (seg_v < 0.99 * V_PULSE)
slope = np.polyfit(seg_t[m], np.log(1 - seg_v[m] / V_PULSE), 1)[0]
tau_b = -1.0 / slope

print("-" * 62)
print("[瞬态] 时间常数 tau")
print(f"  手算            : {TAU*1e6:8.2f} us")
print(f"  仿真 63.2% 读数法: {tau_a*1e6:8.2f} us   误差 {abs(tau_a-TAU)/TAU*100:5.2f}%")
print(f"  仿真 最小二乘拟合: {tau_b*1e6:8.2f} us   误差 {abs(tau_b-TAU)/TAU*100:5.2f}%")
print()

# ---- 画瞬态图
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t * 1e3, v_src, "--", color="tab:gray", lw=1.2, label="Vin (square wave)")
ax.plot(t * 1e3, v_out, "-", color="tab:blue", lw=2, label="Vout (across C)")
ax.axhline(target, color="tab:red", ls=":", lw=1.2)
ax.annotate(f"63.2% = {target:.3f} V", xy=(t[-1] * 1e3 * 0.02, target),
            xytext=(t[-1] * 1e3 * 0.02, target + 0.25), color="tab:red", fontsize=9)
ax.plot([(t_step + tau_a) * 1e3], [target], "o", color="tab:red", ms=6)
ax.annotate(f"tau = {tau_a*1e6:.1f} us", xy=((t_step + tau_a) * 1e3, target),
            xytext=((t_step + tau_a) * 1e3 + 0.15, target - 1.1),
            color="tab:red", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (V)")
ax.set_title(f"RC low-pass transient response  (R={R/1e3:g} kohm, C={C*1e9:g} nF, tau={TAU*1e6:.0f} us)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rc_transient.png"), dpi=150)
plt.close(fig)

# ================================================================ 2. AC 扫频
c2 = Circuit("RC low pass ac")
c2.SinusoidalVoltageSource("in", "src", c2.gnd, amplitude=V_AMP @ u_V,
                           frequency=1 @ u_kHz, ac_magnitude=V_AMP @ u_V)
c2.R(1, "src", "out", R @ u_Ohm)
c2.C(1, "out", c2.gnd, C @ u_F)

sim2 = CircuitSimulator.factory(c2, simulator="ngspice-shared", ngspice_shared=ng)
an2 = sim2.ac(start_frequency=1 @ u_Hz, stop_frequency=1 @ u_MHz,
              number_of_points=100, variation="dec")
f = np.asarray(an2.frequency)
h = np.asarray(an2.nodes["out"]) / V_AMP
mag_db = 20 * np.log10(np.abs(h))
phase = np.angle(h, deg=True)

idx3 = int(np.argmin(np.abs(mag_db + 3.0103)))
f_meas = f[idx3]
# 高频段斜率（取最后两个十倍频点）
slope_meas = (mag_db[-20] - mag_db[-40]) / (np.log10(f[-20]) - np.log10(f[-40]))

print("-" * 62)
print("[AC 扫频] 截止频率 fc")
print(f"  手算            : {FC:8.2f} Hz")
print(f"  仿真 -3dB 点    : {f_meas:8.2f} Hz   误差 {abs(f_meas-FC)/FC*100:5.2f}%")
print(f"  仿真 -3dB 处相位: {phase[idx3]:8.2f} deg  (理论 -45)")
print(f"  高频段斜率      : {slope_meas:8.2f} dB/decade  (理论 -20)")
print(f"  低频增益        : {mag_db[0]:8.4f} dB  (理论 0)")
print()

# ---- 画波特图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.semilogx(f, mag_db, "-", color="tab:blue", lw=2)
ax1.axhline(-3.0103, color="tab:red", ls=":", lw=1.2)
ax1.axvline(f_meas, color="tab:red", ls=":", lw=1.2)
ax1.plot([f_meas], [mag_db[idx3]], "o", color="tab:red", ms=6)
ax1.annotate(f"f_c = {f_meas:.1f} Hz\n(hand calc {FC:.1f} Hz)",
             xy=(f_meas, mag_db[idx3]), xytext=(f_meas * 0.5, -18),
             color="tab:red", fontsize=9,
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax1.set_ylabel("|H| (dB)")
ax1.set_title("RC low-pass Bode plot")
ax1.grid(which="both", alpha=0.3)

ax2.semilogx(f, phase, "-", color="tab:green", lw=2)
ax2.axhline(-45, color="tab:red", ls=":", lw=1.2)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Phase (deg)")
ax2.grid(which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rc_bode.png"), dpi=150)
plt.close(fig)

# ================================================================ 3. 对比表
def err(a, b):
    return abs(a - b) / abs(b) * 100

print("=" * 62)
print("对比表（粘进 README，再自己补「误差来源分析」一列）")
print("=" * 62)
rows = [
    ("tau (63.2% 读数法)", f"{TAU*1e6:.2f} us", f"{tau_a*1e6:.2f} us", err(tau_a, TAU)),
    ("tau (最小二乘拟合)", f"{TAU*1e6:.2f} us", f"{tau_b*1e6:.2f} us", err(tau_b, TAU)),
    ("截止频率 fc", f"{FC:.2f} Hz", f"{f_meas:.2f} Hz", err(f_meas, FC)),
    ("-3dB 处相位", "-45.00 deg", f"{phase[idx3]:.2f} deg", err(phase[idx3], -45)),
    ("高频斜率", "-20.00 dB/dec", f"{slope_meas:.2f} dB/dec", err(slope_meas, -20)),
]
print(f"{'指标':<20}{'手算值':>16}{'仿真值':>18}{'相对误差':>12}")
print("-" * 62)
for name, th, sim, e in rows:
    print(f"{name:<20}{th:>16}{sim:>18}{e:>11.2f}%")
print("=" * 62)
print(f"图片已保存到: {OUT_DIR}")
