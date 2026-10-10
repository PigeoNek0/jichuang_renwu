"""
电路① RC 低通滤波电路 —— 理论计算 + PySpice 仿真 + 对比表
================================================================
跑完这个脚本会得到：
  images/rc_transient.png   方波输入/输出瞬态波形（含 tau 标注）
  images/rc_bode.png        波特图（幅频 + 相频，含 -3dB 标注）
  终端打印的「手算 vs 仿真」对比表     -> 直接粘进 README
  以及两个误差来源的排除实验（改变采样步长 / 扫频点数，看误差是否随之变化）

运行：
    python rc_filter.py
"""
import os
import sys

# 输出重定向到文件/管道时, Windows 默认用 GBK, 打印特殊字符会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# matplotlib 默认缓存目录在用户目录, 部分环境不可写会报警告; 指到项目内（已在 .gitignore）
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "mplcache"))

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
PERIOD = 1.0 / F_SQUARE
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

ng = NgSpiceShared.new_instance()    # 整个脚本只建一次，全程复用


def err(a, b):
    return abs(a - b) / abs(b) * 100


# ================================================================ 仿真封装
def run_transient(step_time, rise_time=1e-6):
    """跑一次瞬态分析，返回 (t, v_src, v_out, seg_t, seg_v)。

    seg_* 是充电区间（已把时间原点挪到阶跃处），供提取 tau 用。
    rise_time 默认 1 us —— 这个上升沿会给「63.2% 读数法」带来一个与采样步长
    无关的固定误差，见后面实验 1b 的验证。
    """
    c = Circuit("RC low pass transient")
    c.PulseVoltageSource("in", "src", c.gnd,
                         initial_value=0 @ u_V, pulsed_value=V_PULSE @ u_V,
                         delay_time=(PERIOD / 4) @ u_s, rise_time=rise_time @ u_s,
                         fall_time=1 @ u_us, pulse_width=(PERIOD / 2) @ u_s,
                         period=PERIOD @ u_s)
    c.R(1, "src", "out", R @ u_Ohm)
    c.C(1, "out", c.gnd, C @ u_F)

    sim = CircuitSimulator.factory(c, simulator="ngspice-shared", ngspice_shared=ng)
    an = sim.transient(step_time=step_time @ u_s, end_time=(2 * PERIOD) @ u_s)
    t = np.asarray(an.time)
    v_out = np.asarray(an.nodes["out"])
    v_src = np.asarray(an.nodes["src"])

    # ⚠️ 坑：不要用「相邻采样点跳变量」去找阶跃沿。ngspice 会在上升沿内部自动插入很多小步长
    #    （实测 0V -> 0.44 -> 0.75 -> 1.38 -> 2.63 -> 5.0V），相邻点跳变永远到不了阈值，判据必然失效。
    #    网表里的 delay_time 是我们自己设的、精确已知，直接用它作为充电区间起点最可靠。
    t_start = PERIOD / 4.0            # = PulseVoltageSource 的 delay_time
    mask = (t >= t_start) & (t < t_start + PERIOD / 2.0)
    seg_t = t[mask] - t_start
    seg_v = v_out[mask]
    if len(seg_t) < 10:
        sys.exit("充电区间采样点太少，检查脉冲源参数与 step_time")
    return t, v_src, v_out, seg_t, seg_v


def tau_from_segment(seg_t, seg_v):
    """从充电区间提取 tau：返回 (63.2% 读数法, 最小二乘拟合)。"""
    # 方法 A：63.2% 读数法 —— 取「第一个 ≥63.2% 稳态值」的采样点
    target = 0.632 * V_PULSE
    k = int(np.argmax(seg_v >= target))
    tau_a = seg_t[k]
    # 方法 B：最小二乘拟合 ln(1 - v/V) = -t/tau
    m = (seg_v > 0.01 * V_PULSE) & (seg_v < 0.99 * V_PULSE)
    slope = np.polyfit(seg_t[m], np.log(1 - seg_v[m] / V_PULSE), 1)[0]
    tau_b = -1.0 / slope
    return tau_a, tau_b


def run_ac(n_points):
    """跑一次 AC 扫频，返回 (f, mag_db, phase)。"""
    c = Circuit("RC low pass ac")
    c.SinusoidalVoltageSource("in", "src", c.gnd, amplitude=V_AMP @ u_V,
                              frequency=1 @ u_kHz, ac_magnitude=V_AMP @ u_V)
    c.R(1, "src", "out", R @ u_Ohm)
    c.C(1, "out", c.gnd, C @ u_F)

    sim = CircuitSimulator.factory(c, simulator="ngspice-shared", ngspice_shared=ng)
    an = sim.ac(start_frequency=1 @ u_Hz, stop_frequency=1 @ u_MHz,
                number_of_points=n_points, variation="dec")
    f = np.asarray(an.frequency)
    h = np.asarray(an.nodes["out"]) / V_AMP
    return f, 20 * np.log10(np.abs(h)), np.angle(h, deg=True)


def fc_from_sweep(f, mag_db, phase):
    """取最接近 -3dB 的扫频点，返回 (f_c, 该点相位, 下标)。

    ⚠️ 这就是误差的来源：它只在**采样点上**找，不在点之间插值。
    实测网格点间距 2.33%（100 点/十倍频），所以读数最多能偏 ±1.16%。
    """
    idx = int(np.argmin(np.abs(mag_db + 3.0103)))
    return f[idx], phase[idx], idx


def fc_by_interpolation(f, mag_db):
    """在 -3 dB 的两个相邻网格点之间做对数插值，返回 (f_c, 下标)。

    用来验证「fc 的误差来自取最近格点而非插值」——如果插值后误差塌掉，
    就说明原因确实在采样离散，而不是模型或数值上的别的什么。
    """
    db_target = -3.0103
    s = np.sign(mag_db - db_target)
    k = int(np.argmax(s[:-1] != s[1:]))       # 找出变号的位置
    f1, f2 = np.log10(f[k]), np.log10(f[k + 1])
    d1, d2 = mag_db[k], mag_db[k + 1]
    frac = (db_target - d1) / (d2 - d1)       # 线性插值比例
    return 10 ** (f1 + frac * (f2 - f1)), k


# ================================================================ 0. 标题
print("=" * 62)
print("电路① RC 低通滤波")
print("=" * 62)
print(f"参数        : R = {R:g} ohm, C = {C*1e9:g} nF")
print(f"理论 tau    : R*C = {TAU*1e6:.2f} us")
print(f"理论 fc     : 1/(2*pi*R*C) = {FC:.2f} Hz")
print()

# ================================================================ 1. 瞬态
STEP_BASE = TAU / 100.0                # 基准步长 = tau/100 = 1 us
t, v_src, v_out, seg_t, seg_v = run_transient(STEP_BASE)
tau_a, tau_b = tau_from_segment(seg_t, seg_v)

print("-" * 62)
print("[瞬态] 时间常数 tau")
print(f"  手算            : {TAU*1e6:8.2f} us")
print(f"  仿真 63.2% 读数法: {tau_a*1e6:8.2f} us   误差 {err(tau_a, TAU):5.2f}%")
print(f"  仿真 最小二乘拟合: {tau_b*1e6:8.2f} us   误差 {err(tau_b, TAU):5.2f}%")
print()

# ---- 画瞬态图
target = 0.632 * V_PULSE
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t * 1e3, v_src, "--", color="tab:gray", lw=1.2, label="Vin (square wave)")
ax.plot(t * 1e3, v_out, "-", color="tab:blue", lw=2, label="Vout (across C)")
ax.axhline(target, color="tab:red", ls=":", lw=1.2)
ax.annotate(f"63.2% = {target:.3f} V", xy=(t[-1] * 1e3 * 0.02, target),
            xytext=(t[-1] * 1e3 * 0.02, target + 0.25), color="tab:red", fontsize=9)
ax.plot([(PERIOD / 4.0 + tau_a) * 1e3], [target], "o", color="tab:red", ms=6)
ax.annotate(f"tau = {tau_a*1e6:.1f} us", xy=((PERIOD / 4.0 + tau_a) * 1e3, target),
            xytext=((PERIOD / 4.0 + tau_a) * 1e3 + 0.15, target - 1.1),
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
N_POINTS_BASE = 100
f, mag_db, phase = run_ac(N_POINTS_BASE)
f_meas, phase_meas, idx3 = fc_from_sweep(f, mag_db, phase)
# 高频段斜率（取最后两个十倍频点）
slope_meas = (mag_db[-20] - mag_db[-40]) / (np.log10(f[-20]) - np.log10(f[-40]))

print("-" * 62)
print("[AC 扫频] 截止频率 fc")
print(f"  手算            : {FC:8.2f} Hz")
print(f"  仿真 -3dB 点    : {f_meas:8.2f} Hz   误差 {err(f_meas, FC):5.2f}%")
print(f"  仿真 -3dB 处相位: {phase_meas:8.2f} deg  (理论 -45)")
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

# ================================================================ 3. 误差来源的排除实验
print("=" * 62)
print("[排除实验] 改变一个变量, 看误差是否随之变化 -> 确认或排除某条原因")
print("=" * 62)

# ---- 实验 1a：缩小采样步长
# 如果 tau 的读数误差纯粹来自步长量化，步长缩到 1/10，误差也该缩到 1/10
STEP_FINE = TAU / 1000.0
_, _, _, seg_t_f, seg_v_f = run_transient(STEP_FINE)
tau_a_f, tau_b_f = tau_from_segment(seg_t_f, seg_v_f)
ratio = err(tau_a_f, TAU) / err(tau_a, TAU)
print(f"实验1a 采样步长 {STEP_BASE*1e6:g} us -> {STEP_FINE*1e6:g} us（tau/100 -> tau/1000）")
print(f"       63.2% 读数法误差: {err(tau_a, TAU):.2f}%  ->  {err(tau_a_f, TAU):.2f}%"
      f"   （变成 {ratio:.2f} 倍）")
print(f"       最小二乘拟合误差: {err(tau_b, TAU):.2f}%  ->  {err(tau_b_f, TAU):.2f}%"
      f"   （本来就不随步长变）")
# 判据：步长缩到 1/10，纯量化误差也应缩到约 0.10 倍；缩不到就说明还有别的成分
if ratio > 0.30:
    print(f"       ⚠️ 步长缩到 1/10，误差只缩到 {ratio:.2f} 倍（纯量化应为 ~0.10 倍）")
    print(f"       => 读数误差 = 步长量化（随步长缩） + 一个与步长无关的固定成分（不缩）")
else:
    print(f"       -> 误差随步长按比例缩小 => 确认来自采样步长量化 ✓")
print()

# ---- 实验 1b：把脉冲源上升沿从 1 us 压到 1 ns（步长保持 tau/1000）
# 若实验 1a 剩下的那个固定误差来自源的 1 us 上升沿，它应该随之消失
RISE_FINE = 1e-9
_, _, _, seg_t_r, seg_v_r = run_transient(STEP_FINE, rise_time=RISE_FINE)
tau_a_r, _ = tau_from_segment(seg_t_r, seg_v_r)
print(f"实验1b 脉冲源上升沿 1 us -> 1 ns（步长保持 tau/1000）")
print(f"       63.2% 读数法误差: {err(tau_a_f, TAU):.2f}%  ->  {err(tau_a_r, TAU):.2f}%")
if err(tau_a_r, TAU) < err(tau_a_f, TAU) * 0.6:
    print(f"       -> 固定误差随之显著减小 => 确认它来自脉冲源 1 us 的上升沿 ✓")
else:
    print(f"       -> 固定误差基本没变 => 上升沿不是主因，需另找原因")
print()

# ---- 实验 2a：加密 AC 扫频网格
N_POINTS_FINE = 200
f_f, mag_db_f, _ = run_ac(N_POINTS_FINE)
f_meas_f, _, _ = fc_from_sweep(f_f, mag_db_f, _)
print(f"实验2a 扫频点数 {N_POINTS_BASE} -> {N_POINTS_FINE} 点/十倍频")
print(f"       相邻点频率比 {10**(1.0/N_POINTS_BASE):.4f} -> {10**(1.0/N_POINTS_FINE):.4f}")
print(f"       fc 误差: {err(f_meas, FC):.2f}%  ->  {err(f_meas_f, FC):.2f}%")
print(f"       说明：{f_meas:.2f} Hz 恰好等于 10^3.2，是两套网格共同的格点，")
print(f"             所以加密网格后「最近的格点」仍然是它，误差纹丝不动。")
print()

# ---- 实验 2b：改成在 -3 dB 两侧格点之间做对数插值
f_int, k_int = fc_by_interpolation(f, mag_db)
print(f"实验2b 把 fc 从「取最近格点」改成「相邻两点对数插值」")
print(f"       夹住 -3 dB 的两个格点: {f[k_int]:.2f} Hz ({mag_db[k_int]:.4f} dB)"
      f"  与  {f[k_int+1]:.2f} Hz ({mag_db[k_int+1]:.4f} dB)")
print(f"       fc 误差: {err(f_meas, FC):.2f}%  ->  {err(f_int, FC):.4f}%")
if err(f_int, FC) < err(f_meas, FC) * 0.2:
    print(f"       -> 误差塌掉 => 确认 fc 的误差来自采样离散（取最近点而非插值）✓")
else:
    print(f"       -> 插值后误差没有明显改善 => 该原因不成立，需另找")
print()

# ================================================================ 4. 对比表
print("=" * 62)
print("对比表（粘进 README，再自己补「误差来源分析」一列）")
print("=" * 62)
rows = [
    ("tau (63.2% 读数法)", f"{TAU*1e6:.2f} us", f"{tau_a*1e6:.2f} us", err(tau_a, TAU)),
    ("tau (最小二乘拟合)", f"{TAU*1e6:.2f} us", f"{tau_b*1e6:.2f} us", err(tau_b, TAU)),
    ("截止频率 fc", f"{FC:.2f} Hz", f"{f_meas:.2f} Hz", err(f_meas, FC)),
    ("-3dB 处相位", "-45.00 deg", f"{phase_meas:.2f} deg", err(phase_meas, -45)),
    ("高频斜率", "-20.00 dB/dec", f"{slope_meas:.2f} dB/dec", err(slope_meas, -20)),
]
print(f"{'指标':<20}{'手算值':>16}{'仿真值':>18}{'相对误差':>12}")
print("-" * 62)
for name, th, sim_v, e in rows:
    print(f"{name:<20}{th:>16}{sim_v:>18}{e:>11.2f}%")
print("=" * 62)
print(f"图片已保存到: {OUT_DIR}")
