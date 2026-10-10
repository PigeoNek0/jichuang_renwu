"""
电路② 戴维南定理验证 —— 理论计算 + PySpice 仿真 + 对比表
================================================================
跑完这个脚本会得到：
  images/thevenin_load_line.png   端口伏安特性曲线（原网络 vs 戴维南等效，两条线应完全重合）
  终端打印的「手算 vs 仿真」对比表  -> 直接粘进 README

运行：
    python thevenin.py

被测网络（含源二端网络，端口为 A 与 GND，A 为 +）：
两条 R-V 串联支路并联在端口上（手绘原图见 README 2.1）：

        端口 A（+）
          ●
          │
    ┌─────┴─────┐
    │           │
 [ R1 1k ]   [ R2 3k ]
    │           │
 [ V1 12V ]  [ V2 4V ]      ← 两个电源正极均朝上（接电阻那一侧）
    │           │
    └─────┬─────┘
          ⏚ GND（端口 −）

写成网表就是两条支路并联：
  A — R1(1k) — n1 — V1(12V) — GND
  A — R2(3k) — n2 — V2(4V)  — GND

手算结果：V_oc = 10 V，R_th = 750 ohm，I_sc = 13.3333 mA

方法说明：
  测 V_oc —— 端口开路（并联 1GΩ 假负载防止浮空节点报 "no DC path to ground"）
  测 I_sc —— 端口串一个 0V 电压源当电流表，读它的支路电流
  测 R_th —— 独立源置零，端口外加 1A 电流源，端口电压的数值就是 R_th
  负载验证 —— 循环重建电路（.dc 扫电阻不被 ngspice 支持）
"""
import os
import sys

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

# ---------------------------------------------------------------- 网络参数（对应手绘原图）
V1, R1 = 12.0, 1e3   # 支路 1：1 kΩ 串 12 V
V2, R2 = 4.0, 3e3    # 支路 2：3 kΩ 串 4 V
FAKE_LOAD = 1e9      # 1 GΩ 假负载，仅用于"端口开路"时给浮空节点一条直流通路

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

ng = NgSpiceShared.new_instance()    # 整个脚本只建一次，全程复用


def make_sim(circuit):
    return CircuitSimulator.factory(circuit, simulator="ngspice-shared", ngspice_shared=ng)


def network(port=None):
    """搭含源二端网络（两条 R-V 支路并联）。

    port=None    -> 端口开路（自动加 1GΩ 假负载）
    port='short' -> 端口接 0V 电压源当电流表
    port=<数值>  -> 端口接该阻值的负载 R_L
    """
    c = Circuit("thevenin network")
    c.V("1", "n1", c.gnd, V1 @ u_V)      # V1 正极接 n1（朝上）
    c.R("1", "n1", "a", R1 @ u_Ohm)
    c.V("2", "n2", c.gnd, V2 @ u_V)      # V2 正极接 n2（朝上）
    c.R("2", "a", "n2", R2 @ u_Ohm)
    if port is None:
        c.R("fake", "a", c.gnd, FAKE_LOAD @ u_Ohm)
    elif port == "short":
        c.V("sc", "a", c.gnd, 0 @ u_V)
    else:
        c.R("L", "a", c.gnd, port @ u_Ohm)
    return c


def equivalent(r_load):
    """戴维南等效电路：V_th 串联 R_th，再接负载 R_L"""
    c = Circuit("thevenin equivalent")
    c.V("th", "t", c.gnd, VTH @ u_V)
    c.R("th", "t", "a", RTH @ u_Ohm)
    c.R("L", "a", c.gnd, r_load @ u_Ohm)
    return c


# ================================================================ 1. 理论计算
print("=" * 68)
print("电路② 戴维南定理验证")
print("=" * 68)
print(f"网络参数: V1 = {V1:g} V, R1 = {R1/1e3:g} kohm   (支路 1)")
print(f"          V2 = {V2:g} V, R2 = {R2/1e3:g} kohm   (支路 2)")
print()
print("[手算] 对端口节点 A 写 KCL（流出为正）——开路时端口无电流，两条支路电流之和为零：")
print(f"       (V_A - {V1:g})/{R1/1e3:g}k + (V_A - {V2:g})/{R2/1e3:g}k = 0")
print(f"       两边同乘 {R2/1e3:g}k：{R2/1e3:g}(V_A - {V1:g}) + (V_A - {V2:g}) = 0 -> "
      f"{R2/1e3+1:g}*V_A = {V1*R2/1e3 + V2:g}")

# (V_A-V1)/R1 + (V_A-V2)/R2 = 0
G1, G2 = 1.0 / R1, 1.0 / R2
VTH = (V1 * G1 + V2 * G2) / (G1 + G2)
RTH = 1.0 / (G1 + G2)
ISC = VTH / RTH

print(f"       V_A = (V1*G1 + V2*G2) / (G1+G2) = {VTH:.6f} V")
print(f"       -> V_th = V_oc = {VTH:.6f} V")
print()
print(f"[手算] 独立源置零后 R_th = R1//R2 = {RTH:.6f} ohm")
print(f"[手算] I_sc = V_th / R_th = {ISC*1e3:.6f} mA")
print(f"       自检（叠加法）：V_oc = {V1:g}*{R2/1e3:g}/({R1/1e3:g}+{R2/1e3:g})"
      f" + {V2:g}*{R1/1e3:g}/({R1/1e3:g}+{R2/1e3:g})"
      f" = {V1*R2/(R1+R2):g} + {V2*R1/(R1+R2):g} = {VTH:g} V")
print()

# ================================================================ 2. 测开路电压 V_oc
op = make_sim(network(None)).operating_point()
voc = float(np.asarray(op.nodes["a"]).ravel()[0])

# ================================================================ 3. 测短路电流 I_sc
# ⚠️ 坑：branches['vsc'] 的原始读数就是 I_sc，符号与手算一致，不要加负号。
#    第一版凭"SPICE 电流方向约定"想当然加了负号，误差直接变成 200%。
op_sc = make_sim(network("short")).operating_point()
isc = float(np.asarray(op_sc.branches["vsc"]).ravel()[0])

# ================================================================ 4. 激励法独立验算 R_th
# 内部独立源全部置零：电压源 -> 短路（0V 源）；端口外加 1A 电流源灌入
c_ext = Circuit("excitation method")
c_ext.R("1", "n1", "a", R1 @ u_Ohm)
c_ext.R("2", "a", "n2", R2 @ u_Ohm)
c_ext.V("1", "n1", c_ext.gnd, 0 @ u_V)      # V1 置零 = 短路
c_ext.V("2", "n2", c_ext.gnd, 0 @ u_V)      # V2 置零 = 短路
c_ext.I("inject", c_ext.gnd, "a", 1 @ u_A)  # 1A 灌入端口
op_ext = make_sim(c_ext).operating_point()
rth_ext = float(np.asarray(op_ext.nodes["a"]).ravel()[0])   # 数值上等于 R_th（单位 ohm）


def err(a, b):
    return abs(a - b) / abs(b) * 100.0


print("-" * 68)
print("[仿真] 等效参数")
print(f"  V_oc   : 手算 {VTH:12.6f} V    仿真 {voc:12.6f} V    误差 {err(voc, VTH):6.4f}%")
print(f"  I_sc   : 手算 {ISC*1e3:12.6f} mA   仿真 {isc*1e3:12.6f} mA   误差 {err(isc, ISC):6.4f}%")
print(f"  R_th   : 手算 {RTH:12.6f} ohm  激励法 {rth_ext:12.6f} ohm  误差 {err(rth_ext, RTH):6.4f}%")
print(f"           （定义法 V_oc/I_sc = {voc/isc:.6f} ohm，与激励法互验）")
print(f"           注：开路时并联的 1GΩ 假负载会让 V_oc 偏低 {RTH/(RTH+FAKE_LOAD):.2e}"
      f"（即 {RTH/(RTH+FAKE_LOAD)*100:.6f}%），")
print(f"              这正是上表 V_oc 那一行误差的全部来源，不是求解器容差。")
print()

# ================================================================ 5. 负载验证表
LOADS = [100.0, RTH, 1e3, 10e3]
print("-" * 68)
print("[仿真] 等效电路替换后接负载的验证（戴维南定理的核心证据）")
print("-" * 68)
print(f"{'R_L':>10}{'原网络 V':>13}{'等效 V':>13}{'原网络 I(mA)':>15}{'等效 I(mA)':>15}{'一致':>7}")
print("-" * 68)

rows = []
for rl in LOADS:
    # 原网络：端口接 R_L
    oa = make_sim(network(rl)).operating_point()
    va = float(np.asarray(oa.nodes["a"]).ravel()[0])
    ia = va / rl                       # ⚠️ 端口电流用 V(port)/R_L，不要读 i(v1)
    # 等效电路
    ob = make_sim(equivalent(rl)).operating_point()
    vb = float(np.asarray(ob.nodes["a"]).ravel()[0])
    ib = vb / rl
    same = "OK" if abs(va - vb) < 1e-5 else "DIFF"
    rows.append((rl, va, vb, ia, ib, same))
    print(f"{rl:>10.1f}{va:>13.6f}{vb:>13.6f}{ia*1e3:>15.6f}{ib*1e3:>15.6f}{same:>7}")
print("-" * 68)
print(f"注：两条 V 列的差异在 1e-6 V 量级，是 ngspice 的数值求解容差，不是物理差异。")
print(f"    R_L = R_th = {RTH:g} ohm 时端口电压恰为 V_oc/2 = {VTH/2:.6f} V（最大功率传输点）。")
print()

# ================================================================ 6. 端口伏安特性曲线
# 做法：端口直接接一个电压源 Vp（初始 0V），用 .dc 扫 Vp，读它的支路电流
#       —— 一条扫描同时得到原网络和等效电路的完整 I-V 特性。
print("-" * 68)
print("[仿真] 端口伏安特性（.dc 扫端口电压源）")


def iv_curve(kind):
    c = Circuit(f"iv {kind}")
    if kind == "original":
        c.V("1", "n1", c.gnd, V1 @ u_V)
        c.R("1", "n1", "a", R1 @ u_Ohm)
        c.V("2", "n2", c.gnd, V2 @ u_V)
        c.R("2", "a", "n2", R2 @ u_Ohm)
    else:
        c.V("th", "t", c.gnd, VTH @ u_V)
        c.R("th", "t", "a", RTH @ u_Ohm)
    c.V("p", "a", c.gnd, 0 @ u_V)      # 端口电压源，用 .dc 扫它
    # ⚠️ .dc 的正确语法：值是 slice 对象，键是元件名；扫到 12 V 才能盖住 V_oc = 10 V
    an = make_sim(c).dc(Vp=slice(-2, 12.01, 0.25))
    v = np.asarray(an.nodes["a"])
    i = np.asarray(an.branches["vp"])   # SPICE 约定：流入 + 端的电流
    return v, i


v_orig, i_orig = iv_curve("original")
v_equiv, i_equiv = iv_curve("equiv")
print(f"  扫描点 {len(v_orig)} 个，端口电压 {v_orig[0]:.1f} ~ {v_orig[-1]:.1f} V")
print(f"  两条 I-V 曲线最大偏差 = {np.max(np.abs(i_orig - i_equiv))*1e6:.4f} uA")
print()

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(v_orig, i_orig * 1e3, "-", color="tab:blue", lw=2.6, label="Original network")
ax.plot(v_equiv, i_equiv * 1e3, "--", color="tab:red", lw=1.4, label="Thevenin equivalent")
ax.plot([0], [isc * 1e3], "o", color="tab:green", ms=7, zorder=5)
ax.annotate(f"short circuit\nI_sc = {isc*1e3:.2f} mA", xy=(0, isc * 1e3),
            xytext=(0.6, isc * 1e3 + 1.2), color="tab:green", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=1))
ax.plot([VTH], [0], "s", color="tab:purple", ms=7, zorder=5)
ax.annotate(f"open circuit\nV_oc = {VTH:.3f} V", xy=(VTH, 0),
            xytext=(VTH - 4.4, 1.6), color="tab:purple", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1))
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("Port voltage $V_{port}$ (V)")
ax.set_ylabel("Port current $I_{port}$ (mA)")
ax.set_title("Thevenin theorem: port I-V characteristic\n"
             "(the two curves coincide -> the networks are indistinguishable at the port)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "thevenin_load_line.png"), dpi=150)
plt.close(fig)

# ================================================================ 7. 对比表
print("=" * 68)
print("对比表（粘进 README，再自己补「误差来源」一列）")
print("=" * 68)
print(f"{'参数':<16}{'手算值':>15}{'仿真值':>15}{'相对误差':>12}")
print("-" * 68)
for name, th, sm in [
    ("V_oc = V_th", f"{VTH:.6f} V", f"{voc:.6f} V"),
    ("I_sc", f"{ISC*1e3:.6f} mA", f"{isc*1e3:.6f} mA"),
    ("R_th(定义法)", f"{RTH:.6f} ohm", f"{voc/isc:.6f} ohm"),
    ("R_th(激励法)", f"{RTH:.6f} ohm", f"{rth_ext:.6f} ohm"),
]:
    th_num = float(th.split()[0])
    sm_num = float(sm.split()[0])
    print(f"{name:<16}{th:>15}{sm:>15}{err(sm_num, th_num):>11.4f}%")
print("=" * 68)
print(f"图片已保存到: {os.path.join(OUT_DIR, 'thevenin_load_line.png')}")
print()
print("思考题（写进 README 会加分）：")
print("  为什么 R_L 越大，两个电路的端口电压越接近 V_th？")
print("  为什么 R_L -> 无穷时电流都趋于 0？")
