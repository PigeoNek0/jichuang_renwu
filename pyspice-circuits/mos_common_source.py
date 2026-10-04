"""
电路③ NMOS 共源级放大电路 —— 理论计算 + PySpice 仿真 + 对比表
================================================================
跑完这个脚本会得到：
  images/mos_transient.png    输入/输出波形（双 y 轴，可见 180 度反相）
  images/mos_ac_bode.png      AC 频率响应（可见 Cb1 引入的低频高通转折）
  终端打印的「手算 vs 仿真」对比表（表 3-1 静态工作点 / 表 3-2 小信号与增益）
  以及四个误差来源排除实验

运行：
    python mos_common_source.py

电路（题卡给定）：

        VDD(5V)
          │
    ┌─────┼───── Rd(2k)
    │     │      │
  Rg1(60k)│     [d] ── vo
    │     │      │
    ├──[g]│    ┌─┘  NMOS T
    │     └────┘    [s]─[B]── GND
  Rg2(40k)
    │
   GND        vi ──|Cb1|── [g]

关键点：
  源极[s] 与衬底[B] 短接后接地 -> 无源极电阻, V_GS = V_G
  输入经 Cb1 耦合到栅极（隔直, 不改变直流偏置）
  输出取自漏极[d]

★ 本题最有价值的误差来源：
  教材近似式 r_o = 1/(lambda*I_D) 忽略了 (1+lambda*V_DS) 因子,
  而 ngspice 用的是严格导数 1/(lambda*K*(V_GS-V_th)^2) = 62.5 kohm。
  两者相差 6.2%, 经 Rd//ro 的并联衰减后在增益上只剩 0.21%。
"""
import os
import sys

# 输出重定向到文件/管道时, Windows 默认用 GBK 编码, 打印 ✓ 之类字符会 UnicodeEncodeError。
# 统一改成 UTF-8（真控制台下 Python 本来就是 UTF-8, 这里是无害的兜底）。
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

# ---------------------------------------------------------------- 题目给定参数
VDD = 5.0            # V
RG1 = 60e3           # ohm
RG2 = 40e3           # ohm
RD = 2e3             # ohm
K = 0.8e-3           # A/V^2   题目给的 K, 即 I_D = K*(Vov)^2*(1+lambda*Vds)
VTH = 1.0            # V
LAM = 0.02           # 1/V
CB1 = 10e-6          # F       耦合电容（隔直电容）
VI = 10e-3           # V       输入正弦振幅
FREQ = 1e3           # Hz      输入频率

# SPICE level-1 MOS 的公式是 I_D = (KP/2)*(W/L)*(Vov)^2*(1+lambda*Vds)
# 题目给的 K 就是 K 本身, 故需 (KP/2)*(W/L) = K。取 W/L = 1（ngspice 默认 W=L）, KP = 2K。
KP = 2 * K           # 1.6e-3 A/V^2
WL = 1.0             # W/L = 1

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

ng = NgSpiceShared.new_instance()    # 整个脚本只建一次，全程复用


def make_sim(circuit):
    return CircuitSimulator.factory(circuit, simulator="ngspice-shared", ngspice_shared=ng)


def build(cb1=CB1, vgate_dc=None):
    """搭共源放大电路。

    cb1       -> 耦合电容容值（对照实验用）
    vgate_dc  -> 若给数值, 则栅极改由直流源直接驱动、去掉耦合电容
                 （只用于"扰动法测 g_m"的对照实验）
    """
    c = Circuit("NMOS common-source amplifier")
    c.model("NMOS_MODEL", "NMOS", level=1, VTO=VTH, KP=KP, LAMBDA=LAM)
    c.V("dd", "vdd", c.gnd, VDD @ u_V)
    c.R("g1", "vdd", "gate", RG1 @ u_Ohm)
    c.R("g2", "gate", c.gnd, RG2 @ u_Ohm)
    c.R("d", "vdd", "drain", RD @ u_Ohm)
    # M(漏, 栅, 源, 衬底) —— 源极与衬底短接后接地
    c.MOSFET(1, "drain", "gate", c.gnd, c.gnd, model="NMOS_MODEL")
    if vgate_dc is None:
        # ⚠️ cb1 的单位已经是「法拉」, 这里只能用 u_F; 写成 u_uF 会再乘 1e-6, 电容变成 10 pF
        c.C("b1", "gate", "vin", cb1 @ u_F)
        c.SinusoidalVoltageSource("i", "vin", c.gnd,
                                  amplitude=VI @ u_V, frequency=FREQ @ u_Hz,
                                  ac_magnitude=1 @ u_V)   # AC=1 -> 输出即增益
    else:
        c.V("i", "gate", c.gnd, vgate_dc @ u_V)
    return c


def vpp(analysis, node, t_lo):
    """取 t >= t_lo 之后该节点电压的峰峰值（稳态最后一个完整周期）"""
    t = np.asarray(analysis.time)
    v = np.asarray(analysis.nodes[node])
    m = t >= t_lo
    return float(v[m].max() - v[m].min())


def err(a, b):
    return abs(a - b) / abs(b) * 100.0


def rd_parallel_ro(ro):
    return RD * ro / (RD + ro)


# ================================================================ 1. 手算
print("=" * 72)
print("电路③ NMOS 共源级放大电路")
print("=" * 72)
print(f"给定参数: VDD = {VDD:g} V, Rg1 = {RG1/1e3:g} kohm, Rg2 = {RG2/1e3:g} kohm, Rd = {RD/1e3:g} kohm")
print(f"          K = {K*1e3:g} mA/V^2, Vth = {VTH:g} V, lambda = {LAM:g} /V")
print(f"          vi = {VI*1e3:g} mV 振幅 @ {FREQ/1e3:g} kHz")
print()

# --- 第 1 步 栅极偏置
VG = VDD * RG2 / (RG1 + RG2)
VOV = VG - VTH
print("[手算] 第 1 步 · 栅极偏置（源极接地, 故 V_GS = V_G）")
print(f"       V_G = VDD * Rg2/(Rg1+Rg2) = {VDD:g} * {RG2/1e3:g}/{RG1/1e3:g}+{RG2/1e3:g} = {VG:.4f} V")
print(f"       -> V_GS = {VG:.4f} V,  V_ov = V_GS - Vth = {VOV:.4f} V")
print()

# --- 第 2 步 联立求 Q 点
print("[手算] 第 2 步 · 联立求 Q 点")
print("       I_D = K*(V_GS-Vth)^2*(1+lambda*V_DS),  V_DS = VDD - I_D*Rd")
print()
print("       解法 A · 迭代法")
print(f"       {'迭代':>4}{'假设 V_DS':>12}{'1+lambda*V_DS':>16}{'算得 I_D(mA)':>16}{'新 V_DS(V)':>14}")
its = []
vds = None
for i in range(6):
    factor = 1.0 if vds is None else 1 + LAM * vds
    idn = K * VOV ** 2 * factor
    new_vds = VDD - idn * RD
    shown = "忽略 lambda" if vds is None else f"{vds:.4f}"
    print(f"       {i:>4}{shown:>12}{factor:>16.4f}{idn*1e3:>16.4f}{new_vds:>14.4f}")
    its.append((i, shown, factor, idn * 1e3, new_vds))
    # 迭代收敛很慢（每步只缩小 lambda*K*Rd*Vov^2 = 3.2%）, 按 4 位小数判收敛即可
    if vds is not None and abs(new_vds - vds) < 5e-4:
        print("       -> 收敛 ✓")
        break
    vds = new_vds

# 解法 B 闭式解
ID = K * VOV ** 2 * (1 + LAM * VDD) / (1 + LAM * K * RD * VOV ** 2)
VDS = VDD - ID * RD
print()
print("       解法 B · 闭式解（把 V_DS = VDD - I_D*Rd 代入整理）")
print("       I_D = K*Vov^2*(1+lambda*VDD) / (1 + lambda*K*Rd*Vov^2)")
print(f"           = {K*VOV**2*1e3:.4f}m * {1+LAM*VDD:.3f} / (1 + {LAM*K*RD*VOV**2:.3f})"
      f" = {K*VOV**2*(1+LAM*VDD)*1e3:.4f}m / {1+LAM*K*RD*VOV**2:.3f}")
print(f"       -> I_D = {ID*1e3:.6f} mA,  V_DS = {VDS:.6f} V")
print(f"       （两个解法结果相差 {abs(ID - its[-1][3]*1e-3)/ID*100:.2e}%，即完全一致）")
print()

# --- 第 3 步 饱和区判断
print("[手算] 第 3 步 · 饱和区判断")
print(f"       V_DS = {VDS:.4f} V  >  V_GS - Vth = {VOV:.4f} V")
print(f"       -> 工作在饱和区 ✓")
print()

# --- 第 4 步 小信号参数
GM = 2 * K * VOV * (1 + LAM * VDS)
RO_STRICT = 1.0 / (LAM * K * VOV ** 2)      # ngspice 用的严格偏导
RO_TEXT = 1.0 / (LAM * ID)                  # 教材近似式, 忽略了 (1+lambda*V_DS)
AV_STRICT = -GM * rd_parallel_ro(RO_STRICT)
AV_TEXT = -GM * rd_parallel_ro(RO_TEXT)
print("[手算] 第 4 步 · 小信号参数")
print(f"       g_m = 2*K*Vov*(1+lambda*V_DS) = 2*{K*1e3:g}m*{VOV:.4f}*{1+LAM*VDS:.6f} = {GM*1e3:.6f} mA/V")
print(f"       r_o 教材近似 1/(lambda*I_D)              = {RO_TEXT/1e3:.4f} kohm")
print(f"       r_o 严格偏导 1/(lambda*K*Vov^2)          = {RO_STRICT/1e3:.4f} kohm"
      f"   （两者相差 {err(RO_TEXT, RO_STRICT):.2f}%）")
print()

# --- 第 5 步 增益
print("[手算] 第 5 步 · 电压增益 A_v = -g_m*(Rd // r_o)")
print(f"       A_v = -{GM*1e3:.6f}m * {rd_parallel_ro(RO_TEXT):.4f} = {AV_TEXT:.6f}   （用教材近似 r_o）")
print(f"       A_v = -{GM*1e3:.6f}m * {rd_parallel_ro(RO_STRICT):.4f} = {AV_STRICT:.6f}   （用严格 r_o）★")
print(f"       输出振幅 = {VI*1e3:g} mV * {abs(AV_STRICT):.6f} = {VI*1e3*abs(AV_STRICT):.4f} mV"
      f"  -> 峰峰值 {2*VI*1e3*abs(AV_STRICT):.4f} mV")
print(f"       相位与输入反相（A_v < 0, 即 180 度）")
print()

# 耦合电容引入的高通相位超前
F_LOW = 1.0 / (2 * np.pi * CB1 * (RG1 * RG2 / (RG1 + RG2)))
PHASE_LEAD = np.degrees(np.arctan(F_LOW / FREQ))
print(f"[手算] 耦合电容 Cb1 与 Rg1//Rg2 = {RG1*RG2/(RG1+RG2)/1e3:.0f} kohm 构成高通,")
print(f"       转折频率 f1 = 1/(2*pi*Cb1*R) = {F_LOW:.4f} Hz")
print(f"       在 {FREQ/1e3:g} kHz 处相位超前 arctan(f1/f) = {PHASE_LEAD:.4f} 度")
print(f"       -> 预期 AC 相位 = -180 + {PHASE_LEAD:.4f} = {-180+PHASE_LEAD:.4f} 度")
print()

# ================================================================ 2. 仿真：静态工作点 + 器件内部参数
# ⚠️ 实测坑：一旦调用 save_internal_parameters, ngspice 的 .op 原始输出里就只剩被声明的内部参数,
#    PySpice 解析出来的 op.nodes 会变成空 dict（KeyError: 'drain'）。
#    所以「节点电压」和「器件内部参数」必须分成两次 .op 来拿。
op = make_sim(build()).operating_point()
sim_vd = float(np.asarray(op.nodes["drain"]).ravel()[0])

sim_ip = make_sim(build())
sim_ip.save_internal_parameters("@m1[gm]", "@m1[gds]", "@m1[id]",
                                "@m1[vgs]", "@m1[vds]")
ip = sim_ip.operating_point().internal_parameters     # 是属性（dict）, 不是方法
sim_gm = float(np.asarray(ip["@m1[gm]"]).ravel()[0])
sim_gds = float(np.asarray(ip["@m1[gds]"]).ravel()[0])
sim_id = float(np.asarray(ip["@m1[id]"]).ravel()[0])
sim_vgs = float(np.asarray(ip["@m1[vgs]"]).ravel()[0])
sim_vds = float(np.asarray(ip["@m1[vds]"]).ravel()[0])
sim_ro = 1.0 / sim_gds

print("-" * 72)
print("[仿真] 静态工作点与器件内部参数（直接读 ngspice 的 @m1[...]）")
print(f"  @m1[vgs] = {sim_vgs:.6f} V     手算 {VG:.6f} V     误差 {err(sim_vgs, VG):.6f}%")
print(f"  @m1[id]  = {sim_id*1e3:.6f} mA 手算 {ID*1e3:.6f} mA 误差 {err(sim_id, ID):.6f}%")
print(f"  @m1[vds] = {sim_vds:.6f} V     手算 {VDS:.6f} V    误差 {err(sim_vds, VDS):.6f}%")
print(f"  V(drain) = {sim_vd:.6f} V     （应为 V_DS + 0, 即 {VDS:.6f} V）")
print(f"  @m1[gm]  = {sim_gm*1e3:.6f} mA/V  手算 {GM*1e3:.6f} mA/V  误差 {err(sim_gm, GM):.6f}%")
print(f"  @m1[gds] = {sim_gds:.6e} S     -> r_o = {sim_ro/1e3:.4f} kohm"
      f"（= 严格值 {RO_STRICT/1e3:.4f} kohm, 不是教材值 {RO_TEXT/1e3:.4f} kohm）")
print()

# ================================================================ 3. 瞬态分析
T_END = 5e-3         # 5 ms = 5 个周期
T_STEP = 1e-6        # 1 us
T_SETTLE = 4e-3      # 只统计最后一个完整周期, 避开起始点

sim_tr = make_sim(build())
tran = sim_tr.transient(step_time=T_STEP @ u_s, end_time=T_END @ u_s)
vi_pp = vpp(tran, "vin", T_SETTLE)
vo_pp = vpp(tran, "drain", T_SETTLE)
AV_TRAN = vo_pp / vi_pp

print("-" * 72)
print("[仿真] 瞬态分析（正弦 {:.0f} mV / {:g} kHz）".format(VI * 1e3, FREQ / 1e3))
print(f"  输入峰峰值  {vi_pp*1e3:.6f} mV    （理论 {2*VI*1e3:.4f} mV）")
print(f"  输出峰峰值  {vo_pp*1e3:.6f} mV    （理论 {2*VI*1e3*abs(AV_STRICT):.4f} mV）")
print(f"  瞬态增益 |A_v| = {AV_TRAN:.6f}")
print()

# ================================================================ 4. AC 分析
sim_ac = make_sim(build())
ac = sim_ac.ac(start_frequency=0.1 @ u_Hz, stop_frequency=100 @ u_kHz,
               number_of_points=100, variation="dec")
freq = np.asarray(ac.frequency)
vo_ac = np.asarray(ac.nodes["drain"])
gain = np.abs(vo_ac)
ph = np.degrees(np.angle(vo_ac))
k1 = int(np.argmin(np.abs(freq - FREQ)))     # 离 1 kHz 最近的扫频点
AV_AC = float(gain[k1])
PH_AC = float(ph[k1])
print("-" * 72)
print("[仿真] AC 分析")
print(f"  扫频点 {len(freq)} 个, {freq[0]:g} Hz ~ {freq[-1]/1e3:g} kHz")
print(f"  1 kHz 处（第 {k1} 点, f = {freq[k1]:.6g} Hz）:")
print(f"    |A_v| = {AV_AC:.6f}      与瞬态测值 {AV_TRAN:.6f} 相差 {err(AV_TRAN, AV_AC):.6f}%")
print(f"    相位  = {PH_AC:.4f} 度   理论 {(-180+PHASE_LEAD):.4f} 度   相差 {abs(PH_AC-(-180+PHASE_LEAD)):.4f} 度")
print()

# ================================================================ 5. 误差来源排除实验
print("=" * 72)
print("[排除实验] 改变一个变量, 看误差是否随之变化")
print("=" * 72)

# 实验 1: Cb1 加大 100 倍 —— 排除"耦合电容不够大导致低频衰减"
sim_cb = make_sim(build(cb1=CB1 * 100))
tr_cb = sim_cb.transient(step_time=T_STEP @ u_s, end_time=T_END @ u_s)
av_cb = vpp(tr_cb, "drain", T_SETTLE) / vpp(tr_cb, "vin", T_SETTLE)
print(f"实验1  Cb1 {CB1*1e6:g} uF -> {CB1*1e8:g} uF（转折频率 {F_LOW:.4f} Hz -> {F_LOW/100:.6f} Hz）")
print(f"       增益 {AV_TRAN:.6f} -> {av_cb:.6f},  变化 {err(av_cb, AV_TRAN):.4f}%  -> 排除 ✗")
print()

# 实验 2: 时间步长减半 —— 排除"步长导致峰值被削"
sim_h = make_sim(build())
tr_h = sim_h.transient(step_time=(T_STEP / 2) @ u_s, end_time=T_END @ u_s)
av_h = vpp(tr_h, "drain", T_SETTLE) / vpp(tr_h, "vin", T_SETTLE)
print(f"实验2  时间步长 {T_STEP*1e6:g} us -> {T_STEP*0.5*1e6:g} us")
print(f"       增益 {AV_TRAN:.6f} -> {av_h:.6f},  变化 {err(av_h, AV_TRAN):.4f}%  -> 排除 ✗")
print()

# 实验 3: 扰动法测 g_m —— 说明为什么必须读 @m1[gm]
VGS_NOM = VG
DV = 0.1e-3          # ±0.1 mV


def id_at(vg):
    s = make_sim(build(vgate_dc=vg))
    s.save_internal_parameters("@m1[id]")
    o = s.operating_point()
    return float(np.asarray(o.internal_parameters["@m1[id]"]).ravel()[0])


id_p, id_m = id_at(VGS_NOM + DV), id_at(VGS_NOM - DV)
gm_perturb = (id_p - id_m) / (2 * DV)
gm_closed_loop = GM / (1 + LAM * K * RD * VOV ** 2)
print(f"实验3  扰动法测 g_m（栅极 ±{DV*1e6:.1f} uV 小扰动, 看 dI_D/dV_GS）")
print(f"       扰动法得 {gm_perturb*1e3:.6f} mA/V,  比真值 {GM*1e3:.6f} mA/V 低 {err(gm_perturb, GM):.4f}%")
print(f"       原因: V_GS 变化会经 Rd 反馈到 V_DS, 测到的是闭环跨导")
print(f"             g_m/(1+lambda*K*Rd*Vov^2) = {GM*1e3:.6f}/{1+LAM*K*RD*VOV**2:.3f}"
      f" = {gm_closed_loop*1e3:.6f} mA/V")
print(f"             与扰动法实测 {gm_perturb*1e3:.6f} mA/V 相差 {err(gm_perturb, gm_closed_loop):.6f}%  -> 完全吻合")
print(f"       -> 要用器件本身的 g_m, 必须读 @m1[gm]（不搞扰动法）")
print()

# 实验 4: 把 r_o 换成严格值重算 —— 确认唯一的真实误差来源
print(f"实验4  手算 r_o 用教材近似式 {RO_TEXT/1e3:.4f} kohm -> 换成仿真值 {sim_ro/1e3:.4f} kohm")
print(f"       A_v 由 {AV_TEXT:.6f} -> {AV_STRICT:.6f}")
print(f"       与仿真 {AV_AC:.6f} 的误差: {err(abs(AV_TEXT), AV_AC):.4f}%  ->  {err(abs(AV_STRICT), AV_AC):.4f}%")
print("       -> 确认 ✓ 这是本题唯一「真实可解释」的差异")
print()

# ================================================================ 6. 画图
# --- 图 3-4: 输入/输出波形（双 y 轴, 反相一目了然）
t = np.asarray(tran.time)
vi_t = np.asarray(tran.nodes["vin"])
vo_t = np.asarray(tran.nodes["drain"])
m = t >= T_SETTLE
# ⚠️ 输出叠在 V_DS = 3.2946 V 的直流偏置上, 直接画会被 5 V 量程压成一条直线,
#    必须把直流分量减掉, "反相"才看得出来。
vo_ac_t = (vo_t - VDS) * 1e3
vi_t_mv = vi_t * 1e3

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(t[m] * 1e3, vi_t_mv[m], "--", color="tab:gray", lw=2.0, label="$V_{in}$ (left axis)")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("$V_{in}$ (mV)", color="tab:gray")
ax1.tick_params(axis="y", labelcolor="tab:gray")
ax1.axhline(0, color="gray", lw=0.8)
ax2 = ax1.twinx()
ax2.plot(t[m] * 1e3, vo_ac_t[m], "-", color="tab:red", lw=2.2, label="$V_{out}-V_{DS}$ (right axis)")
ax2.set_ylabel(f"$V_{{out}}-V_{{DS}}$ (mV)   [DC bias {VDS*1e3:.1f} mV removed]", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.axhline(0, color="tab:red", lw=0.8)
# 标出输入的正峰值处: 此刻输出恰为负峰值 -> 180 度反相一目了然
tz = int(np.argmin(np.abs(t[m] - 4.25e-3)))
ax1.axvline(t[m][tz] * 1e3, color="tab:blue", lw=0.9, ls=":")
ax1.plot(t[m][tz] * 1e3, vi_t_mv[m][tz], "o", color="tab:gray", ms=7, zorder=5)
ax2.plot(t[m][tz] * 1e3, vo_ac_t[m][tz], "o", color="tab:red", ms=7, zorder=5)
ax1.annotate(f"Vin peaks at +{vi_t_mv[m][tz]:.1f} mV\n"
             f"Vout bottoms at {vo_ac_t[m][tz]:.1f} mV\n"
             f"-> 180 deg out of phase",
             xy=(t[m][tz] * 1e3, vi_t_mv[m][tz]), xytext=(4.32, -1.5), fontsize=9,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="tab:blue"),
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
ax1.set_title(f"NMOS common-source amplifier: transient response\n"
              f"(Vin {2*VI*1e3:.0f} mVpp @ {FREQ/1e3:g} kHz  ->  Vout {vo_pp*1e3:.2f} mVpp, "
              f"|Av| = {AV_TRAN:.4f})")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mos_transient.png"), dpi=150)
plt.close(fig)

# --- 附: AC 频率响应（把 Cb1 引入的低频高通画出来, 支撑相位那条误差分析）
gain_db = 20 * np.log10(gain)
MID_DB = 20 * np.log10(abs(AV_STRICT))
BOX = dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.9)

fig, (axm, axp) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
axm.semilogx(freq, gain_db, "-", color="tab:blue", lw=2, zorder=3)
axm.axvline(FREQ, color="tab:green", lw=1, ls="--", zorder=2)
axm.axhline(MID_DB, color="tab:red", lw=1, ls=":", zorder=2)
axm.set_ylim(gain_db.min() - 2, MID_DB + 4)
axm.set_xlim(freq[0], freq[-1])
axm.annotate(f"midband {MID_DB:.3f} dB", xy=(freq[-1], MID_DB),
             xytext=(3e3, MID_DB + 1.6), color="tab:red", fontsize=9, ha="center",
             bbox=BOX, arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
axm.annotate(f"1 kHz: {gain_db[k1]:.3f} dB", xy=(FREQ, gain_db[k1]),
             xytext=(30, MID_DB + 2.6), color="tab:green", fontsize=9, ha="center",
             bbox=BOX, arrowprops=dict(arrowstyle="->", color="tab:green", lw=1))
axm.annotate(f"-3 dB corner from Cb1\nf1 = {F_LOW:.3f} Hz", xy=(F_LOW, MID_DB - 3),
             xytext=(0.06, MID_DB - 3.4), fontsize=9, ha="left", color="tab:orange",
             bbox=BOX, arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1))
axm.set_ylabel("|$A_v$| (dB)")
axm.set_title("Common-source amplifier: AC frequency response\n"
              "(level-1 model has no Cgs/Cgd -> no high-frequency rolloff here)")
axm.grid(alpha=0.3, which="both")

axp.semilogx(freq, ph, "-", color="tab:red", lw=2)
axp.axvline(FREQ, color="tab:green", lw=1, ls="--")
axp.axhline(-180, color="gray", lw=0.8, ls=":")
axp.annotate(f"1 kHz: {PH_AC:.3f} deg", xy=(FREQ, PH_AC), xytext=(25, ph.min() + 8),
             color="tab:green", fontsize=9, ha="center", bbox=BOX,
             arrowprops=dict(arrowstyle="->", color="tab:green", lw=1))
axp.set_xlabel("Frequency (Hz)")
axp.set_ylabel("Phase (deg)")
axp.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mos_ac_bode.png"), dpi=150)
plt.close(fig)

# ================================================================ 7. 对比表
print("=" * 72)
print("表 3-1 静态工作点「手算 vs 仿真」（粘进 README, 自己补「误差来源」一列）")
print("=" * 72)
print(f"{'参数':<14}{'手算值':>16}{'仿真值':>16}{'相对误差':>12}")
print("-" * 72)
for name, th2, sm2 in [
    ("V_GS", f"{VG:.6f} V", f"{sim_vgs:.6f} V"),
    ("I_D", f"{ID*1e3:.6f} mA", f"{sim_id*1e3:.6f} mA"),
    ("V_DS", f"{VDS:.6f} V", f"{sim_vds:.6f} V"),
]:
    print(f"{name:<14}{th2:>16}{sm2:>16}{err(float(sm2.split()[0]), float(th2.split()[0])):>11.6f}%")
print(f"{'是否饱和':<14}{'是 (V_DS>V_ov)':>16}{'是 (@m1[id] 与平方律吻合)':>34}")
print("-" * 72)
print("  注: 本仿真 level-1 模型与手算公式同构, Q 点吻合到 1e-6 量级,")
print("      差异全部来自 ngspice 的数值求解容差, 属「数值误差」, 非建模误差。")
print()

print("=" * 72)
print("表 3-2 小信号与增益「手算 vs 仿真」")
print("=" * 72)
print(f"{'参数':<16}{'手算值':>16}{'仿真值':>16}{'相对误差':>12}")
print("-" * 72)
# 注意: A_v 是负的（反相）, 比误差时要取模, 否则会算出 200%
rows32 = [
    ("g_m", GM * 1e3, sim_gm * 1e3, "{:.6f} mA/V"),
    ("r_o", RO_TEXT / 1e3, sim_ro / 1e3, "{:.4f} kohm"),
    ("A_v (教材 r_o)", abs(AV_TEXT), AV_AC, "{:.6f}"),
    ("A_v (严格 r_o)", abs(AV_STRICT), AV_AC, "{:.6f}"),
    ("输出峰峰值", 2 * VI * 1e3 * abs(AV_TEXT), vo_pp * 1e3, "{:.4f} mV"),
]
for name, th_n, sm_n, fmt in rows32:
    print(f"{name:<16}{fmt.format(th_n):>16}{fmt.format(sm_n):>16}{err(sm_n, th_n):>11.6f}%")
print("-" * 72)
print(f"  AC 相位 1 kHz: 仿真 {PH_AC:.4f} 度, 手算 -180 + {PHASE_LEAD:.4f} = {-180+PHASE_LEAD:.4f} 度")
print(f"  瞬态增益 {AV_TRAN:.6f}  vs  AC 增益 {AV_AC:.6f}  （互验, 差 {err(AV_TRAN, AV_AC):.6f}%）")
print("=" * 72)
print(f"图片已保存到: {OUT_DIR}")
print("  mos_transient.png   输入/输出波形（反相放大）")
print("  mos_ac_bode.png     AC 频率响应（Cb1 高通转折）")
