"""
电路图绘制脚本 —— 生成 README 里需要的原理图（PNG）
================================================================
任务书要求「电路图自己画（手绘拍照 / 画图软件均可）」。
本脚本用 matplotlib 图元把五张图画出来，好处是：
  · 参数改了重跑一次即可，图永远和仿真脚本里的参数一致
  · 矢量重绘、清晰、可版本管理（二进制手绘照片做不到）
  · 元件符号、标注风格全部统一

运行：
    python draw_circuits.py

产出（images/ 下）：
    rc_circuit.png          电路① RC 低通（输出取自电容两端）
    thevenin_network.png    电路② 含源二端网络（标注端口 A / GND）
    mos_circuit.png         电路③ NMOS 共源放大完整电路
    mos_dc_path.png         电路③ 直流通路（Cb1 开路）
    mos_small_signal.png    电路③ 小信号等效模型
"""
import os
import sys

# 输出重定向时 Windows 用 GBK, 打印中文会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# matplotlib 默认缓存目录在用户目录, 部分环境不可写会报警告
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "mplcache"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
except ImportError:
    sys.exit("缺少 matplotlib,请先安装:pip install matplotlib")

# 图上要写中文, DejaVu Sans 没有汉字, 必须在 Windows 自带字体里挑一个
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII, 免得渲染成方框

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

LW = 1.8                  # 导线线宽
FS = 12                   # 标注字号
BLUE = "#1f3f7a"          # 主色（导线、元件）
RED = "#c0392b"           # 强调色（节点、端子）
GRAY = "#555555"


# ================================================================ 绘图图元
def wire(ax, *pts, lw=LW, color=BLUE, ls="-"):
    """折线导线，pts 为 (x, y) 序列"""
    # ⚠️ 必须用 linestyle= 关键字; 写成 plot(x, y, ls) 会把线型当格式字符串解析,
    #    "-" 恰好合法所以看不出问题, 虚线元组 (0, (4, 3)) 会直接报 ValueError
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw,
            linestyle=ls, solid_capstyle="round", zorder=2)


def dashed(ax, *pts, color=GRAY):
    wire(ax, *pts, lw=1.1, color=color, ls=(0, (4, 3)))


def node_dot(ax, x, y, color=RED, ms=6):
    ax.plot([x], [y], "o", color=color, ms=ms, zorder=5)


def terminal(ax, x, y, text, dx=0.0, dy=0.35, color=RED, fs=FS, ha="center", va="bottom"):
    """端子（开口圆圈）+ 标签"""
    ax.plot([x], [y], "o", mfc="white", mec=color, mew=1.6, ms=8, zorder=6)
    ax.text(x + dx, y + dy, text, color=color, fontsize=fs, ha=ha, va=va,
            fontweight="bold", zorder=7)


def resistor(ax, x1, y1, x2, y2, label="", value="", side=1, box=(1.6, 0.62)):
    """电阻（IEC 矩形符号）。orient 由端点坐标自动判断。"""
    horizontal = abs(x2 - x1) > abs(y2 - y1)
    if horizontal:
        cx, cy = (x1 + x2) / 2, y1
        # 框长要按可用跨距收缩, 否则框会把引线和节点一起吞进去
        w, h = min(box[0], 0.72 * abs(x2 - x1)), box[1]
        wire(ax, (x1, y1), (cx - w / 2, cy))
        wire(ax, (cx + w / 2, cy), (x2, y2))
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="white",
                               edgecolor=BLUE, lw=LW, zorder=3))
        ax.text(cx, cy + side * (h / 2 + 0.2), label, ha="center", va="bottom",
                fontsize=FS, color=BLUE, fontweight="bold", zorder=7)
        if value:
            ax.text(cx, cy - side * (h / 2 + 0.2), value, ha="center", va="top",
                    fontsize=FS, color=GRAY, zorder=7)
    else:
        cx, cy = x1, (y1 + y2) / 2
        h, w = min(box[0], 0.72 * abs(y2 - y1)), box[1]
        wire(ax, (x1, y1), (cx, cy - h / 2))
        wire(ax, (cx, cy + h / 2), (x2, y2))
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="white",
                               edgecolor=BLUE, lw=LW, zorder=3))
        ax.text(cx + side * (w / 2 + 0.25), cy + 0.18, label,
                ha="left" if side > 0 else "right", va="center",
                fontsize=FS, color=BLUE, fontweight="bold", zorder=7)
        if value:
            ax.text(cx + side * (w / 2 + 0.25), cy - 0.32, value,
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=FS, color=GRAY, zorder=7)


def capacitor(ax, x1, y1, x2, y2, label="", value="", side=1, gap=0.26, plate=0.85):
    """电容（两条平行板）。orient 由端点坐标自动判断。"""
    horizontal = abs(x2 - x1) > abs(y2 - y1)
    if horizontal:
        cx, cy = (x1 + x2) / 2, y1
        wire(ax, (x1, y1), (cx - gap, cy))
        wire(ax, (cx + gap, cy), (x2, y2))
        for dx in (-gap, gap):
            ax.plot([cx + dx, cx + dx], [cy - plate / 2, cy + plate / 2],
                    color=BLUE, lw=LW + 0.6, zorder=3, solid_capstyle="butt")
        ax.text(cx, cy + plate / 2 + 0.22, label, ha="center", va="bottom",
                fontsize=FS, color=BLUE, fontweight="bold", zorder=7)
        if value:
            ax.text(cx, cy - plate / 2 - 0.22, value, ha="center", va="top",
                    fontsize=FS, color=GRAY, zorder=7)
    else:
        cx, cy = x1, (y1 + y2) / 2
        wire(ax, (x1, y1), (cx, cy - gap))
        wire(ax, (cx, cy + gap), (x2, y2))
        for dy in (-gap, gap):
            ax.plot([cx - plate / 2, cx + plate / 2], [cy + dy, cy + dy],
                    color=BLUE, lw=LW + 0.6, zorder=3, solid_capstyle="butt")
        ax.text(cx + side * (plate / 2 + 0.25), cy + 0.18, label,
                ha="left" if side > 0 else "right", va="center",
                fontsize=FS, color=BLUE, fontweight="bold", zorder=7)
        if value:
            ax.text(cx + side * (plate / 2 + 0.25), cy - 0.32, value,
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=FS, color=GRAY, zorder=7)


def v_source(ax, x, y1, y2, label="", value="", glyph="dc", side=-1, r=0.75):
    """电压源圆圈。glyph='dc' 画 +/−，'square' 画方波，'sine' 画正弦。"""
    cy = (y1 + y2) / 2
    wire(ax, (x, y1), (x, cy - r))
    wire(ax, (x, cy + r), (x, y2))
    ax.add_patch(Circle((x, cy), r, facecolor="white", edgecolor=BLUE, lw=LW, zorder=3))
    s = r * 0.5
    if glyph == "square":
        ax.plot([x - s * 1.1, x - s * 0.37, x - s * 0.37, x + s * 0.37, x + s * 0.37, x + s * 1.1],
                [cy - s * 0.52, cy - s * 0.52, cy + s * 0.52, cy + s * 0.52,
                 cy - s * 0.52, cy - s * 0.52], color=BLUE, lw=1.6, zorder=4)
    elif glyph == "sine":
        import numpy as np
        tt = np.linspace(-1, 1, 60)
        ax.plot(x + tt * s * 1.1, cy + np.sin(tt * np.pi) * s * 0.6,
                color=BLUE, lw=1.6, zorder=4)
    else:   # dc：两条长短线
        ax.plot([x - s * 0.5, x + s * 0.5], [cy + s * 0.42, cy + s * 0.42],
                color=BLUE, lw=1.8, zorder=4)
        ax.plot([x - s * 0.8, x + s * 0.8], [cy - s * 0.42, cy - s * 0.42],
                color=BLUE, lw=1.8, zorder=4)
    ax.text(x - r - 0.16, cy + r * 0.74, "+", ha="right", va="center",
            fontsize=FS + 2, color=BLUE, fontweight="bold", zorder=7)
    ax.text(x - r - 0.16, cy - r * 0.74, "-", ha="right", va="center",
            fontsize=FS + 2, color=BLUE, fontweight="bold", zorder=7)
    if label:
        ax.text(x + side * (r + 0.25), cy + 0.18, label,
                ha="left" if side > 0 else "right", va="center",
                fontsize=FS, color=BLUE, fontweight="bold", zorder=7)
    if value:
        ax.text(x + side * (r + 0.25), cy - 0.32, value,
                ha="left" if side > 0 else "right", va="center",
                fontsize=FS, color=GRAY, zorder=7)


def ground(ax, x, y, size=0.42, label="GND", up=False):
    """接地符号（三根递减横线）。up=True 时符号朝上画（用于「交流地」）。"""
    d = 1 if up else -1
    wire(ax, (x, y), (x, y + d * size * 0.6))
    for i, w in enumerate((size * 1.5, size * 1.0, size * 0.55)):
        yy = y + d * (size * 0.6 + i * size * 0.42)
        ax.plot([x - w / 2, x + w / 2], [yy, yy], color=BLUE, lw=LW, zorder=3,
                solid_capstyle="butt")
    if label:
        ax.text(x, y + d * (size * 0.6 + 3 * size * 0.42 + 0.06), label,
                ha="center", va="bottom" if up else "top",
                fontsize=FS - 1, color=GRAY, zorder=7)


def mosfet_nmos(ax, xg, yc, label="T", model="", lead_x=None):
    """NMOS 符号：栅极竖条 + 三段沟道条 + 漏/源/衬底引线。
    返回 (漏节点, 源节点) 坐标，供外部接线。衬底与源极在外部短接。
    lead_x 给定则三条引线都拐到该 x（便于和外部竖线正交相接）。"""
    ch = xg + 0.42                      # 沟道条的 x
    xd = ch + 0.95 if lead_x is None else lead_x
    gb_t, gb_b = yc + 0.55, yc - 0.55
    ch_t, ch_b = yc + 1.05, yc - 1.05
    ax.plot([xg, xg], [gb_b, gb_t], color=BLUE, lw=LW + 0.6, zorder=3,
            solid_capstyle="butt")      # 栅极条
    for a, b in ((ch_b, ch_b + 0.42), (yc - 0.30, yc + 0.30), (ch_t - 0.42, ch_t)):
        ax.plot([ch, ch], [a, b], color=BLUE, lw=LW + 0.6, zorder=3,
                solid_capstyle="butt")  # 沟道条（三段）
    wire(ax, (ch, ch_t), (xd, ch_t))    # 漏
    wire(ax, (ch, ch_b), (xd, ch_b))    # 源
    wire(ax, (ch, yc), (xd, yc))        # 衬底
    ax.add_patch(FancyArrowPatch((xd - 0.16, yc), (ch + 0.06, yc),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=BLUE, lw=1.4, zorder=4))
    ax.text(xg - 0.12, yc, label, ha="right", va="center", fontsize=FS,
            color=BLUE, fontweight="bold", zorder=7)
    if model:
        ax.text(xd + 0.62, yc - 0.02, model, ha="left", va="top", fontsize=FS - 1,
                color=GRAY, zorder=7)
    return (xd, ch_t), (xd, ch_b)


def isource(ax, x, y1, y2, label="", value="", side=1, r=0.55):
    """受控电流源（圆圈 + 向下箭头）"""
    cy = (y1 + y2) / 2
    wire(ax, (x, y1), (x, cy + r))
    wire(ax, (x, cy - r), (x, y2))
    ax.add_patch(Circle((x, cy), r, facecolor="white", edgecolor=BLUE, lw=LW, zorder=3))
    ax.add_patch(FancyArrowPatch((x, cy + r * 0.6), (x, cy - r * 0.6),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=BLUE, lw=1.5, zorder=4))
    ax.text(x + side * (r + 0.22), cy + 0.16, label,
            ha="left" if side > 0 else "right", va="center", fontsize=FS,
            color=BLUE, fontweight="bold", zorder=7)
    if value:
        ax.text(x + side * (r + 0.22), cy - 0.34, value,
                ha="left" if side > 0 else "right", va="center", fontsize=FS,
                color=GRAY, zorder=7)


def vgs_arrow(ax, x, y_top, y_bot, label="$v_{gs}$"):
    """栅源电压 vgs 的双向箭头"""
    ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot),
                                 arrowstyle="<|-|>", mutation_scale=12,
                                 color=RED, lw=1.3, zorder=4))
    ax.text(x - 0.16, (y_top + y_bot) / 2, label, ha="right", va="center",
            fontsize=FS, color=RED, fontweight="bold", zorder=7)


def finish(ax, title, xlim, ylim):
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_title(title, fontsize=FS + 3, color="#222222", pad=14)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  已生成 {path}")


# ================================================================ 电路① RC 低通
def draw_rc():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    YT, YB = 3.0, 0.0
    XS, XR1, XR2, XC = 0.6, 1.4, 3.2, 5.4

    v_source(ax, XS, YB, YT, label="$v_{in}$", value="方波 0~5 V / 250 Hz",
             glyph="square", side=1)
    wire(ax, (XS, YT), (XR1, YT))
    resistor(ax, XR1, YT, XR2, YT, label="$R$", value="1 kΩ", side=1)
    wire(ax, (XR2, YT), (XC, YT))
    node_dot(ax, XC, YT)
    capacitor(ax, XC, YB, XC, YT, label="$C$", value="100 nF", side=1)
    wire(ax, (XS, YB), (XC, YB))
    ground(ax, XC, YB, label="GND")

    wire(ax, (XC, YT), (XC + 1.3, YT))
    terminal(ax, XC + 1.3, YT, "$v_{out}$", dy=0.3)

    finish(ax, "电路①  RC 低通滤波电路\n（输出取自电容两端，高频被旁路到地）",
           (-2.6, 8.0), (-1.5, 4.6))
    save(fig, "rc_circuit.png")


# ================================================================ 电路② 戴维南
def draw_thevenin():
    """电路② 含源二端网络：两条 R-V 串联支路并联，端口在右侧（A 为 +）。

    拓扑与手绘原图一致：
        A —— R1(1k) —— n1 —— V1(12V, 正极朝上) —— GND
        A —— R2(3k) —— n2 —— V2(4V,  正极朝上) —— GND
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    Y_TOP, Y_MID, Y_BOT = 3.9, 2.3, 0.0      # 上轨 / 电阻与电源的分界 / 下轨
    X1, X2 = 2.0, 4.6                        # 两条支路的 x
    X_GND, X_PORT = 1.1, 6.8                 # 接地引出点 / 端口端子

    # ---- 上轨（端口 +）与下轨（端口 −）
    wire(ax, (X1, Y_TOP), (X_PORT, Y_TOP))
    wire(ax, (X_GND, Y_BOT), (X_PORT, Y_BOT))
    for x in (X1, X2):
        node_dot(ax, x, Y_TOP)               # 支路与上轨的接点
    for x in (X1, X_GND, X2):
        node_dot(ax, x, Y_BOT)               # 支路与下轨的接点

    # ---- 支路 1：R1(1k) 串 V1(12V)
    resistor(ax, X1, Y_TOP, X1, Y_MID, label="$R_1$", value="1 kΩ", side=-1)
    v_source(ax, X1, Y_BOT, Y_MID, label="$V_1$", value="12 V", glyph="dc", side=1, r=0.6)

    # ---- 支路 2：R2(3k) 串 V2(4V)
    resistor(ax, X2, Y_TOP, X2, Y_MID, label="$R_2$", value="3 kΩ", side=1)
    v_source(ax, X2, Y_BOT, Y_MID, label="$V_2$", value="4 V", glyph="dc", side=1, r=0.6)

    # ---- 接地（下轨即端口 −）
    ground(ax, X_GND, Y_BOT, label="GND（端口 −）")

    # ---- 端口端子
    terminal(ax, X_PORT, Y_TOP, "端口 A（+）", dy=0.30)
    terminal(ax, X_PORT, Y_BOT, "端口 −", dy=-0.32, va="top")

    finish(ax, "电路②  含源二端网络（两条 $R$-$V$ 支路并联，端口为 A 与 GND）\n"
               "求 $V_{oc}$、$I_{sc}$、$R_{th}$，再用戴维南等效替换后接负载验证",
           (-1.3, 9.4), (-1.8, 5.6))
    save(fig, "thevenin_network.png")


# ================================================================ 电路③ 共源放大
def draw_mos_amp(dc_path=False):
    """dc_path=True 时画直流通路（去掉 Cb1 与输入源）"""
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    Y_RAIL, Y_G, Y_GND = 4.6, 2.6, 0.0
    X_RG, X_GATE_END = 1.9, 5.15
    X_D, X_S = 6.9, 6.9

    # ---- 顶部电源轨
    x_rail_l = X_RG if dc_path else 1.9
    wire(ax, (x_rail_l, Y_RAIL), (X_D, Y_RAIL))
    wire(ax, (X_RG, Y_RAIL), (X_RG, Y_RAIL + 0.9))
    terminal(ax, X_RG, Y_RAIL + 0.9, "$V_{DD}$ = 5 V", dy=0.28)

    # ---- 分压偏置
    resistor(ax, X_RG, Y_RAIL, X_RG, Y_G, label="$R_{g1}$", value="60 kΩ", side=-1)
    node_dot(ax, X_RG, Y_G)
    resistor(ax, X_RG, Y_G, X_RG, Y_GND + 0.75, label="$R_{g2}$", value="40 kΩ", side=-1)
    ground(ax, X_RG, Y_GND + 0.75, label="GND")

    # ---- 栅极引线
    wire(ax, (X_RG, Y_G), (X_GATE_END, Y_G))

    # ---- 输入：vi —— Cb1 —— 栅极（直流通路里没有这两样）
    if not dc_path:
        wire(ax, (-2.5, Y_G), (-1.15, Y_G))
        terminal(ax, -2.5, Y_G, "$v_i$", dx=-0.35, dy=0.0, ha="right", va="center")
        capacitor(ax, -1.15, Y_G, 1.05, Y_G, label="$C_{b1}$", value="10 µF", side=1)

    # ---- 漏极负载 Rd 与 MOS
    # 三条引线都拐到 X_D, 这样衬底/源极可以用一条竖线正交接下地
    (dr_x, dr_y), (sr_x, sr_y) = mosfet_nmos(ax, X_GATE_END, Y_G, label="T",
                                             model="NMOS", lead_x=X_D)
    # 漏极向上接 Rd
    resistor(ax, X_D, Y_RAIL, X_D, dr_y, label="$R_d$", value="2 kΩ", side=1)
    node_dot(ax, X_D, dr_y)
    # 源极[s]与衬底[B] 短接后接地: 一条竖线从衬底高度贯到地, 源极引线在此打点相接
    wire(ax, (X_D, Y_G), (X_D, Y_GND + 0.75))
    node_dot(ax, X_D, sr_y)
    ground(ax, X_D, Y_GND + 0.75, label="GND")

    # ---- 输出端子 vo
    wire(ax, (X_D, dr_y), (X_D + 1.15, dr_y))
    terminal(ax, X_D + 1.15, dr_y, "$v_o$", dx=0.3, dy=0.0, ha="left", va="center")

    note = "（$C_{b1}$ 开路：隔直电容在直流下相当于断路，输入支路消失）" if dc_path else ""
    finish(ax, ("电路③  NMOS 共源级放大电路" + ("——直流通路" if dc_path else "")) +
               f"\n{'静态偏置：' + note if dc_path else '源极[s]与衬底[B]短接接地，输入经 $C_{{b1}}$ 耦合，输出取自漏极'}",
           (-4.0, 9.4), (-1.6, 6.4))
    save(fig, "mos_dc_path.png" if dc_path else "mos_circuit.png")


# ================================================================ 电路③ 小信号模型
def draw_mos_small_signal():
    """小信号等效模型。

    ⚠️ 这里**不画 MOS 管符号**——模型的定义就是把管子换成
       「栅极开路 + 漏源之间并联 g_m·v_gs 受控电流源和 r_o」。
       再把管子符号画出来是多余的，而且会让图挤成一团。
    """
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    Y_G, Y_BOT = 4.2, 0.9          # 栅极高度 / 源极（地）轨
    X_G, X_D, X_RO = 2.2, 5.6, 7.2
    X_VGS = 3.2

    # ---- 输入：vi —— 栅极 —— Rg1//Rg2 到地
    wire(ax, (-1.6, Y_G), (X_G, Y_G))
    terminal(ax, -1.6, Y_G, "$v_i$", dx=-0.35, dy=0.0, ha="right", va="center")
    node_dot(ax, X_G, Y_G)
    resistor(ax, X_G, Y_G, X_G, Y_BOT, label="$R_{g1}\\,||\\,R_{g2}$",
             value="24 kΩ", side=-1)
    ground(ax, X_G, Y_BOT, label="GND")

    # ---- vgs：从栅极节点到源极（=地）的电势差
    dashed(ax, (X_G, Y_G), (X_VGS, Y_G))
    dashed(ax, (X_VGS, Y_BOT), (X_G, Y_BOT))
    vgs_arrow(ax, X_VGS, Y_G, Y_BOT, label="$v_{gs}$")
    ax.text(X_VGS - 0.2, Y_G + 0.42, "栅极开路，栅流为 0", ha="center", va="bottom",
            fontsize=FS - 2, color=GRAY, zorder=7)

    # ---- 漏极节点：gm·vgs 向下 + ro 并联 + Rd 接交流地
    node_dot(ax, X_D, Y_G)
    isource(ax, X_D, Y_G, Y_BOT, label="", side=-1, r=0.62)
    ax.text(X_D - 0.95, (Y_G + Y_BOT) / 2, "$g_m v_{gs}$", ha="right", va="center",
            fontsize=FS, color=BLUE, fontweight="bold", zorder=7)
    ground(ax, X_D, Y_BOT, label="")

    # ro 支路：与受控源并联（同一个漏源两端）
    wire(ax, (X_D, Y_G), (X_RO, Y_G))
    wire(ax, (X_D, Y_BOT), (X_RO, Y_BOT))
    resistor(ax, X_RO, Y_G, X_RO, Y_BOT, label="$r_o$", value="62.5 kΩ", side=1)

    # Rd：VDD 交流置零 -> 上端接「交流地」
    resistor(ax, X_D, Y_G + 2.0, X_D, Y_G, label="$R_d$", value="2 kΩ", side=1)
    ground(ax, X_D, Y_G + 2.0, label="VDD 置零 = 交流地", up=True)

    # 输出端子
    wire(ax, (X_RO, Y_G), (X_RO + 1.25, Y_G))
    terminal(ax, X_RO + 1.25, Y_G, "$v_o$", dx=0.32, dy=0.0, ha="left", va="center")

    finish(ax, "电路③  小信号等效模型\n"
               "（直流源置零、$C_{b1}$ 短路、MOS 换成 $g_m v_{gs}$ 受控源并联 $r_o$）\n"
               "$A_v = -g_m(R_d \\parallel r_o)$，输出与输入反相（$A_v$ 带负号）",
           (-3.4, 11.0), (-1.2, 7.6))
    save(fig, "mos_small_signal.png")


if __name__ == "__main__":
    print("绘制电路图 ->", OUT_DIR)
    draw_rc()
    draw_thevenin()
    draw_mos_amp(dc_path=False)
    draw_mos_amp(dc_path=True)
    draw_mos_small_signal()
    print("完成。")
