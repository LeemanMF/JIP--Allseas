"""PIGGY wet-end drawings, v2: two figures with report-size text.

  piggy_wet_end_attachment.pdf : View A (1:40), Section B (1:25), Detail C (1:4)
  piggy_wet_end_cable.pdf      : Detail D (2:1), Detail F (10:1), Detail E (NTS)

Both are 160 mm wide. Include them with width=\\textwidth; if \\textwidth is not 160 mm,
set PAPER_W below to your \\textwidth so the text prints at the intended size.
Geometry in mm; stated scales are true when printed at 100 %.
Item numbers refer to the parts tables in the report (LaTeX); each figure is numbered from 1.
"""
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- fonts (Linux or Windows)
FONT = "DejaVu Sans"
for cand, files in [("FreeSans", ["/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                                  "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]),
                    ("Arial", ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"])]:
    if all(os.path.exists(f) for f in files):
        for f in files:
            fm.fontManager.addfont(f)
        FONT = cand
        break
plt.rcParams.update({"font.family": FONT, "pdf.fonttype": 42, "hatch.linewidth": 0.4})
BOLD = 600 if FONT == "FreeSans" else "bold"

# ---------------------------------------------------------------- style
INK, MID = "#1d1d1f", "#5a5e64"
STEEL_F, STEEL_H = "#dde0e4", "#80858d"
CONC_F, CONC_H = "#eeeae2", "#a39d90"
AC_F, BLOCK_F = "#4f535a", "#c3c8ce"
STRAP_C, ALLOY_C = "#4a4e55", "#8b6f3e"
CABLE, DAS_C, PUMP_C, SPARE_C = "#e0701b", "#e0701b", "#2f6db5", "#ffffff"
FS_T, FS_L, FS_D, FS_B = 9.0, 8.0, 7.5, 7.5        # title, label, dimension, balloon (pt)
LW_O, LW_T, LW_D = 0.85, 0.5, 0.4
BAL_R = 2.45
PAPER_W = 160.0
SHOW_SCALE = False      # scale text next to view titles

fig = ax = None


def new_fig(h):
    global fig, ax
    fig = plt.figure(figsize=(PAPER_W / 25.4, h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAPER_W)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")


class View:
    def __init__(self, px, py, dx, dy, scale):
        self.px, self.py, self.dx, self.dy, self.s = px, py, dx, dy, scale

    def p(self, x, y):
        return (self.px + (x - self.dx) * self.s, self.py + (y - self.dy) * self.s)

    def pts(self, xy):
        xy = np.asarray(xy, float)
        return np.column_stack([self.px + (xy[:, 0] - self.dx) * self.s,
                                self.py + (xy[:, 1] - self.dy) * self.s])


def poly(xy, fc="none", ec=INK, lw=LW_O, hatch=None, hc=MID, z=2, ls="-"):
    ax.add_patch(Polygon(xy, closed=True, facecolor=fc, edgecolor=ec, lw=lw, hatch=hatch,
                         hatchcolor=hc if hatch else None, zorder=z, linestyle=ls,
                         joinstyle="round"))


def line(xy, c=INK, lw=LW_T, ls="-", z=3):
    xy = np.asarray(xy)
    ax.plot(xy[:, 0], xy[:, 1], color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="butt",
            dash_capstyle="butt", solid_joinstyle="round")


def circ(c, r, fc="none", ec=INK, lw=LW_T, z=4, ls="-"):
    ax.add_patch(Circle(c, r, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def text(x, y, s, size=FS_L, ha="left", va="center", weight="normal", c=INK, z=8, rot=0):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, fontweight=weight, color=c,
                   zorder=z, rotation=rot, linespacing=1.15)


def arc_pts(cx, cy, r, a0, a1, n=200):
    a = np.radians(np.linspace(a0, a1, n))
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def annulus(v, r_in, r_out, n=360):
    return v.pts(np.vstack([arc_pts(0, 0, r_out, 0, 360, n), arc_pts(0, 0, r_in, 360, 0, n)]))


def arrowhead(tip, direction, size=1.4, c=INK):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    tip = np.asarray(tip, float)
    base = tip - d * size
    ax.add_patch(Polygon([tip, base + n * size * 0.27, base - n * size * 0.27], closed=True,
                         fc=c, ec=c, lw=0.2, zorder=7))


def dim_h(x0, x1, y, label):
    line([(x0, y), (x1, y)], lw=LW_D, z=6)
    arrowhead((x0, y), (-1, 0))
    arrowhead((x1, y), (1, 0))
    text((x0 + x1) / 2, y + 0.9, label, size=FS_D, ha="center", va="bottom")


def dim_v(x, y0, y1, label="", left=True):
    line([(x, y0), (x, y1)], lw=LW_D, z=6)
    arrowhead((x, y0), (0, -1))
    arrowhead((x, y1), (0, 1))
    if label:
        text(x + (-1.0 if left else 1.0), (y0 + y1) / 2, label, size=FS_D,
             ha="right" if left else "left")


def ext(p0, p1):
    line([p0, p1], lw=LW_D, z=6)


def balloon(target, at, num):
    t, a = np.asarray(target, float), np.asarray(at, float)
    d = t - a
    start = a + d / np.linalg.norm(d) * BAL_R
    line([start, t], lw=LW_T, z=8)
    circ(t, 0.45, fc=INK, ec="none", z=9)
    circ(a, BAL_R, fc="white", ec=INK, lw=0.6, z=9)
    text(a[0], a[1] - 0.05, str(num), size=FS_B, ha="center", weight=BOLD, z=10)


def view_title(x, y, title, scale_txt):
    t = text(x, y, title, size=FS_T, weight=BOLD, va="baseline")
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    line([(bb.x0, y - 1.2), (bb.x1, y - 1.2)], lw=0.7, z=8)
    if SHOW_SCALE:
        text(bb.x1 + 2.0, y, scale_txt, size=FS_L - 0.5, va="baseline", c=MID)


def break_v(x, y0, y1, amp=1.1):
    ym = (y0 + y1) / 2
    line([(x, y0), (x, ym - 1.8), (x - amp, ym - 0.7), (x + amp, ym + 0.7), (x, ym + 1.8),
          (x, y1)], lw=LW_T, z=7)


def centre_line(xy):
    line(xy, c=MID, lw=0.35, ls=(0, (10, 2, 1.5, 2)), z=3)


# ---------------------------------------------------------------- reference geometry (mm)
R_ST_IN, R_ST, R_AC, R_CWC = 579.6, 609.6, 613.6, 673.6     # 48 in, 3LPE 4 mm, CWC 60 mm
D_CAB, DAYLIGHT = 28.0, 50.0
Y_CAB = R_CWC + DAYLIGHT + D_CAB / 2
BLK_HW, BLK_CH, BLK_LEN = 120.0, 22.0, 300.0
BLK_TOP = Y_CAB + D_CAB / 2 + 25.0
POCKET = D_CAB / 2 + 1.0


def blk_lower():
    th = math.degrees(math.asin(BLK_HW / R_CWC))
    return np.vstack([arc_pts(0, 0, R_CWC, 90 + th, 90 - th, 80), [(BLK_HW, Y_CAB)],
                      arc_pts(0, Y_CAB, POCKET, 360, 180, 60), [(-BLK_HW, Y_CAB)]])


def blk_lid():
    return np.vstack([[(-BLK_HW, Y_CAB)], arc_pts(0, Y_CAB, POCKET, 180, 0, 60), [(BLK_HW, Y_CAB)],
                      [(BLK_HW, BLK_TOP - BLK_CH)], [(BLK_HW - BLK_CH, BLK_TOP)],
                      [(-BLK_HW + BLK_CH, BLK_TOP)], [(-BLK_HW, BLK_TOP - BLK_CH)]])


def strap_geom(g):
    r = R_CWC + g
    c = np.array([BLK_HW + g * 0.7, BLK_TOP - BLK_CH + g * 0.3])
    t = math.atan2(c[1], c[0]) - math.acos(r / np.hypot(*c))
    T = np.array([r * math.cos(t), r * math.sin(t)])
    return r, c, math.degrees(t), T


def strap_loop(g):
    r, c, t, _ = strap_geom(g)
    arc = arc_pts(0, 0, r, t, -180 - t, 500)
    top = np.array([(-c[0], c[1]), (-(BLK_HW - BLK_CH) - g * 0.3, BLK_TOP + g),
                    ((BLK_HW - BLK_CH) + g * 0.3, BLK_TOP + g), (c[0], c[1])])
    return np.vstack([arc, top, arc[:1]])


# ================================================================ FIGURE 1: attachment
H1 = 143.0
new_fig(H1)

# ---- View A (1:40)
A = View(px=6.0, py=105.5, dx=0.0, dy=0.0, scale=1 / 40)
X_END, FJ, BLKS = 5600.0, (2500.0, 3100.0), (800.0, 4800.0)
view_title(3.0, 139.5, "VIEW A — SIDE ELEVATION", "SCALE 1:40")
for xs in [(0, FJ[0]), (FJ[1], X_END)]:
    poly(A.pts([(xs[0], -R_CWC), (xs[1], -R_CWC), (xs[1], R_CWC), (xs[0], R_CWC)]),
         fc=CONC_F, ec="none", z=2)
poly(A.pts([(FJ[0], -R_CWC), (FJ[1], -R_CWC), (FJ[1], R_CWC), (FJ[0], R_CWC)]),
     fc="#f7f5f0", ec="none", hatch="xxx", hc="#c3bdb1", z=2)
for yy in (-R_CWC, R_CWC):
    line([A.p(0, yy), A.p(X_END, yy)], lw=LW_O, z=4)
for xx in FJ:
    line([A.p(xx, -R_CWC), A.p(xx, R_CWC)], lw=LW_T, z=4)
centre_line([A.p(-150, 0), A.p(X_END + 150, 0)])
for xx in (0, X_END):
    break_v(A.p(xx, 0)[0], A.p(0, -R_CWC)[1], A.p(0, R_CWC)[1])
line([A.p(0, Y_CAB), A.p(X_END, Y_CAB)], c=CABLE, lw=D_CAB * A.s * 72 / 25.4, z=5)
for sgn in (1, -1):
    line([A.p(0, Y_CAB + sgn * D_CAB / 2), A.p(X_END, Y_CAB + sgn * D_CAB / 2)], lw=0.2, z=5)
for bx in BLKS:
    poly(A.pts([(bx - BLK_LEN / 2, R_CWC), (bx + BLK_LEN / 2, R_CWC), (bx + BLK_LEN / 2, BLK_TOP),
                (bx - BLK_LEN / 2, BLK_TOP)]), fc=BLOCK_F, ec=INK, lw=0.5, z=6)
    for sx, w, cc in ((-100, 32, STRAP_C), (0, 19, ALLOY_C), (100, 32, STRAP_C)):
        poly(A.pts([(bx + sx - w / 2, -R_CWC - 3), (bx + sx + w / 2, -R_CWC - 3),
                    (bx + sx + w / 2, BLK_TOP + 4), (bx + sx - w / 2, BLK_TOP + 4)]),
             fc=cc, ec="none", z=7)
bxp = A.p(BLKS[1], 0)[0]
for yy, sgn in ((A.p(0, BLK_TOP)[1] + 3.0, 1), (A.p(0, -R_CWC)[1] - 3.0, -1)):
    line([(bxp, yy), (bxp, yy - sgn * 1.8)], lw=1.1, z=8)
    line([(bxp, yy), (bxp + 3.4, yy)], lw=0.55, z=8)
    arrowhead((bxp + 4.2, yy), (1, 0), size=1.6)
    text(bxp + 5.0, yy, "B", size=FS_T, weight=BOLD)
yd = A.p(0, BLK_TOP)[1] + 6.5
xb0, xb1 = A.p(BLKS[0], 0)[0], A.p(BLKS[1], 0)[0]
ext((xb0, A.p(0, BLK_TOP)[1] + 1.2), (xb0, yd + 1.1))
ext((xb1, A.p(0, BLK_TOP)[1] + 5.0), (xb1, yd + 1.1))
dim_h(xb0, xb1, yd, "4 000 (start value; 3 blocks per 12.2 m joint)")
xd = A.p(X_END, 0)[0] + 4.5
for yy in (R_CWC, -R_CWC):
    ext((A.p(X_END, 0)[0] + 0.8, A.p(0, yy)[1]), (xd + 0.9, A.p(0, yy)[1]))
dim_v(xd, A.p(0, -R_CWC)[1], A.p(0, R_CWC)[1])
text(xd + 1.2, A.p(0, 0)[1], "Ø1 347", size=FS_D, ha="left", rot=90)
balloon(A.p(350, Y_CAB), (10.0, 131.0), 5)
balloon(A.p(BLKS[0] - 130, BLK_TOP - 30), (18.0, 131.0), 6)
balloon(A.p(BLKS[0] - 100, -380), (16.5, 84.0), 8)
balloon(A.p(BLKS[0], -500), (25.5, 84.0), 9)
balloon(A.p(2800, -300), (76.0, 84.0), 4)
balloon(A.p(5300, -300), (143.0, 84.0), 3)

# ---- Section B (1:25)
B = View(px=32.0, py=37.0, dx=0.0, dy=0.0, scale=1 / 25)
view_title(3.0, 76.0, "SECTION B", "SCALE 1:25")
poly(annulus(B, R_ST_IN, R_ST), fc=STEEL_F, ec=INK, lw=LW_O, hatch="/////", hc=STEEL_H, z=2)
poly(annulus(B, R_ST, R_AC), fc=AC_F, ec="none", z=2)
poly(annulus(B, R_AC, R_CWC), fc=CONC_F, ec=INK, lw=LW_O, hatch="...", hc=CONC_H, z=2)
centre_line([B.p(-R_CWC - 70, 0), B.p(R_CWC + 70, 0)])
centre_line([B.p(0, -R_CWC - 70), B.p(0, BLK_TOP + 60)])
line(B.pts(strap_loop(5.0)), c=STRAP_C, lw=0.6, z=5)
poly(B.pts(blk_lower()), fc=BLOCK_F, ec=INK, lw=0.4, z=6)
poly(B.pts(blk_lid()), fc=BLOCK_F, ec=INK, lw=0.4, z=6)
circ(B.p(0, Y_CAB), D_CAB / 2 * B.s, fc=CABLE, ec=INK, lw=0.3, z=7)
cC = B.p(0, Y_CAB - 30)
circ(cC, 7.0, ec=INK, lw=0.5, ls=(0, (3, 1.6)), z=8)
text(cC[0] + 5.8, cC[1] + 6.2, "C", size=FS_T, weight=BOLD)
for ang, r, at, n in ((-128, 595, (6.0, 9.0), 1), (-100, 611.6, (16.0, 4.5), 2),
                      (-50, 645, (58.0, 9.0), 3)):
    a = math.radians(ang)
    balloon(B.p(r * math.cos(a), r * math.sin(a)), at, n)
balloon(B.p(0, Y_CAB), (48.0, 70.5), 5)

# ---- Detail C (1:4)
C = View(px=113.0, py=55.0, dx=0.0, dy=Y_CAB, scale=1 / 4)
view_title(70.0, 76.0, "DETAIL C", "SCALE 1:4")
XW, Y_CUT = 168.0, 585.0


def clip_arc(r):
    a = math.degrees(math.asin(XW / r))
    return arc_pts(0, 0, r, 90 + a, 90 - a, 160)


poly(C.pts(np.vstack([clip_arc(R_CWC), clip_arc(R_AC)[::-1]])), fc=CONC_F, ec="none",
     hatch="...", hc=CONC_H, z=2)
poly(C.pts(np.vstack([clip_arc(R_AC), clip_arc(R_ST)[::-1]])), fc=AC_F, ec="none", z=2)
st = clip_arc(R_ST)
poly(C.pts(np.vstack([st, [(XW, Y_CUT), (-XW, Y_CUT)]])), fc=STEEL_F, ec="none",
     hatch="/////", hc=STEEL_H, z=1)
line(C.pts(clip_arc(R_CWC)), lw=LW_O, z=4)
line(C.pts(st), lw=LW_T, z=4)
for sx in (-XW, XW):
    break_v(C.p(sx, 0)[0], C.p(0, Y_CUT)[1], C.p(0, R_CWC + 8)[1])
line([C.p(-XW, Y_CUT), C.p(XW, Y_CUT)], lw=LW_T, ls=(0, (1.6, 1.2)), z=4)
_, c_r, _, T_r = strap_geom(1.8)
E_r = c_r + (XW - c_r[0]) / (T_r[0] - c_r[0]) * (T_r - c_r)
line(C.pts([(-E_r[0], E_r[1]), (-c_r[0], c_r[1]), (-(BLK_HW - BLK_CH) - 0.5, BLK_TOP + 1.8),
            ((BLK_HW - BLK_CH) + 0.5, BLK_TOP + 1.8), (c_r[0], c_r[1]), (E_r[0], E_r[1])]),
     c=STRAP_C, lw=1.0, z=7)
poly(C.pts(blk_lower()), fc=BLOCK_F, ec=INK, lw=LW_O, z=6)
poly(C.pts(blk_lid()), fc=BLOCK_F, ec=INK, lw=LW_O, z=6)
circ(C.p(0, Y_CAB), D_CAB / 2 * C.s, fc=CABLE, ec=INK, lw=0.55, z=7)
circ(C.p(0, Y_CAB), 7.0 * C.s, fc="#f2c9a4", ec=INK, lw=0.3, z=7)
centre_line([C.p(0, Y_CUT + 5), C.p(0, BLK_TOP + 18)])
yt = C.p(0, BLK_TOP)[1]
for sx in (-BLK_HW, BLK_HW):
    ext((C.p(sx, 0)[0], C.p(0, BLK_TOP - BLK_CH)[1] + 0.7), (C.p(sx, 0)[0], yt + 6.2))
dim_h(C.p(-BLK_HW, 0)[0], C.p(BLK_HW, 0)[0], yt + 5.2, "240 (indicative)")
XD = -32.0
xdl = C.p(XD, 0)[0]
y_top_c = C.p(0, math.sqrt(R_CWC ** 2 - XD ** 2))[1]
y_cab_b = C.p(0, Y_CAB - D_CAB / 2)[1]
ext((C.p(-1.0, 0)[0], y_cab_b), (xdl - 0.9, y_cab_b))
dim_v(xdl, y_top_c, y_cab_b, "50", left=True)
text(xdl - 1.0, (y_top_c + y_cab_b) / 2 - 3.0, "daylight", size=FS_D - 0.5, ha="right", c=MID)
balloon(C.p(-95, BLK_TOP - 12), (78.0, 65.0), 7)
balloon(C.p(-90, Y_CAB - 32), (78.0, 46.0), 6)
balloon(C.p(-6, Y_CAB + 8), (102.0, 69.0), 5)
balloon(C.p(*(c_r + 0.45 * (E_r - c_r))), (153.5, 61.0), 8)
def on_r(x, r):
    """Point at horizontal position x on the circle of radius r (data mm)."""
    return C.p(x, math.sqrt(r ** 2 - x ** 2))


balloon(on_r(125, (R_AC + R_CWC) / 2), (153.5, 32.0), 3)   # mid concrete
balloon(on_r(112, (R_ST + R_AC) / 2), (153.5, 24.5), 2)    # mid 3LPE band
balloon(on_r(60, R_ST - 12), (133.0, 9.0), 1)               # steel wall (balloon below cut)

fig.savefig(os.path.join(HERE, "piggy_wet_end_attachment.pdf"))
fig.savefig(os.path.join(HERE, "piggy_wet_end_attachment.png"), dpi=300)

# ================================================================ FIGURE 2: cable and in-line parts
H2 = 137.0
new_fig(H2)

# ---- Detail D (2:1)
D = View(px=40.0, py=88.0, dx=0.0, dy=0.0, scale=2.0)
view_title(3.0, 133.5, "DETAIL A — SENSING CABLE", "SCALE 2:1")
R_OUT, R_BED, R_IS, R_TUBE, R_TUBE_IN = 14.0, 8.8, 5.6, 2.25, 1.85
circ(D.p(0, 0), R_OUT * D.s, fc="#2f3135", ec=INK, lw=LW_O, z=2)
r2, d2 = 10.1, 2.4
n2 = int(2 * math.pi * r2 / (d2 * 1.04))
for k in range(n2):
    a = 2 * math.pi * k / n2
    circ(D.p(r2 * math.cos(a), r2 * math.sin(a)), d2 / 2 * 0.96 * D.s, fc="#b9bdc3", lw=0.3, z=3)
circ(D.p(0, 0), R_BED * D.s, fc="#6f737a", lw=0.3, z=3)
r1, d1 = 7.2, 2.0
n1 = int(2 * math.pi * r1 / (d1 * 1.04))
for k in range(n1):
    a = 2 * math.pi * k / n1 + 0.12
    circ(D.p(r1 * math.cos(a), r1 * math.sin(a)), d1 / 2 * 0.96 * D.s, fc="#b9bdc3", lw=0.3, z=4)
circ(D.p(0, 0), R_IS * D.s, fc="#45484d", lw=0.3, z=4)
circ(D.p(0, 0), R_TUBE * D.s, fc="#c8ccd2", lw=0.3, z=5)
circ(D.p(0, 0), R_TUBE_IN * D.s, fc="#fff3c4", lw=0.2, z=5)
cF = D.p(0, 0)
circ(cF, 6.6, ec="white", lw=0.6, ls=(0, (3, 1.6)), z=8)
text(cF[0] + 4.8, cF[1] + 5.4, "B", size=FS_T, weight=BOLD, c="white")
yD = D.p(0, R_OUT)[1] + 3.2
for sx in (-R_OUT, R_OUT):
    ext((D.p(sx, 0)[0], D.p(0, 0)[1] + 0.6), (D.p(sx, 0)[0], yD + 1.1))
dim_h(D.p(-R_OUT, 0)[0], D.p(R_OUT, 0)[0], yD, "Ø28 (indicative)")
balloon(D.p(R_OUT * 0.74, -R_OUT * 0.70), (82.0, 58.0), 6)
balloon(D.p(r2 * math.cos(-0.2), r2 * math.sin(-0.2)), (82.0, 79.0), 4)
balloon(D.p(R_BED * math.cos(0.5) - 0.2, R_BED * math.sin(0.5) - 0.2), (82.0, 104.0), 5)
balloon(D.p(r1 * math.cos(2.3), r1 * math.sin(2.3)), (4.0, 113.0), 4)
balloon(D.p(-5.2, -2.6), (4.0, 64.0), 3)

# ---- Detail F (10:1)
F = View(px=122.0, py=91.0, dx=0.0, dy=0.0, scale=10.0)
view_title(98.0, 133.5, "DETAIL B", "SCALE 10:1")
circ(F.p(0, 0), R_TUBE * F.s, fc="#c8ccd2", lw=LW_O, z=4)
circ(F.p(0, 0), R_TUBE_IN * F.s, fc="#fff3c4", lw=0.4, z=5)
fib = [(0, 0)] + [(0.62 * math.cos(math.pi / 2 * k + 0.4), 0.62 * math.sin(math.pi / 2 * k + 0.4))
                  for k in range(4)] + \
      [(1.28 * math.cos(2 * math.pi * k / 7 + 0.2), 1.28 * math.sin(2 * math.pi * k / 7 + 0.2))
       for k in range(7)]
cols = [SPARE_C, DAS_C, PUMP_C, DAS_C, PUMP_C] + [SPARE_C] * 7
for (fx, fy), cc in zip(fib, cols):
    circ(F.p(fx, fy), 0.125 * F.s, fc=cc, lw=0.4, z=6)
yF = F.p(0, R_TUBE)[1] + 3.2
for sx in (-R_TUBE, R_TUBE):
    ext((F.p(sx, 0)[0], F.p(0, 0)[1] + 0.6), (F.p(sx, 0)[0], yF + 1.1))
dim_h(F.p(-R_TUBE, 0)[0], F.p(R_TUBE, 0)[0], yF, "Ø4.5")
balloon(F.p(R_TUBE * math.cos(-0.5), R_TUBE * math.sin(-0.5)), (152.0, 76.0), 2)
a14 = 2 * math.pi * 6 / 7 + 0.2
balloon(F.p(1.28 * math.cos(a14), 1.28 * math.sin(a14)), (152.0, 84.0), 1)
for i, (cc, lab) in enumerate([(DAS_C, "DAS sensing fibre (2)"),
                               (PUMP_C, "Amplifier pump fibre (2, conditional)"),
                               (SPARE_C, "Spare fibre (8)")]):
    yy = 63.0 - i * 4.4
    circ((99.5, yy), 1.15, fc=cc, lw=0.4, z=8)
    text(102.2, yy, lab, size=FS_L - 0.3)

# ---- Detail E (NTS)
view_title(3.0, 44.0, "DETAIL C — IN-LINE COMPONENTS", "NOT TO SCALE")
EX0, EX1 = 3.0, 157.0
Y_CB, Y_CT, Y_C = 9.0, 16.0, 20.6
poly([(EX0, Y_CB), (EX1, Y_CB), (EX1, Y_CT), (EX0, Y_CT)], fc=CONC_F, ec="none",
     hatch="...", hc=CONC_H, z=2)
line([(EX0, Y_CT), (EX1, Y_CT)], lw=LW_O, z=4)
line([(EX0, Y_CB), (EX1, Y_CB)], lw=LW_T, ls=(0, (1.6, 1.2)), z=4)
SEG = [(3.0, 41.0), (41.0, 80.0), (80.0, 119.0), (119.0, 157.0)]
for xb in (41.0, 80.0, 119.0):
    poly([(xb - 1.5, Y_CB - 2), (xb + 1.5, Y_CB - 2), (xb + 1.5, 33.0), (xb - 1.5, 33.0)],
         fc="white", ec="none", z=9)
    break_v(xb - 1.5, Y_CB - 1.5, 32.0)
    break_v(xb + 1.5, Y_CB - 1.5, 32.0)


def mini_block(xc, w=4.4, h=6.6):
    poly([(xc - w / 2, Y_CT), (xc + w / 2, Y_CT), (xc + w / 2, Y_CT + h), (xc - w / 2, Y_CT + h)],
         fc=BLOCK_F, lw=0.5, z=6)


def cable_run(x0, x1):
    line([(x0, Y_C), (x1, Y_C)], c=CABLE, lw=3.4, z=5)


def housing(x0, x1, r, taper, fc, ls="-"):
    poly([(x0 - taper, Y_C - 1.2), (x0, Y_C - r), (x1, Y_C - r), (x1 + taper, Y_C - 1.2),
          (x1 + taper, Y_C + 1.2), (x1, Y_C + r), (x0, Y_C + r), (x0 - taper, Y_C + 1.2)],
         fc=fc, lw=0.6, z=8, ls=ls)


cable_run(3.0, 39.5)
for xc in (12.0, 30.0):
    mini_block(xc)
cable_run(42.5, 78.5)
mini_block(52.5, w=7.0, h=9.0)
mini_block(68.5, w=7.0, h=9.0)
housing(50.0, 71.0, 3.2, 3.5, "#9aa0a8")
cable_run(81.5, 117.5)
housing(89.5, 109.5, 2.9, 3.0, "#eef0f2", ls=(0, (2.6, 1.4)))
cable_run(120.5, 157.0)
for s0, s1 in ((121.0, 132.6), (133.4, 145.0), (145.8, 157.0)):
    poly([(s0, Y_CT), (s1, Y_CT), (s1, Y_CT + 6.0), (s1 - 1.2, Y_CT + 8.2), (s0 + 1.2, Y_CT + 8.2),
          (s0, Y_CT + 6.0)], fc="#7d8590", lw=0.55, z=7)
    line([(s0 + 0.7, Y_C), (s1 - 0.7, Y_C)], c=CABLE, lw=0.8, ls=(0, (1.4, 0.9)), z=8)
balloon((12.0, Y_CT + 5.5), (12.0, 36.5), 7)
balloon((60.5, Y_C + 3.2), (60.5, 36.5), 8)
balloon((99.5, Y_C + 2.9), (99.5, 36.5), 9)
balloon((139.0, Y_CT + 8.2), (139.0, 36.5), 10)
for (x0, x1), lab in zip(SEG, ["Standard span", "Splice at a break point",
                               "Only where hosts are\n>300 km apart", "Exposed, trawled\nstretches"]):
    text((x0 + x1) / 2, 3.6, lab, size=FS_L - 0.3, ha="center", c=MID)

fig.savefig(os.path.join(HERE, "piggy_wet_end_cable.pdf"))
fig.savefig(os.path.join(HERE, "piggy_wet_end_cable.png"), dpi=300)
print("ok", FONT)
