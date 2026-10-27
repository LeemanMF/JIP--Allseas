"""
PIGGY - Mechanical feasibility of piggybacking the DAS sensing cable during S-lay
=================================================================================

WHAT IS CALCULATED HERE
  C1  Overbend stretching   cable strain if clamped vs sliding, per vessel     EA-RP-009 Eq. 6, 13   -> Figure 1
  C2  Sliding requirement   can block friction stretch the cable?              own check
  C3  Splash zone           wave load per block vs block lateral capacity      EA-RP-009 Eq. 15-19   -> Figure 2
  C4  Rigid bodies          gap between rigid body and pipe on the stinger     chord sag L^2 / (8 R)
  C5  Sagbend               cable bending when overlength makes it bow         EA-RP-009 App. IX
  C6  Quick checks          cable mass for the route, joining radius vs MBR    arithmetic

WHAT IS NOT CALCULATED (referred to sources in the report)
  - Stinger envelope 10-2 o'clock, seabed 9-3 o'clock         T. van Ginneken, interview 23-09-2026
  - Pipeline rotation (15-40 deg after A&R), anti-rotation clamp  Allseas rotation data; EA-RP-009 5.6, 5.9
  - Joining-radius method (plot R_FL on vessel GA)             EA-RP-009 5.4
  - Strapping (2 carbon steel + 1 Alloy 625, order 2-1-3)      EA-RP-009 4.8

HOW TO USE
  1. Add stinger radii to VESSELS (section 1). The smallest radius is the design (harshest) case.
  2. Fill any input marked  None  (section 2). Checks that need it show "MISSING INPUT" until then.
  3. Run:  python piggy_mechanical_feasibility.py      (needs numpy, scipy, matplotlib)

OUTPUT (written next to this file)
  mech_fig1_overbend.pdf/.png       Figure 1: cable strain vs stinger radius, with all vessels
  mech_fig2_splash_zone.pdf/.png    Figure 2: load per block vs block spacing
  mech_summary_table.tex            rows for the summary table in the report
  mech_vessel_table.tex             rows for the vessel table (C1 per vessel)
  results_mechanical_feasibility.csv  all results
"""
import csv
import math
import os

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================================================
# 1. VESSELS  -  stinger radius range per vessel (None = not known yet; the vessel is then skipped)
# =====================================================================================================
VESSELS = [
    # name                  stinger radius range [m]   source
    #                       (min, max) - one value also allowed, e.g. 150.0
    ("Lorelay",             (70.0, 270.0),             "T. van Ginneken (23-09-2026) - confirm"),
    ("Solitaire",           (80.0, 400.0),             ""),
    ("Audacia",             (80.0, 300.0),             ""),
    ("Tog Mor",             None,                      ""),
    ("Pioneering Spirit",   (110.0, 425.0),            ""),
]


# =====================================================================================================
# 2. INPUTS
# =====================================================================================================
# --- Main pipeline (Tier 1 reference case: 48 in trunk line, concrete coated) ---
OD_STEEL = 1.2192          # m    48 in line pipe                                    Design Criteria v0.1
T_COATING = 0.004          # m    3LPE anti-corrosion coating                        ASSUMED - confirm
T_CONCRETE = 0.060         # m    concrete weight coating                            Allseas concrete-crushing memo

# --- Sensing cable (Hexatronic 25-ton DA, datasheet 28701-25ton DA, 2023) ---
D_CABLE = 0.030            # m    outer diameter (datasheet 28 mm; team uses 30 mm)
MASS_CABLE = 2.3           # kg/m mass in air                                         Hexatronic datasheet
MBR_CABLE = 1.5            # m    min. bend radius with tensile load                  Hexatronic datasheet
ARMOUR = [(14, 0.003), (20, 0.003)]   # (number, diameter m) of armour wires          Hexatronic datasheet
E_STEEL = 200e9            # Pa   Young's modulus of steel wire                       ASSUMED (handbook)
STRAIN_LIMIT = 0.0035      # -    max. cable strain without loading the fibre         own design target - confirm

# --- Piggyback blocks ---
DAYLIGHT = 0.050           # m    clearance between coating and cable                 EA-RP-009
BLOCK_SPACING = 4.0        # m    3 blocks per 12.2 m joint (start value)             design choice
LATERAL_CAPACITY = 3924.0  # N    block on concrete coating, moved at 0.40 t          Allseas test EN/13200/280070
LATERAL_CAPACITY_SHAPED = 6377.0   # N  block shaped to the coated pipe, 0.65 t       Allseas test EN/13200/280070
FRICTION_PADDED = (6000.0, 11000.0)  # N  axial friction per block with 3 mm rubber pad  Lankhorst test M06418

# --- Splash zone (EA-RP-009 4.7 and worked example 4.9, Audacia, Hs 3.5 m / Tz 9 s) ---
RHO = 1025.0               # kg/m3 seawater
C_SLAM = 3.0               # -    slamming coefficient                                EA-RP-009 1.4
C_ADDED_MASS = 1.0         # -    added mass coefficient                              EA-RP-009 1.4
C_DRAG_SINGLE = 0.80       # -    drag coefficient single cylinder                    EA-RP-009 4.6.2
R_INTERACTION = 1.73       # -    Kalghatgi & Sayer ratio (1.45-2.00)                 EA-RP-009 Eq. 14
HS, TZ = 3.5, 9.0          # m, s sea state                                           EA-RP-009 Table 4.1
V_CURRENT = 0.772          # m/s  current                                             EA-RP-009 Table 4.2
PIPE_MOTIONS = {           # wave angle [deg]: (pipe velocity m/s, pipe acceleration m/s2)  EA-RP-009 Table 4.1
    0: (0.000, 0.000), 30: (0.713, 0.384), 50: (1.482, 1.156), 70: (1.873, 1.197), 90: (1.527, 0.896),
    110: (1.884, 1.265), 130: (1.512, 0.877), 150: (0.802, 0.437), 180: (0.000, 0.000)}

# --- Rigid bodies passing over the stinger ---
RIGID_BODIES = {           # length [m]
    "MPS shell, 4 m": 4.0,                 # EA-RP-009 2.2.4
    "MPS shell, 6 m": 6.0,                 # EA-RP-009 2.2.4
    "Splice joint housing": None,          # LOOK UP: cable supplier
    "Amplifier housing (conditional)": None,   # LOOK UP: cable / DAS supplier
}

# --- From lay analysis / vessel drawings (LOOK UP) ---
SAGBEND_STRAIN = None      # -    max. bending strain of main line in sagbend          lay analysis
JOINING_RADIUS = None      # m    radius where cable meets main line (R_FL)            vessel GA, EA-RP-009 5.4

# --- Route ---
ROUTE_LENGTH = 500.0       # km   Tier 1 design reference length                      Design Criteria v0.1


# =====================================================================================================
# 3. DERIVED GEOMETRY
# =====================================================================================================
OD_TOTAL = OD_STEEL + 2 * T_COATING + 2 * T_CONCRETE           # pipe outer diameter incl. coatings
AXIS_TO_CABLE = OD_TOTAL / 2 + DAYLIGHT + D_CABLE / 2           # main-line axis -> cable axis
AXIS_TO_CABLE_TOP = OD_TOTAL / 2 + DAYLIGHT + D_CABLE           # main-line axis -> cable outer fibre
R_CRITICAL = AXIS_TO_CABLE / STRAIN_LIMIT - OD_TOTAL / 2      # below this radius a clamped cable is overstrained
EA_CABLE = E_STEEL * sum(n * math.pi * d ** 2 / 4 for n, d in ARMOUR)   # armour only (sheaths ignored)

def _as_range(r):
    return (float(r), float(r)) if isinstance(r, (int, float)) else (float(r[0]), float(r[1]))


KNOWN_VESSELS = [(n, _as_range(r), s) for n, r, s in VESSELS if r is not None]   # (name, (min, max), source)
DESIGN_VESSEL, (R_DESIGN, _), _ = min(KNOWN_VESSELS, key=lambda x: x[1][0])    # smallest radius = harshest


# =====================================================================================================
# 4. RESULT BOOKKEEPING
# =====================================================================================================
RESULTS = []


def result(check, quantity, value, unit="", limit=None, report=True, note=""):
    """Store one result. Status is derived from value <= limit (or >= for 'min' limits)."""
    if value is None:
        status = "MISSING INPUT"
    elif limit is None:
        status = ""
    elif isinstance(limit, tuple):                      # ("min", value): value must be >= limit
        status = "OK" if value >= limit[1] else "NOT OK"
    else:
        status = "OK" if value <= limit else "NOT OK"
    RESULTS.append(dict(check=check, quantity=quantity, value=value, unit=unit,
                        limit=limit[1] if isinstance(limit, tuple) else limit,
                        status=status, report=report, note=note))


def fmt(x, unit):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if unit == "%":
        return f"{100 * x:.2f} %"
    s = f"{x:.0f}" if abs(x) >= 100 else f"{x:.3g}"
    return f"{s} {unit}".strip()


# =====================================================================================================
# 5. CHECKS
# =====================================================================================================
def strain_if_clamped(stinger_radius, clock_deg_from_top=0.0):
    """EA-RP-009 Eq. 6 and 13: strain of a cable that cannot slide over the overbend."""
    offset = AXIS_TO_CABLE * math.cos(math.radians(clock_deg_from_top))   # = dR * sin(alpha)
    return offset / (stinger_radius + OD_TOTAL / 2)


def strain_if_sliding(stinger_radius):
    """Bending strain only: D / (2 R) at the cable's own radius."""
    return D_CABLE / (2 * (stinger_radius + OD_TOTAL / 2 + AXIS_TO_CABLE))


def c1_overbend():
    """C1 - overbend stretching. Design vessel in the summary; every vessel in the vessel table."""
    result("C1", f"Strain if clamped, 12 o'clock ({DESIGN_VESSEL})", strain_if_clamped(R_DESIGN), "%", STRAIN_LIMIT)
    result("C1", f"Strain if clamped, 2 o'clock ({DESIGN_VESSEL})", strain_if_clamped(R_DESIGN, 60), "%", STRAIN_LIMIT)
    result("C1", f"Strain if sliding ({DESIGN_VESSEL})", strain_if_sliding(R_DESIGN), "%", STRAIN_LIMIT)
    result("C1", "Min. stinger radius for a clamped cable", R_CRITICAL, "m", note="below this the cable must slide")


def c2_sliding_requirement():
    """C2 - force needed to stretch the cable to the strain limit vs friction of padded blocks."""
    force_needed = EA_CABLE * STRAIN_LIMIT
    result("C2", "Axial force to stretch cable to strain limit", force_needed / 1e3, "kN",
           note=f"EA = {EA_CABLE / 1e6:.0f} MN (armour wires only)")
    for f in sorted(FRICTION_PADDED, reverse=True):
        n = force_needed / f
        result("C2", f"Padded blocks to reach that force ({f / 1e3:.0f} kN each)", n, "blocks",
               note=f"{n * BLOCK_SPACING:.0f} m of pipe at {BLOCK_SPACING:.0f} m spacing")
    result("C2", "Requirement", "low-friction slot, no rubber pad")


def splash_load_per_block(cable_d, spacing):
    """C3 helper - max. lateral load on one block over all wave angles (EA-RP-009 Eq. 15-19)."""
    area = cable_d * spacing                                   # projected area between two blocks
    volume = math.pi / 4 * cable_d ** 2 * spacing
    c_drag = R_INTERACTION * (1 + cable_d / OD_TOTAL) * C_DRAG_SINGLE          # Eq. 15
    u_wave = math.pi * HS / TZ
    worst = (0.0, None)
    for angle, (v_pipe, a_pipe) in PIPE_MOTIONS.items():
        u = u_wave * abs(math.cos(math.radians(90 - angle)))
        f_slam = 0.5 * RHO * C_SLAM * area * (v_pipe + u) ** 2                                   # Eq. 16-17
        f_sub = 0.5 * RHO * c_drag * area * (v_pipe + V_CURRENT + u) ** 2 \
            + RHO * C_ADDED_MASS * volume * a_pipe                                                # Eq. 18-19
        worst = max(worst, (max(f_slam, f_sub), angle))
    return worst


def c3_splash_zone():
    load, angle = splash_load_per_block(D_CABLE, BLOCK_SPACING)
    result("C3", f"Max. lateral load per block (wave {angle} deg)", load / 1e3, "kN", LATERAL_CAPACITY / 1e3)
    result("C3", "Utilisation, standard block", load / LATERAL_CAPACITY, "", 1.0)
    result("C3", "Utilisation, block shaped to coating", load / LATERAL_CAPACITY_SHAPED, "", 1.0, report=False)
    result("C3", "Max. block spacing at utilisation 1.0", BLOCK_SPACING * LATERAL_CAPACITY / load, "m",
           note="load is linear in spacing")


def c4_rigid_bodies():
    radius = R_DESIGN + OD_TOTAL / 2 + AXIS_TO_CABLE
    for name, length in RIGID_BODIES.items():
        gap = None if length is None else length ** 2 / (8 * radius)
        result("C4", f"Gap over stinger: {name}", None if gap is None else gap * 1e3, "mm", DAYLIGHT * 1e3,
               note="" if gap is None else f"angle change {math.degrees(length / radius):.1f} deg")


def _bow(overlength, span, exact):
    """Amplitude and radius of a sinusoidal bow that absorbs the overlength between two blocks.
    exact=False (default) follows EA-RP-009 App. IX; exact=True uses the full arc length and peak curvature."""
    k = math.pi / span
    if exact:
        arc = lambda q: quad(lambda x: math.sqrt(1 + (q * k * math.cos(k * x)) ** 2), 0, span)[0]
    else:                                   # EA-RP-009 App. IX integrates 1 + y'^2 (no square root)
        arc = lambda q: quad(lambda x: 1 + (q * k * math.cos(k * x)) ** 2, 0, span)[0]
    q = brentq(lambda q: arc(q) - (span + overlength), 1e-6, span)
    if exact:
        r_min = 1 / (q * k ** 2)            # true peak curvature of the sine
    else:                                   # App. IX: circle through both ends and the midpoint
        side = math.hypot(span / 2, q)
        r_min = side * side * span / (2 * span * q)
    return q, r_min


def sagbend_strain(main_strain, span, od_main, d_cable, offset, exact=False):
    r_sagbend = od_main / (2 * main_strain)
    q, r_min = _bow(span * offset / r_sagbend, span, exact)
    return q, r_min, math.hypot(d_cable / (2 * r_min), d_cable / (2 * r_sagbend))


def c5_sagbend():
    # verification: EA-RP-009 App. IX worked example (16 in main line, 3.5 in piggyback, 6 m, 0.15 %)
    ex = dict(main_strain=0.0015, span=6.0, od_main=0.406, d_cable=0.089, offset=(0.406 + 2 * 0.089) / 2 + 0.05)
    q_rp, _, e_rp = sagbend_strain(**ex)
    result("C5", "Verification: App. IX example", f"bow {q_rp:.3f} m, strain {100 * e_rp:.3f} %",
           report=False, note="RP reports 0.136 m and 0.138 %")
    if SAGBEND_STRAIN is None:
        result("C5", "Cable bending strain in sagbend", None, "%", STRAIN_LIMIT, note="needs SAGBEND_STRAIN")
        return
    q, r_min, eps = sagbend_strain(SAGBEND_STRAIN, BLOCK_SPACING, OD_STEEL, D_CABLE, AXIS_TO_CABLE_TOP)
    result("C5", "Cable bending strain in sagbend", eps, "%", STRAIN_LIMIT, note=f"bow {q * 1e3:.0f} mm")
    result("C5", "Min. bow radius", r_min, "m", ("min", MBR_CABLE))


def c6_quick_checks():
    result("C6", "Cable mass for route", MASS_CABLE * ROUTE_LENGTH, "t",
           note=f"{ROUTE_LENGTH:.0f} km x {MASS_CABLE} kg/m")
    result("C6", "Joining radius", JOINING_RADIUS, "m", ("min", MBR_CABLE), note="must be >= cable MBR")


# =====================================================================================================
# 6. OUTPUT
# =====================================================================================================
def print_report():
    print(f"Design vessel: {DESIGN_VESSEL} (stinger radius {R_DESIGN:.0f} m) | cable {D_CABLE * 1e3:.0f} mm | "
          f"blocks every {BLOCK_SPACING:.0f} m\n")
    for r in RESULTS:
        lim = "" if r["limit"] is None else f"limit {fmt(r['limit'], r['unit'])}"
        print(f"{r['check']}  {r['quantity'][:55]:55s} {fmt(r['value'], r['unit']):>34s}  {lim:16s} "
              f"{r['status']:13s} {r['note']}")
    print(f"\nPer vessel (C1) - clamped cable at 12 o'clock exceeds the limit below R = {R_CRITICAL:.0f} m:")
    for name, (r0, r1), _ in KNOWN_VESSELS:
        print(f"  {name:18s} R = {r0:4.0f}-{r1:4.0f} m   clamped 12 o'clock {100 * strain_if_clamped(r0):.2f}-"
              f"{100 * strain_if_clamped(r1):.2f} %   -> {vessel_verdict(r0, r1)}")
    missing = [n for n, r, _ in VESSELS if r is None]
    if missing:
        print("  Stinger radius not yet filled in:", ", ".join(missing))


def vessel_verdict(r0, r1):
    if r0 >= R_CRITICAL:
        return "clamping acceptable over full range"
    if r1 < R_CRITICAL:
        return "cable must slide over full range"
    return f"cable must slide below {R_CRITICAL:.0f} m"


def tex(s):
    return str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("^2", r"$^2$")


def write_tables():
    with open(os.path.join(HERE, "results_mechanical_feasibility.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "quantity", "value", "unit", "limit", "status", "note"])
        for r in RESULTS:
            w.writerow([r[k] for k in ("check", "quantity", "value", "unit", "limit", "status", "note")])

    rows = [f"{r['check']} & {tex(r['quantity'])} & {tex(fmt(r['value'], r['unit']))} & "
            f"{tex(fmt(r['limit'], r['unit']) if r['limit'] is not None else '--')} & {tex(r['status'] or '--')} \\\\"
            for r in RESULTS if r["report"]]
    with open(os.path.join(HERE, "mech_summary_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join([r"\begin{tabularx}{\textwidth}{@{}l X r r l@{}}", r"\toprule",
                           r"\textbf{Check} & \textbf{Quantity} & \textbf{Result} & \textbf{Limit} & \textbf{Status} \\",
                           r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", ""]))

    rows = [f"{tex(n)} & {r0:.0f}--{r1:.0f} & {100 * strain_if_clamped(r0):.2f}--{100 * strain_if_clamped(r1):.2f}\\,\\% & "
            f"{tex(vessel_verdict(r0, r1))} \\\\" for n, (r0, r1), _ in sorted(KNOWN_VESSELS, key=lambda x: x[1][0])]
    with open(os.path.join(HERE, "mech_vessel_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join([r"\begin{tabular}{@{}l r r l@{}}", r"\toprule",
                           r"\textbf{Vessel} & \textbf{Stinger radius [m]} & \textbf{Strain if clamped, 12 o'clock} & "
                           r"\textbf{Result} \\", r"\midrule", *rows,
                           r"\bottomrule", r"\end{tabular}", ""]))


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, accent, grey = "#1d1d1f", "#e0701b", "#62666c"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1 - overbend: strain vs stinger radius (top) and vessel stinger-radius ranges (bottom)
    vessels = sorted(KNOWN_VESSELS, key=lambda x: x[1][0], reverse=True)       # smallest radius (design vessel) at the top
    r_hi = 25 * math.ceil(1.05 * max(r1 for _, (_, r1), _ in vessels) / 25)
    radii = np.linspace(50, r_hi, 400)
    fig, (ax, axv) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 3.0 + 0.22 * len(vessels)),
                                  gridspec_kw={"height_ratios": [3.0, 0.42 * len(vessels)], "hspace": 0.16})
    ax.plot(radii, [100 * strain_if_clamped(r) for r in radii], "-", color=ink, lw=1.3, label="Clamped, 12 o'clock")
    ax.plot(radii, [100 * strain_if_clamped(r, 60) for r in radii], ":", color=ink, lw=1.3, label="Clamped, 2 o'clock")
    slide = "#2f6db5"                                   # own colour so the sliding line doesn't vanish into the axis
    eps_slide = [100 * strain_if_sliding(r) for r in radii]
    ax.plot(radii, eps_slide, color=slide, lw=2.0, label="Sliding", zorder=4, clip_on=False)   # drawn over the x-axis
    ax.annotate(f"Sliding ≈ {100 * strain_if_sliding(R_DESIGN):.2f} %", (R_DESIGN, 100 * strain_if_sliding(R_DESIGN)),
                xytext=(R_DESIGN + 40, 0.06 * 100 * strain_if_clamped(radii[0])), fontsize=7, color=slide, va="bottom",
                arrowprops=dict(arrowstyle="-", color=slide, lw=0.7, shrinkA=0, shrinkB=0))
    ax.axhline(100 * STRAIN_LIMIT, color=accent, lw=1.3, label="Cable strain limit")
    for a_ in (ax, axv):
        a_.axvline(R_CRITICAL, color=grey, lw=0.8, ls="--", zorder=1)
    ax.annotate(f"{R_CRITICAL:.0f} m", (R_CRITICAL, 1.02 * 100 * strain_if_clamped(radii[0])), xytext=(4, 0),
                textcoords="offset points", fontsize=7, color=grey, va="top")
    ax.set_ylabel("Cable strain on overbend [%]")
    ax.set_ylim(0, 1.05 * 100 * strain_if_clamped(radii[0]))
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.tick_params(labelbottom=False)
    for i, (name, (r0, r1), _) in enumerate(vessels):
        lo, hi = r0, min(r1, R_CRITICAL)
        if hi > lo:
            axv.barh(i, hi - lo, left=lo, height=0.55, color=accent, edgecolor="none")
        lo2 = max(r0, R_CRITICAL)
        if r1 > lo2:
            axv.barh(i, r1 - lo2, left=lo2, height=0.55, color="#c9ccd1", edgecolor="none")
    axv.set_yticks(range(len(vessels)), [n for n, _, _ in vessels], fontsize=7)
    axv.tick_params(axis="y", length=0)
    axv.spines["left"].set_visible(False)
    axv.set_ylim(-0.6, len(vessels) - 0.4)
    axv.set_xlabel("Stinger radius [m]")
    axv.set_xlim(50, r_hi)
    from matplotlib.patches import Patch
    axv.legend(handles=[Patch(color=accent, label="Cable must slide"), Patch(color="#c9ccd1", label="Clamping acceptable")],
               frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.0 + 0.62 / len(vessels)), ncol=2, borderaxespad=0.1)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"mech_fig1_overbend.{ext}"), dpi=300)
    plt.close(fig)

    # Figure 2 - splash zone: load per block vs block spacing
    spacings = np.linspace(2, 12.2, 80)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(spacings, [splash_load_per_block(D_CABLE, s)[0] / 1e3 for s in spacings], color=ink, lw=1.2,
            label=f"Load per block, cable Ø{D_CABLE * 1e3:.0f} mm")
    ax.axhline(LATERAL_CAPACITY / 1e3, color=accent, lw=1.3, label="Block capacity, standard")
    ax.axhline(LATERAL_CAPACITY_SHAPED / 1e3, color=accent, lw=1.3, ls="--", label="Block capacity, shaped to pipe")
    ax.axvline(BLOCK_SPACING, color=grey, lw=0.8, ls="-.")
    ax.text(BLOCK_SPACING + 0.15, 0.3, "design spacing", color=grey, fontsize=7)
    ax.set_xlabel("Block spacing [m]")
    ax.set_ylabel("Max. lateral load per block [kN]")
    ax.set_ylim(0, 1.15 * LATERAL_CAPACITY_SHAPED / 1e3)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"mech_fig2_splash_zone.{ext}"), dpi=300)
    plt.close(fig)


# =====================================================================================================
if __name__ == "__main__":
    c1_overbend()
    c2_sliding_requirement()
    c3_splash_zone()
    c4_rigid_bodies()
    c5_sagbend()
    c6_quick_checks()
    print_report()
    write_tables()
    make_figures()
