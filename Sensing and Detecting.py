"""
Sensing and detection performance of the piggybacked DAS cable (Nord Stream 2 reference case)
======================================================================================================

WHAT IS CALCULATED HERE
  S1  Optical reach        one-way loss budget -> fibre length one interrogator channel can read
  S2  Route coverage       monitored share of Nord Stream 2 with hosts at both landfalls        -> Figure 1
  S3  Bandwidth            highest sample rate and Nyquist frequency against fibre length       c / (2 n L)
  S4  Noise                interrogator self-noise along the fibre, as noise-equivalent pressure -> Figure 2
  S5  Coupling             stick/slip of the cable in the blocks, pipe-breathing coupling factor
  S6  Detection            sonar equation per threat, detection distance along the fibre         -> Figure 3
  S7  Quiet targets        required source level, scaling of hydrophone detection ranges
  S8  Localisation, alert latency, raw data volume, optical reserve for repairs (used in 4.4)

MODEL IN ONE LINE
  A DAS channel is treated as a hydrophone whose self-noise is the interrogator noise converted to an
  acoustic pressure through the coupling factor kappa. A threat is detected when the passive sonar
  equation gives a non-negative signal excess:
        SE = SL_f - TL(R) - NL_f(z, kappa) + 5 log10(B T) - DT  >= 0          (energy detector)

HOW TO USE
  Run:  python piggy_sensing_performance.py        (needs numpy, scipy, matplotlib)
  All inputs are in section 1. Values marked ASSUMED are engineering assumptions of this study.

OUTPUT (written next to this file)
  sens_fig1_coverage.pdf/.png        Figure 1: optical reach on the Nord Stream 2 route
  sens_fig2_noise_band.pdf/.png      Figure 2: noise-equivalent pressure along the fibre, and band vs span
  sens_fig3_detection.pdf/.png       Figure 3: detection distance along the fibre per threat
  results_sensing_performance.csv    all results
"""
import csv
import math
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================================================
# 1. INPUTS
# =====================================================================================================
# --- Physical constants and Baltic water (5 degC, ~10 psu; same values as the earlier detection model) ---
C_LIGHT = 2.998e8          # m/s
N_EFF = 1.4685             # -      effective refractive index used in the ONYX datasheet (footnote 5)
RHO_W = 1010.0             # kg/m3  Baltic bottom water                                  ASSUMED
C_W = 1443.0               # m/s    speed of sound, Baltic bottom water                    ASSUMED
H_REF = 80.0               # m      representative water depth (route 15-210 m)            Allseas project page

# --- Interrogator: SLB ONYX peta EX (single channel, extended range) - datasheet received Oct 2026 ---
BUDGET_DB = 34.0           # dB     max. one-way optical loss budget per fibre              datasheet
FIBRE_LOSS = 0.20          # dB/km  G.652 fibre at 1550 nm                                   datasheet
GAUGE = 6.4                # m      gauge length of the noise test                           datasheet (SEAFOM MSP-02)
RANGE_EX = 150.0           # km     standard range, single channel                           datasheet
# Self-noise against distance at 100 Hz, 6.4 m gauge, 500 Hz sample rate (SEAFOM MSP-02 test, 150 km fibre),
# digitised from the datasheet plot (accuracy about +/- 1 dB). Unit: dB re 1 rad/sqrt(Hz).
EX_NOISE_Z_KM = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100, 112.5, 125, 137.5, 150]
EX_NOISE_DB_RAD = [-61.0, -60.4, -59.7, -58.8, -57.5, -55.5, -52.3, -48.4, -43.8, -38.8, -34.2, -29.9, -25.8]
RAD_TO_STRAIN_DB = -155.6  # dB     offset between the rad and strain axes of the same plot   datasheet

# --- Optical loss items (S1) ---
SPLICE_SPACING = 10.0      # km     break points / joints along the route (Table A.12)        team design
SPLICE_LOSS = 0.10         # dB     per fusion splice, planning value                         ASSUMED
N_CONNECTORS = 2           # -      connectors at the interrogator / patch panel
CONNECTOR_LOSS = 0.50      # dB     per mated connector, planning value                       ASSUMED
RESERVE_AGEING = 1.0       # dB     ageing of fibre and splices over the design life          ASSUMED
RESERVE_REPAIR = 1.0       # dB     reserve for repairs (one ROV jumper with 2 wet-mate conn.) ASSUMED
WETMATE_LOSS = 0.50        # dB     ROV wet-mate optical connector, max at 1550 nm            Teledyne RS8 datasheet
RANGE_AMPLIFIED = 270.0    # km     single-ended reach with remote optical pumping (lab)      Yamamoto et al. 2026

# --- Reference route: Nord Stream 2 ---
ROUTE_KM = 1234.0          # km                                                               report Ch. 4
LUBMIN = (54.1405, 13.6600)          # deg  German landfall (approximate)                     map
RUPTURE_NS2A = (54.8767, 15.4100)    # deg  54 52.6 N, 015 24.6 E, line A, Sept 2022          Danish Energy Agency

# --- Pipe and cable (from Section 4.2) ---
OD_STEEL = 1.2192          # m
WALL = (0.0268, 0.0344, 0.0410)      # m   wall thickness range (Nord Stream: 34.4/30.9/26.8 mm; NS2 up to 41 mm)
E_STEEL = 207e9            # Pa
NU_STEEL = 0.30
T_CONCRETE = 0.060         # m
E_CONCRETE = 30e9          # Pa     concrete weight coating                                   ASSUMED
EA_CABLE = 48.1e6          # N      armour only                                               Appendix A.5
CABLE_WEIGHT_SUB = 1.8     # kg/m   in seawater                                               Hexatronic 2023
BLOCK_SPACING = 4.0        # m                                                                 Section 4.2
MU_BLOCK = (0.1, 0.3)      # -      friction cable/block slot, low-friction slot              ASSUMED

# --- Propagation (S6) ---
EXCESS_LOSS = {"low": 1.0, "base": 3.0, "high": 10.0}   # dB/km  seabed and scattering loss    ASSUMED
SOURCE_DEPTH_SURFACE = 5.0  # m     ship noise source depth (propeller)                        ASSUMED

# --- Ambient noise, Baltic (S4/S6): 63 Hz one-third octave band level, dB re 1 uPa ---
AMBIENT_BAND = {"typical": 100.0, "worst": 120.0}  # worst = DC-3.2; typical near shipping     ASSUMED
THIRD_OCT_BW_63 = 0.2316 * 63.0                    # Hz bandwidth of the 63 Hz band

# --- Detection (S6) ---
BAND = (10.0, 200.0)       # Hz     analysis band; below the 6.4 m gauge null (225 Hz) and Nyquist at 150 km
PD, PFA = 0.9, 1e-6        # -      probability of detection / false alarm per decision      ASSUMED
KAPPA = {"poor": 0.01, "breathing": 0.05, "base": 0.1, "buried-like": 1.0}

# Threats: (label, band source level [dB re 1 uPa @ 1 m] or SEL for impulses, integration time [s],
#           vertical offset to the cable [m], required lateral distance [m] (DC-1.2), basis)
THREATS = [
    ("Ship dragging anchor",   177.0, 10.0, H_REF - SOURCE_DEPTH_SURFACE, 1000.0,
     "lowest merchant-ship level, 177-188 dB (McKenna et al., 2012)"),
    ("Anchor drop, 5 t",       189.0,  1.0, 1.0,                          1000.0,
     "SEL, 71 kJ x 1e-3 radiated (A.6); range 179-199 dB"),
    ("Trawler towing gear",    170.0, 10.0, H_REF - SOURCE_DEPTH_SURFACE, 1700.0,
     "mean small vessel 170 dB, fishing vessels 145-195 dB (Parsons et al., 2021)"),
]
SHIP_SL_RANGE = (177.0, 188.0)

# --- Quiet targets (S7) ---
AUV_SL = 124.0             # dB     Autosub, 100-5000 Hz broadband                            Griffiths et al. 2001
HYDROPHONE_RANGE = {"Diver (open circuit)": 100.0, "Small ROV": 22.0}   # m  Korenbaum 2020; Cai 2011
AMBIENT_SPECTRAL_100_300 = 85.0   # dB re 1 uPa^2/Hz, 100-300 Hz, coastal ambient at those trials  ASSUMED

RESULTS = []


def result(check, quantity, value, unit="", note=""):
    RESULTS.append({"check": check, "quantity": quantity, "value": value, "unit": unit, "note": note})
    v = f"{value:,.3g}" if isinstance(value, (int, float)) else str(value)
    print(f"  {check:4s} {quantity:<62s} {v:>12s} {unit:<14s} {note}")


# =====================================================================================================
# 2. MODEL FUNCTIONS
# =====================================================================================================
_noise_interp = PchipInterpolator(EX_NOISE_Z_KM, EX_NOISE_DB_RAD)


def ex_noise_db_rad(z_km):
    return float(_noise_interp(np.clip(z_km, 0.0, RANGE_EX)))


def strain_noise_db(z_km):
    """dB re 1 strain/sqrt(Hz) at 6.4 m gauge."""
    return ex_noise_db_rad(z_km) + RAD_TO_STRAIN_DB


def bulk_modulus():
    return RHO_W * C_W ** 2


def nep_db(z_km, kappa):
    """Noise-equivalent pressure spectral level, dB re 1 uPa^2/Hz: the pressure whose fibre strain equals the
    interrogator noise. eps_fibre = kappa * p / (rho c^2)  ->  p_n = N_eps * rho c^2 / kappa."""
    return strain_noise_db(z_km) + 20 * math.log10(bulk_modulus()) + 120.0 - 20 * math.log10(kappa)


def ambient_spectral_db(level="typical"):
    return AMBIENT_BAND[level] - 10 * math.log10(THIRD_OCT_BW_63)


def total_noise_db(z_km, kappa, ambient="typical"):
    a, b = nep_db(z_km, kappa), ambient_spectral_db(ambient)
    return 10 * math.log10(10 ** (a / 10) + 10 ** (b / 10))


def thorp_db_per_km(f_hz):
    f = f_hz / 1000.0
    return 0.11 * f ** 2 / (1 + f ** 2) + 44 * f ** 2 / (4100 + f ** 2) + 2.75e-4 * f ** 2 + 0.003


def transmission_loss(R, H=H_REF, excess=EXCESS_LOSS["base"], f_hz=100.0):
    """Spherical spreading to the water depth, cylindrical beyond, plus water absorption over the whole path and
    excess (seabed and scattering) loss over the part of the path beyond one water depth, where the sound has
    started to interact with the seabed."""
    R = max(R, 1.0)
    spread = 20 * math.log10(R) if R <= H else 20 * math.log10(H) + 10 * math.log10(R / H)
    return spread + thorp_db_per_km(f_hz) * R / 1000.0 + excess * max(R - H, 0.0) / 1000.0


def detection_threshold():
    """Required 5 log10(d) for an energy detector: d = (Q^-1(Pfa) - Q^-1(Pd))^2 (Gaussian approximation)."""
    from scipy.stats import norm
    d = (norm.isf(PFA) - norm.isf(PD)) ** 2
    return 5 * math.log10(d), d


DT_DB, D_INDEX = detection_threshold()
B_HZ = BAND[1] - BAND[0]


def spectral_source_level(sl_band, T, impulse=False):
    """Band level -> mean spectral level. For an impulse the SEL is spread over T seconds and the band."""
    if impulse:
        return sl_band - 10 * math.log10(T) - 10 * math.log10(B_HZ)
    return sl_band - 10 * math.log10(B_HZ)


def signal_excess(r, z_km, kappa, sl_band, T, dz, impulse=False, excess=EXCESS_LOSS["base"],
                  ambient="typical", H=H_REF):
    R = math.hypot(r, dz)
    return (spectral_source_level(sl_band, T, impulse) - transmission_loss(R, H, excess)
            - total_noise_db(z_km, kappa, ambient) + 5 * math.log10(B_HZ * T) - DT_DB)


def detection_distance(z_km, kappa, sl_band, T, dz, impulse=False, **kw):
    """Largest lateral distance r with SE >= 0 (0 if the threat is not detected even at closest approach)."""
    f = lambda r: signal_excess(r, z_km, kappa, sl_band, T, dz, impulse, **kw)
    if f(0.0) < 0:
        return 0.0
    hi = 60e3
    if f(hi) >= 0:
        return hi
    return brentq(f, 0.0, hi)


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


# =====================================================================================================
# 3. CHECKS
# =====================================================================================================
def s1_optical_reach():
    fixed = N_CONNECTORS * CONNECTOR_LOSS + RESERVE_AGEING + RESERVE_REPAIR
    per_km = FIBRE_LOSS + SPLICE_LOSS / SPLICE_SPACING
    L = (BUDGET_DB - fixed) / per_km
    result("S1", "Fixed losses and reserves (connectors, ageing, repair)", fixed, "dB")
    result("S1", "Distributed loss incl. splices every 10 km", per_km, "dB/km")
    result("S1", "Optical reach per channel", L, "km", "datasheet: 150 km")
    result("S1", "Reach lost per extra dB of loss", 1 / per_km, "km/dB")
    return L


def s2_route_coverage(L_ex):
    cov_ex = min(ROUTE_KM, 2 * L_ex)
    cov_amp = min(ROUTE_KM, 2 * RANGE_AMPLIFIED)
    result("S2", "NS2 monitored with hosts at both landfalls (EX)", cov_ex, "km", f"{100 * cov_ex / ROUTE_KM:.0f} %")
    result("S2", "NS2 monitored with remote amplification (270 km/end)", cov_amp, "km",
           f"{100 * cov_amp / ROUTE_KM:.0f} %")
    result("S2", "Unmonitored middle section (EX)", ROUTE_KM - cov_ex, "km")
    n_hosts = math.ceil((ROUTE_KM - cov_ex) / (2 * L_ex))
    result("S2", "Intermediate hosts for full coverage (2 EX per host)", n_hosts, "-")
    d_rupt = haversine_km(LUBMIN, RUPTURE_NS2A)
    result("S2", "Straight-line distance Lubmin - 2022 rupture of line A", d_rupt, "km")
    return cov_ex, cov_amp, n_hosts, d_rupt


def s3_bandwidth():
    out = {}
    for L in (25, 50, 65, 100, 150):
        fs = C_LIGHT / (2 * N_EFF * L * 1e3)
        out[L] = fs
        result("S3", f"Max. sample rate / Nyquist on a {L} km fibre", fs, "Hz", f"Nyquist {fs / 2:.0f} Hz")
    result("S3", "First gauge-length null on axis (c / L_g)", C_W / GAUGE, "Hz")
    return out


def s4_noise():
    for z in (0, 50, 75, 100, 125, 150):
        result("S4", f"EX self-noise at {z} km (rel. to front of fibre)", ex_noise_db_rad(z) - ex_noise_db_rad(0), "dB",
               f"{ex_noise_db_rad(z):.1f} dB re rad/rtHz; {strain_noise_db(z):.1f} dB re strain/rtHz")
    eps0 = 10 ** (strain_noise_db(0) / 20)
    result("S4", "Strain noise at the front of the fibre", eps0 * 1e12, "pstrain/rtHz")
    for k in (0.01, 0.1, 1.0):
        result("S4", f"Noise-equivalent pressure at 0 km, kappa = {k}", nep_db(0, k), "dB re uPa2/Hz")
    result("S4", "Ambient spectral level, typical / worst (63 Hz band)", ambient_spectral_db("typical"), "dB re uPa2/Hz",
           f"worst {ambient_spectral_db('worst'):.1f}")
    result("S4", "Water strain per pascal 1/(rho c^2)", 1e9 / bulk_modulus(), "nstrain/Pa")


def s5_coupling():
    w = CABLE_WEIGHT_SUB * 9.81 * BLOCK_SPACING
    f_lo, f_hi = MU_BLOCK[0] * w, MU_BLOCK[1] * w
    result("S5", "Friction available per block span (mu 0.1-0.3)", f_lo, "N", f"to {f_hi:.0f} N")
    result("S5", "Force to strain the cable by 1 nanostrain (EA x 1e-9)", EA_CABLE * 1e-9, "N")
    result("S5", "Strain above which the cable slips in the blocks", f_lo / EA_CABLE * 1e6, "microstrain",
           f"to {f_hi / EA_CABLE * 1e6:.2f}")
    R = OD_STEEL / 2
    kap = []
    for t in WALL:
        hoop_stiff = E_STEEL * t + E_CONCRETE * T_CONCRETE
        eps_axial_per_pa = NU_STEEL * R / hoop_stiff          # Poisson strain of a free pipe under pressure
        kap.append(eps_axial_per_pa * bulk_modulus())
    result("S5", "Pipe-breathing coupling factor (wall 26.8 / 34.4 / 41 mm)", kap[1], "-",
           f"{min(kap):.3f}-{max(kap):.3f}")
    k_cable = (1 - 2 * 0.3) / 200e9 * bulk_modulus()
    result("S5", "Cable alone, hydrostatic pressure (steel-dominated)", k_cable, "-")
    return kap


def s6_detection():
    rows = []
    zs = (0, 25, 50, 75, 100, 125, 150)
    for label, sl, T, dz, req, basis in THREATS:
        imp = "drop" in label
        for kname in ("base", "buried-like"):
            k = KAPPA[kname]
            rr = [detection_distance(z, k, sl, T, dz, imp) for z in zs]
            # monitored length: largest z where the requirement is still met
            zgrid = np.linspace(0, 150, 601)
            ok = [z for z in zgrid if detection_distance(z, k, sl, T, dz, imp) >= req]
            zmax = max(ok) if ok else 0.0
            rows.append((label, kname, rr, zmax))
            result("S6", f"{label}, kappa {k}: r at 0/50/100/150 km", round(rr[0]), "m",
                   f"{rr[2]:.0f} / {rr[4]:.0f} / {rr[6]:.0f} m; meets {req:.0f} m up to {zmax:.0f} km")
   
    for label, sl, T, dz, req, basis in THREATS:
        imp = "drop" in label
        for kname in ("base", "buried-like"):
            result("S6", f"At 140 km (rupture site): {label}, kappa {KAPPA[kname]}",
                   detection_distance(140.0, KAPPA[kname], sl, T, dz, imp), "m")
    # louder ship
    for kname in ("base", "buried-like"):
        k = KAPPA[kname]
        zgrid = np.linspace(0, 150, 601)
        ok = [z for z in zgrid if detection_distance(z, k, SHIP_SL_RANGE[1], 10.0, H_REF - 5, False) >= 1000]
        result("S6", f"Loud ship 188 dB, kappa {k}: meets 1000 m up to", max(ok) if ok else 0, "km")
    # vessel overhead at the deepest point of the route (210 m) and in deep water (2000 m)
    for H in (210.0, 2000.0):
        for kname in ("base", "buried-like"):
            k = KAPPA[kname]
            zgrid = np.linspace(0, 150, 601)
            ok = [z for z in zgrid if signal_excess(0, z, k, 177.0, 10.0, H - 5, H=H) >= 0]
            result("S6", f"Ship 177 dB overhead in {H:.0f} m water, kappa {k}: detected up to", max(ok) if ok else 0, "km")
    # sensitivity at the front of the fibre, ship dragging anchor
    base = detection_distance(0, KAPPA["base"], 177.0, 10.0, 75.0)
    result("S6", "Sensitivity (z=0, ship 177 dB): base", base, "m")
    for name, kw in (("excess loss 1 dB/km", dict(excess=1.0)), ("excess loss 10 dB/km", dict(excess=10.0)),
                     ("worst ambient 120 dB", dict(ambient="worst"))):
        result("S6", f"   {name}", detection_distance(0, KAPPA["base"], 177.0, 10.0, 75.0, **kw), "m")
    for kname in ("poor", "breathing", "buried-like"):
        result("S6", f"   kappa {KAPPA[kname]}", detection_distance(0, KAPPA[kname], 177.0, 10.0, 75.0), "m")
    result("S6", "   ship 188 dB", detection_distance(0, KAPPA["base"], 188.0, 10.0, 75.0), "m")
    result("S6", "Detection threshold 5 log10(d)", DT_DB, "dB", f"d = {D_INDEX:.1f} (Pd {PD}, Pfa {PFA:g})")
    result("S6", "Integration gain 5 log10(BT), T = 10 s / 1 s", 5 * math.log10(B_HZ * 10), "dB",
           f"{5 * math.log10(B_HZ * 1):.1f} dB")
    return rows


def required_band_level(r, z_km, kappa, T=10.0, dz=10.0, ambient="typical"):
    R = math.hypot(r, dz)
    return (transmission_loss(R) + total_noise_db(z_km, kappa, ambient) - 5 * math.log10(B_HZ * T) + DT_DB
            + 10 * math.log10(B_HZ))


def s7_quiet():
    for kname in ("base", "buried-like"):
        k = KAPPA[kname]
        result("S7", f"Required band level at 300 m (ROV/AUV), 0 km, kappa {k}", required_band_level(300, 0, k), "dB re uPa")
        result("S7", f"Required band level at 75 m (diver), 0 km, kappa {k}", required_band_level(75, 0, k, dz=5), "dB re uPa")
    auv_inband = AUV_SL + 10 * math.log10((BAND[1] - 100.0) / (5000.0 - 100.0))
    result("S7", "Autosub level in the 100-200 Hz part of the band (flat spectrum)", auv_inband, "dB re uPa")
    for name, rng in HYDROPHONE_RANGE.items():
        for kname in ("base", "buried-like"):
            delta = nep_db(0, KAPPA[kname]) - AMBIENT_SPECTRAL_100_300
            result("S7", f"{name}: hydrophone {rng:.0f} m -> DAS, kappa {KAPPA[kname]}", rng * 10 ** (-delta / 20), "m",
                   f"DAS noise {delta:.0f} dB above ambient")


def s8_other(L_ex):
    t_light = 2 * L_ex * 1e3 * N_EFF / C_LIGHT
    result("S8", "Round-trip light time over the span", t_light * 1e3, "ms")
    budget = [("Light travel, round trip", t_light), ("Acquisition and phase demodulation", 1.0),
              ("Detection window (impulse 1 s, vessel 10 s)", 10.0), ("Classification (built-in GPU)", 1.0),
              ("Alert to control room (MODBUS/OPC UA)", 2.0)]
    total = sum(t for _, t in budget)
    result("S8", "Machine latency event -> alert (sum of budget)", total, "s", "requirement 600 s (DC-2.1)")
    n_ch = int(L_ex * 1e3 / 5.0)
    rate = n_ch * 500 * 4
    result("S8", "Raw data rate, 5 m channels, 500 Hz, float32", rate / 1e6, "MB/s", f"{rate * 86400 / 1e12:.1f} TB/day")
    result("S8", "Days of raw data on 2 x 2 TB / 2 x 8 TB storage", 4e12 / (rate * 86400), "days",
           f"{16e12 / (rate * 86400):.1f} days")
    per_rov_repair = 2 * WETMATE_LOSS + 2 * SPLICE_LOSS
    per_vessel_repair = 2 * SPLICE_LOSS
    result("S8", "Loss of one ROV jumper repair (2 wet-mate + 2 splices)", per_rov_repair, "dB")
    result("S8", "Loss of one vessel cut-and-joint repair (2 joints)", per_vessel_repair, "dB")
    result("S8", "Reach lost per ROV jumper repair", per_rov_repair / (FIBRE_LOSS + SPLICE_LOSS / SPLICE_SPACING), "km")
    return budget, total


# =====================================================================================================
# 4. FIGURES
# =====================================================================================================
def make_figures(L_ex, cov, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ink, accent, grey, light, blue = "#1d1d1f", "#e0701b", "#62666c", "#c9ccd1", "#2f6db5"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    # ---------------- Figure 1: optical reach and detection reach on the Nord Stream 2 route ----------------
    fig, ax = plt.subplots(figsize=(6.0, 2.35))
    d_rupt = haversine_km(LUBMIN, RUPTURE_NS2A)
    zgrid = np.linspace(0, 150, 601)

    def reach_for(kappa, sl=177.0, T=10.0, dz=H_REF - 5, req=1000.0):
        ok = [z for z in zgrid if detection_distance(z, kappa, sl, T, dz) >= req]
        return max(ok) if ok else 0.0

    rows_f1 = [
        ("Optical reach with remote\namplification (lab result)", RANGE_AMPLIFIED, "amp"),
        ("Optical reach, ONYX peta EX", L_ex, "ex"),
        (r"Ship at 1 km detected, $\kappa$ = 1", reach_for(1.0), "det"),
        (r"Ship at 1 km detected, $\kappa$ = 0.1", reach_for(0.1), "det"),
    ]
    for i, (lab, reach, kind) in enumerate(rows_f1[::-1]):
        y = i
        ax.barh(y, ROUTE_KM, color="#eceef1", height=0.56, edgecolor="none")
        for side in ("left", "right"):
            if kind == "ex":
                n = 60  # noise penalty shading along the span (darker = noisier)
                for j in range(n):
                    a, b = j * reach / n, (j + 1) * reach / n
                    frac = min((ex_noise_db_rad((a + b) / 2) - ex_noise_db_rad(0)) / 36.0, 1.0)
                    col = (0.18 + 0.70 * frac, 0.43 - 0.13 * frac, 0.71 - 0.55 * frac)
                    left = a if side == "left" else ROUTE_KM - b
                    ax.barh(y, b - a, left=left, color=col, height=0.56, edgecolor="none")
            elif kind == "amp":
                left = 0 if side == "left" else ROUTE_KM - reach
                ax.barh(y, reach, left=left, color="white", height=0.56, edgecolor=grey, hatch="////", lw=0.6)
            else:
                left = 0 if side == "left" else ROUTE_KM - reach
                ax.barh(y, reach, left=left, color=ink, height=0.56, edgecolor="none")
                if side == "left":
                    if reach > 60:
                        ax.text(reach / 2, y, f"{reach:.0f} km", va="center", ha="center", fontsize=6.5, color="white")
                        ax.text(ROUTE_KM - reach / 2, y, f"{reach:.0f} km", va="center", ha="center", fontsize=6.5,
                                color="white")
                    else:
                        ax.text(reach + 8, y, f"{reach:.0f} km", va="center", fontsize=6.5, color=ink)
                        ax.text(ROUTE_KM - reach - 8, y, f"{reach:.0f} km", va="center", ha="right", fontsize=6.5,
                                color=ink)
    ax.set_yticks(range(len(rows_f1)))
    ax.set_yticklabels([r[0] for r in rows_f1[::-1]], fontsize=7)
    ax.axvline(d_rupt, color=accent, lw=1.2, ls="--")
    ax.text(d_rupt + 10, len(rows_f1) - 0.35, f"2022 rupture of line A ({d_rupt:.0f} km from Lubmin in a straight line)",
            color=accent, fontsize=6.5, va="center", ha="left")
    ax.text(ROUTE_KM / 2, 2, f"{ROUTE_KM - 2 * L_ex:.0f} km without a host", ha="center", va="center", fontsize=6.5,
            color=grey)
    ax.set_ylim(-0.5, len(rows_f1) - 0.1)
    ax.set_xlim(0, ROUTE_KM)
    ax.set_xlabel("Distance along the route from the German landfall at Lubmin (km)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    leg = [Patch(color=(0.18, 0.43, 0.71), label="EX span, low noise"),
           Patch(color=(0.88, 0.30, 0.16), label="EX span, noise up to +35 dB"),
           Patch(facecolor="white", edgecolor=grey, hatch="////", label="amplified, noise not published"),
           Patch(color=ink, label=r"requirement met")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.42, -0.3), ncol=4, frameon=False, fontsize=6.5,
              handlelength=1.6, columnspacing=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"sens_fig1_coverage.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 2: noise-equivalent pressure along the fibre, and band against span ----------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6), gridspec_kw={"wspace": 0.42})
    z = np.linspace(0, 150, 301)
    for k, ls, lab in ((0.01, ":", r"$\kappa$ = 0.01, cable alone"), (0.1, "-", r"$\kappa$ = 0.1, base (on the pipe)"),
                       (1.0, "--", r"$\kappa$ = 1, like a buried cable")):
        a1.plot(z, [nep_db(v, k) for v in z], ls, color=ink, lw=1.2, label=lab)
    a1.legend(loc="upper left", fontsize=6.3, frameon=False, handlelength=2.2, borderaxespad=0.2)
    a1.axhspan(ambient_spectral_db("typical") - 10, ambient_spectral_db("worst"), color=light, alpha=0.6, lw=0)
    a1.text(148, ambient_spectral_db("typical") - 8.5, "Baltic ambient\n(63 Hz band, 90-120 dB)", fontsize=6.5,
            color=grey, va="bottom", ha="right")
    a1.set_xlabel("Position along the fibre (km)")
    a1.set_ylabel("Noise-equivalent pressure\n(dB re 1 µPa²/Hz)")
    a1.set_xlim(0, 150)
    a1.set_ylim(70, 180)
    a1.set_title("(a) Interrogator noise as acoustic pressure", fontsize=8, loc="left")

    L = np.linspace(5, 200, 400)
    nyq = C_LIGHT / (2 * N_EFF * L * 1e3) / 2
    a2.plot(L, nyq, color=ink, lw=1.3)
    bands = [("Ships, anchors", 10, 200, blue), ("ROV motors", 400, 500, accent),
             ("Diver breathing", 70, 1000, "#7a5ea8"), ("AUV (Autosub)", 100, 5000, grey)]
    ypos = {"Ships, anchors": 25, "ROV motors": 450, "Diver breathing": 150, "AUV (Autosub)": 2600}
    xlab = {"Ships, anchors": 62, "ROV motors": 157, "Diver breathing": 62, "AUV (Autosub)": 62}
    for j, (lab, f0, f1, col) in enumerate(bands):
        a2.fill_between([5, 200], f0, f1, color=col, alpha=0.12, lw=0)
        a2.text(xlab[lab], ypos[lab], lab, fontsize=6.3, color=col, ha="left", va="center")
    a2.axvline(150, color=grey, lw=0.8, ls="--")
    a2.text(148, 2.2e4, "EX range", fontsize=6.3, color=grey, ha="right", va="top")
    a2.set_yscale("log")
    a2.set_xlim(5, 200)
    a2.set_ylim(5, 3e4)
    a2.set_xlabel("Fibre length read by one channel (km)")
    a2.set_ylabel("Highest recordable frequency (Hz)")
    a2.set_title("(b) Recordable band against span", fontsize=8, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"sens_fig2_noise_band.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 3: detection distance along the fibre ----------------
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    z = np.linspace(0, 150, 301)
    cols = {"Ship dragging anchor": ink, "Anchor drop, 5 t": accent, "Trawler towing gear": blue}
    for label, sl, T, dz, req, basis in THREATS:
        imp = "drop" in label
        for kname, ls in (("base", "-"), ("buried-like", "--")):
            r = np.array([detection_distance(v, KAPPA[kname], sl, T, dz, imp) for v in z])
            r = np.where(r > 0, r, np.nan)
            ax.plot(z, r, ls, color=cols[label], lw=1.3)
    for req, lab in ((1000, "anchor, 1000 m"), (1700, "trawl gear, 1700 m")):
        ax.axhline(req, color=grey, lw=0.8, ls=":")
        ax.text(151, req, lab, fontsize=6.5, color=grey, va="center", ha="left")
    ax.set_yscale("log")
    ax.set_ylim(30, 2e4)
    ax.set_xlim(0, 150)
    ax.set_xlabel("Position of the threat along the fibre, from the interrogator (km)")
    ax.set_ylabel("Detection distance from the pipeline (m)")
    h = [Line2D([], [], color=cols[l], lw=1.3, label=l) for l in cols]
    h += [Line2D([], [], color=grey, lw=1.2, ls="-", label=r"$\kappa$ = 0.1 (base)"),
          Line2D([], [], color=grey, lw=1.2, ls="--", label=r"$\kappa$ = 1 (like a buried cable)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5, frameon=False, ncol=3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"sens_fig3_detection.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv():
    with open(os.path.join(HERE, "results_sensing_performance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "quantity", "value", "unit", "note"])
        w.writeheader()
        w.writerows(RESULTS)


if __name__ == "__main__":
    print("PIGGY sensing and detection performance - Nord Stream 2 reference case\n")
    L_ex = s1_optical_reach()
    cov = s2_route_coverage(L_ex)
    s3_bandwidth()
    s4_noise()
    s5_coupling()
    rows = s6_detection()
    s7_quiet()
    s8_other(L_ex)
    make_figures(L_ex, cov, rows)
    write_csv()
    print("\nFigures and CSV written to", HERE)
