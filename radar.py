#!/usr/bin/env python3
"""
Ethiopian Aerospace Radar Tracking System
"""
import math
import random
import time
import os
import sys

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
PI    = math.pi
C     = 300000000      # speed of light in m/s
F0    = 9400000000     # radar frequency: 9.4 GHz
LMBDA = C / F0         # wavelength = c / f

# ─────────────────────────────────────────
# TARGET CLASS
# represents one aircraft in the sky
# ─────────────────────────────────────────
class Aircraft:
    def __init__(self, id, name, x, y, altitude, vx, vy, rcs):
        self.id       = id        # e.g. "ET001"
        self.name     = name      # human readable name
        self.x        = x         # east position in meters
        self.y        = y         # north position in meters
        self.altitude = altitude  # height in meters
        self.vx       = vx        # east velocity m/s
        self.vy       = vy        # north velocity m/s
        self.rcs      = rcs       # radar cross section m² (how "visible" it is)

    def move(self, dt):
        """Update position: x = x + v * t (basic kinematics)"""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def range(self):
        """3D distance from radar (at origin): R = sqrt(x² + y² + z²)"""
        return math.sqrt(self.x**2 + self.y**2 + self.altitude**2)

    def azimuth(self):
        """
        Bearing angle from North, clockwise (0° to 360°)
        Uses atan2 to handle all four quadrants correctly
        atan2(east, north) gives angle from North
        """
        angle = math.degrees(math.atan2(self.x, self.y))
        return angle % 360   # keep it 0–360

    def elevation(self):
        """
        Angle above the horizon: phi = atan(altitude / ground_distance)
        ground distance = distance on the flat ground (no altitude)
        """
        ground = math.sqrt(self.x**2 + self.y**2)
        if ground < 1:
            return 90.0
        return math.degrees(math.atan(self.altitude / ground))

    def doppler(self):
        """
        Doppler frequency shift: fd = (2 * vr * f0) / c

        vr = radial velocity = how fast the aircraft moves toward/away from radar
        Formula: vr = (vx*x + vy*y) / R (dot product divided by range)

        Positive fd → aircraft approaching (frequency increases)
        Negative fd → aircraft moving away (frequency decreases)
        """
        R = self.range()
        if R < 1:
            return 0.0
        # Radial velocity: project velocity onto the line from radar to target
        vr = (self.vx * self.x + self.vy * self.y) / R
        fd = (2 * vr * F0) / C
        return fd

    def radial_velocity(self):
        """Radial velocity in m/s (the real-world meaning of Doppler)"""
        R = self.range()
        if R < 1:
            return 0.0
        return (self.vx * self.x + self.vy * self.y) / R

    def received_power_dbm(self):
        """
        Radar Range Equation — how strong is the signal we receive?

              Pt * G² * λ² * σ
        Pr =  ─────────────────
               (4π)³ * R⁴

        Pt = transmit power (W)
        G  = antenna gain
        λ  = wavelength (m)
        σ  = radar cross section (m²)
        R  = range (m)

        We convert watts to dBm: dBm = 10 * log10(watts) + 30
        """
        Pt = 50000          # 50 kW transmit power
        G  = 1000000        # antenna gain (large dish radar)
        R  = self.range()
        if R < 1:
            R = 1

        numerator   = Pt * G * G * LMBDA * LMBDA * self.rcs
        denominator = ((4 * PI) ** 3) * (R ** 4)
        pr_watts    = numerator / denominator

        if pr_watts <= 0:
            return -200.0
        pr_dbm = 10 * math.log10(pr_watts) + 30
        return pr_dbm

    def snr(self):
        """Signal to noise ratio in dB (noise floor ≈ -114 dBm for this radar)"""
        return self.received_power_dbm() - (-114.0)

    def is_detected(self):
        """Target is detected only if SNR > 10 dB and within 400 km"""
        return self.snr() > 10 and self.range() < 400000 and self.altitude > 0

    def speed(self):
        """Total speed in m/s: v = sqrt(vx² + vy²)"""
        return math.sqrt(self.vx**2 + self.vy**2)


# ─────────────────────────────────────────
# ROTATION MATRIX (2D)
# ─────────────────────────────────────────
def rotation_matrix(theta_deg):
    """
    Rotate a 2D point by angle theta (degrees)

    R(θ) = | cos θ  -sin θ |
           | sin θ   cos θ |

    Used to convert between radar frame and map frame.
    """
    t  = math.radians(theta_deg)
    r00 =  math.cos(t)
    r01 = -math.sin(t)
    r10 =  math.sin(t)
    r11 =  math.cos(t)
    return [[r00, r01], [r10, r11]]

def rotate_point(matrix, x, y):
    """Apply 2x2 rotation matrix to point (x, y)"""
    x2 = matrix[0][0] * x + matrix[0][1] * y
    y2 = matrix[1][0] * x + matrix[1][1] * y
    return x2, y2


# ─────────────────────────────────────────
# KALMAN FILTER (1D — for range smoothing)
# ─────────────────────────────────────────
class KalmanFilter:
    """
    Smooths noisy radar range measurements.

    State: [range, range_rate]
    Every scan:
      1. Predict: x = F * x
      2. Update:  x = x + K * (measurement - prediction)

    K = Kalman Gain = how much to trust the new measurement
    """
    def __init__(self, initial_range):
        self.r  = initial_range   # estimated range
        self.vr = 0.0             # estimated range rate
        self.P  = 500.0           # error uncertainty
        self.Q  = 10.0            # process noise
        self.R  = 200.0           # measurement noise
        self.dt = 10.0            # time step (seconds)

    def predict(self):
        self.r  += self.vr * self.dt    # x = x + v*t
        self.P  += self.Q               # uncertainty grows

    def update(self, measured_range):
        K       = self.P / (self.P + self.R)          # Kalman gain
        self.r  = self.r + K * (measured_range - self.r)  # correction
        self.P  = (1 - K) * self.P                    # reduce uncertainty
        return self.r


# ─────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def bar(value, max_val, width=20, char='█', empty='░'):
    filled = int((value / max_val) * width) if max_val > 0 else 0
    filled = max(0, min(width, filled))
    return char * filled + empty * (width - filled)

def direction_arrow(vr):
    """Show if aircraft is approaching or receding"""
    if vr > 2:
        return "◄── APPROACHING"
    elif vr < -2:
        return "──► RECEDING   "
    else:
        return "  ●  CROSSING  "

def status_icon(aircraft):
    if not aircraft.is_detected():
        return "[!] NOT DETECTED"
    snr = aircraft.snr()
    if snr > 30:
        return "[*] STRONG LOCK "
    elif snr > 20:
        return "[+] TRACKING    "
    else:
        return "[~] WEAK SIGNAL "


# ─────────────────────────────────────────
# MINI ASCII RADAR SCOPE
# ─────────────────────────────────────────
def draw_radar_scope(aircraft_list, width=41, height=21):
    """
    Draw a simple ASCII top-down radar view.
    Center = radar station in Addis Ababa.
    Scale: each character = 10 km
    """
    grid = [['.' for _ in range(width)] for _ in range(height)]
    cx, cy = width // 2, height // 2

    # Draw range rings (circles approximated with characters)
    for ring_km in [100, 200, 300]:
        ring_r_chars = ring_km // 10
        for angle_deg in range(0, 360, 3):
            angle_rad = math.radians(angle_deg)
            px = int(cx + ring_r_chars * math.sin(angle_rad))
            py = int(cy - ring_r_chars * math.cos(angle_rad))
            if 0 <= px < width and 0 <= py < height:
                grid[py][px] = '+'

    # Draw axes
    for i in range(width):
        if grid[cy][i] == '.':
            grid[cy][i] = '-'
    for i in range(height):
        if grid[i][cx] == '.':
            grid[i][cx] = '|'
    grid[cy][cx] = 'R'   # Radar center

    # Draw compass labels
    symbols = {
        'N': (cx, 0),
        'S': (cx, height - 1),
        'E': (width - 1, cy),
        'W': (0, cy)
    }
    for label, (lx, ly) in symbols.items():
        grid[ly][lx] = label

    # Plot aircraft
    labels = []
    for ac in aircraft_list:
        if ac.is_detected():
            # Scale: 10 km per character
            px = int(cx + (ac.x / 1000) / 10)
            py = int(cy - (ac.y / 1000) / 10)
            if 0 <= px < width and 0 <= py < height:
                grid[py][px] = ac.id[2]  # use last digit as marker
                labels.append(f"  {ac.id[2]} = {ac.id} ({ac.name[:20]})")

    # Render
    lines = []
    lines.append("  +" + "-" * width + "+")
    for row in grid:
        lines.append("  |" + "".join(row) + "|")
    lines.append("  +" + "-" * width + "+")
    lines.append("     Scale: 1 char = 10 km   +: range rings (100/200/300 km)")
    for l in labels:
        lines.append(l)
    return "\n".join(lines)


# ─────────────────────────────────────────
# MATH BREAKDOWN DISPLAY
# ─────────────────────────────────────────
def show_math(ac):
    """Show the actual equations and their computed values for one aircraft."""
    R  = ac.range()
    az = ac.azimuth()
    el = ac.elevation()
    vr = ac.radial_velocity()
    fd = ac.doppler()
    pr = ac.received_power_dbm()
    sn = ac.snr()

    Pt = 50000
    G  = 1000000
    sig = ac.rcs

    print(f"\n  ╔══ EQUATION BREAKDOWN: {ac.name} ══╗")
    print(f"\n  1. RANGE  (Euclidean Distance)")
    print(f"     R = sqrt(x² + y² + z²)")
    print(f"     R = sqrt({ac.x/1000:.1f}² + {ac.y/1000:.1f}² + {ac.altitude/1000:.1f}²)  [km]")
    print(f"     R = {R/1000:.2f} km")

    print(f"\n  2. AZIMUTH  (Bearing from North)")
    print(f"     θ = atan2(x_east, y_north)")
    print(f"     θ = atan2({ac.x/1000:.1f}, {ac.y/1000:.1f})")
    print(f"     θ = {az:.2f}°")

    print(f"\n  3. ELEVATION  (Angle above horizon)")
    print(f"     φ = atan(altitude / ground_range)")
    gnd = math.sqrt(ac.x**2 + ac.y**2) / 1000
    print(f"     φ = atan({ac.altitude/1000:.2f} / {gnd:.2f})")
    print(f"     φ = {el:.2f}°")

    print(f"\n  4. DOPPLER EFFECT  (Frequency shift)")
    print(f"     vr = (vx·x + vy·y) / R")
    print(f"     vr = ({ac.vx}·{ac.x/1000:.0f} + {ac.vy}·{ac.y/1000:.0f}) / {R/1000:.0f}")
    print(f"     vr = {vr:.2f} m/s   ({'approaching' if vr > 0 else 'receding'})")
    print(f"     fd = (2 × vr × f0) / c")
    print(f"     fd = (2 × {vr:.1f} × {F0:.2e}) / {C:.2e}")
    print(f"     fd = {fd:.0f} Hz  ({fd/1000:.2f} kHz)")

    print(f"\n  5. RADAR RANGE EQUATION  (Signal power)")
    print(f"     Pr = (Pt × G² × λ² × σ) / ((4π)³ × R⁴)")
    print(f"     Pt={Pt:.0e}W  G={G:.0e}  λ={LMBDA*100:.2f}cm  σ={sig}m²  R={R/1000:.1f}km")
    print(f"     Pr = {pr:.1f} dBm      SNR = {sn:.1f} dB")

    print(f"\n  6. ROTATION MATRIX  (Coordinate transform)")
    M = rotation_matrix(az)
    rx, ry = rotate_point(M, ac.x/1000, ac.y/1000)
    print(f"     R(θ) = [[cos{az:.0f}°, -sin{az:.0f}°], [sin{az:.0f}°, cos{az:.0f}°]]")
    print(f"     R(θ) = [[{M[0][0]:.3f}, {M[0][1]:.3f}], [{M[1][0]:.3f}, {M[1][1]:.3f}]]")
    print(f"     Rotated position: ({rx:.1f}, {ry:.1f}) km in radar frame")


# ─────────────────────────────────────────
# CREATE SCENARIO
# ─────────────────────────────────────────
def create_targets():
    return [
        Aircraft("ET001", "Ethiopian Airlines ET-301",
                 x=80000,   y=120000, altitude=11000, vx=-180, vy=-140, rcs=40),
        Aircraft("ET002", "Military Patrol Aircraft",
                 x=-60000,  y=90000,  altitude=5000,  vx=80,   vy=-60,  rcs=5),
        Aircraft("ET003", "Kenya Airways KQ-204",
                 x=150000,  y=-50000, altitude=9500,  vx=-200, vy=100,  rcs=35),
        Aircraft("ET004", "Unknown Drone",
                 x=15000,   y=25000,  altitude=800,   vx=20,   vy=30,   rcs=0.05),
        Aircraft("ET005", "Cargo Flight",
                 x=-200000, y=80000,  altitude=8000,  vx=220,  vy=-40,  rcs=50),
    ]


# ─────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────
def main():
    targets = create_targets()
    kalman  = {t.id: KalmanFilter(t.range()) for t in targets}

    scan_number = 0
    selected_id = None

    while True:
        clear()

        # ── Move all targets forward 10 seconds ──
        scan_number += 1
        for t in targets:
            t.move(dt=10)

        # ── Header ──────────────────────────────
        print("=" * 65)
        print("   ETHIOPIAN AEROSPACE RADAR TRACKING SYSTEM")
        print("   Station: Addis Ababa  |  Frequency: 9.4 GHz (X-band)")
        print(f"   Scan #{scan_number}  |  Time step: 10 seconds per scan")
        print("=" * 65)

        # ── Radar scope ─────────────────────────
        print()
        print(draw_radar_scope(targets))
        print()

        # ── Target table ────────────────────────
        print("-" * 65)
        print(f"  {'ID':<7} {'Aircraft':<28} {'Range':>8} {'Az':>6} {'El':>5} {'SNR':>7}")
        print("-" * 65)

        for t in targets:
            # Kalman filter
            kalman[t.id].predict()
            smooth_r = kalman[t.id].update(t.range())

            if t.is_detected():
                icon = "[+]"
                snr_str = f"{t.snr():+.1f}dB"
            else:
                icon = "[!]"
                snr_str = " LOW  "

            print(f"  {icon} {t.id:<5} {t.name:<28} "
                  f"{t.range()/1000:>6.1f}km "
                  f"{t.azimuth():>5.1f}° "
                  f"{t.elevation():>4.1f}° "
                  f"{snr_str:>7}")

            if t.is_detected():
                vr = t.radial_velocity()
                fd = t.doppler()
                print(f"            {direction_arrow(vr)}  "
                      f"fd={fd/1000:+.2f}kHz  v={t.speed():.0f}m/s  "
                      f"alt={t.altitude/1000:.1f}km")
                print(f"            SNR bar: [{bar(t.snr(), 50)}] {t.snr():.1f}dB")
            print()

        # ── Menu ────────────────────────────────
        print("-" * 65)
        print("  Commands:")
        print("  [1-5] Show equation breakdown for target")
        print("  [Enter] Next scan")
        print("  [q] Quit")
        print("-" * 65)

        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Radar shutdown. Goodbye.")
            break

        if cmd == 'q':
            print("\n  Radar shutdown. Goodbye.")
            break
        elif cmd in ['1', '2', '3', '4', '5']:
            idx = int(cmd) - 1
            clear()
            show_math(targets[idx])
            input("\n  Press Enter to continue...")
        # Enter or anything else → next scan


if __name__ == "__main__":
    main()
