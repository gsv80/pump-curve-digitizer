# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "numpy",
#   "matplotlib",
# ]
# ///

"""
Pump Curve Interpolator

Loads a pump catalog (JSON) produced by the PumpCurveDigitizer.
Given a pump model, speed (rpm), and a target duty point (Q, H),
the script finds the pump, interpolates a new impeller diameter (or number of stages)
using physical scaling (affinity laws) and displays the result with interactive axis adjustment.

Usage:
    uv run pump_interpolator.py [--catalog CATALOG] [--model MODEL] [--rpm RPM]
                                [--Q FLOW] [--H HEAD]
"""

import json
import numpy as np
import argparse
import sys
import math
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better window management
import matplotlib.pyplot as plt

def load_catalog(filename):
    """Load JSON catalog from file."""
    with open(filename, 'r') as f:
        return json.load(f)

def find_pump(catalog, model, rpm):
    """
    Return the pump entry matching model and rpm, case‑insensitive.
    """
    model_clean = model.strip().lower()
    for pump in catalog['pumps']:
        # Some entries might not have 'rpm' field, so we check both existence and value
        pump_model = pump.get('model', '').strip().lower()
        pump_rpm = pump.get('rpm')
        if pump_model == model_clean and pump_rpm == rpm:
            return pump
    return None

def poly_val(coeffs, x):
    """
    Evaluate a polynomial using Horner's method.
    Works for any degree (cubic or quartic).
    """
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result

def interpolate_method1(opt_low, opt_high, factor, var_key):
    """
    Method 1: Linear interpolation of coefficients and ranges for both Q‑H and efficiency.
    """
    v_low = opt_low[var_key]
    v_high = opt_high[var_key]
    v_new = v_low + (v_high - v_low) * factor

    # Q‑H coefficients
    qh_low = opt_low['curves']['q_h']['coefficients']
    qh_high = opt_high['curves']['q_h']['coefficients']
    qh_new = [c_low + (c_high - c_low) * factor
              for c_low, c_high in zip(qh_low, qh_high)]

    # Flow range (linear interpolation)
    qmin_low = opt_low['curves']['q_h']['q_min']
    qmax_low = opt_low['curves']['q_h']['q_max']
    qmin_high = opt_high['curves']['q_h']['q_min']
    qmax_high = opt_high['curves']['q_h']['q_max']
    qmin_new = qmin_low + (qmin_high - qmin_low) * factor
    qmax_new = qmax_low + (qmax_high - qmax_low) * factor

    # Efficiency coefficients and ranges (if both exist)
    eff_coeffs_new = None
    eff_qmin_new = None
    eff_qmax_new = None
    if 'efficiency' in opt_low and 'efficiency' in opt_high:
        eff_low = opt_low['efficiency']['coefficients']
        eff_high = opt_high['efficiency']['coefficients']
        eff_coeffs_new = [c_low + (c_high - c_low) * factor
                          for c_low, c_high in zip(eff_low, eff_high)]
        # Get efficiency ranges, fallback to Q-H range if not present
        eff_qmin_low = opt_low['efficiency'].get('q_min', opt_low['curves']['q_h']['q_min'])
        eff_qmax_low = opt_low['efficiency'].get('q_max', opt_low['curves']['q_h']['q_max'])
        eff_qmin_high = opt_high['efficiency'].get('q_min', opt_high['curves']['q_h']['q_min'])
        eff_qmax_high = opt_high['efficiency'].get('q_max', opt_high['curves']['q_h']['q_max'])
        eff_qmin_new = eff_qmin_low + (eff_qmin_high - eff_qmin_low) * factor
        eff_qmax_new = eff_qmax_low + (eff_qmax_high - eff_qmax_low) * factor

    return {
        'variant': v_new,
        'qh_coeffs': qh_new,
        'qmin': qmin_new,
        'qmax': qmax_new,
        'eff_coeffs': eff_coeffs_new,
        'eff_qmin': eff_qmin_new,
        'eff_qmax': eff_qmax_new,
        'label': f'Method1 (linear coeffs) {v_new:.1f}{"mm" if var_key=="diameter" else "stages"}'
    }

def method2_physical_scaling(opt_low, opt_high, factor, var_key, q_target, h_target):
    """
    Method 2: Build new curve by scaling the closest existing curve using affinity laws.
    Steps:
      1. Interpolate diameter D_new from two bounding curves.
      2. Choose the reference curve (the one with diameter closest to D_new).
      3. Scale that reference curve to D_new using affinity laws.
      4. Refine D_new with one Newton step to hit target point exactly.
      5. Generate points for the new curve and fit a cubic polynomial.
    Returns a dictionary with the new curve data.
    """
    v_low = opt_low[var_key]
    v_high = opt_high[var_key]
    v_new = v_low + (v_high - v_low) * factor

    # Choose reference curve (closest to v_new)
    if abs(v_new - v_low) <= abs(v_new - v_high):
        ref_opt = opt_low
    else:
        ref_opt = opt_high
    d_ref = ref_opt[var_key]
    coeffs_ref = ref_opt['curves']['q_h']['coefficients']
    qmin_ref = ref_opt['curves']['q_h']['q_min']
    qmax_ref = ref_opt['curves']['q_h']['q_max']

    # Scale factor
    k = v_new / d_ref

    # New flow range
    qmin_new = qmin_ref * k
    qmax_new = qmax_ref * k

    # Compute head at target using the scaled curve (first guess)
    q_ref_target = q_target / k
    # Warn if extrapolating
    if q_ref_target < qmin_ref * 0.99 or q_ref_target > qmax_ref * 1.01:
        print(f"⚠️ Target flow {q_target} maps to {q_ref_target:.2f} outside reference range [{qmin_ref}, {qmax_ref}]. Extrapolating.")
    h_target_scaled = (k**2) * poly_val(coeffs_ref, q_ref_target)

    # Refinement: adjust v_new to hit h_target exactly using H ∝ D²
    if h_target_scaled > 0:
        v_new_corrected = v_new * math.sqrt(h_target / h_target_scaled)
        # Clamp to stay between the two bounding diameters (optional but safe)
        v_new_corrected = max(v_low, min(v_high, v_new_corrected))
        if abs(v_new_corrected - v_new) > 1e-3:
            print(f"   Refining diameter: {v_new:.2f} -> {v_new_corrected:.2f}")
            v_new = v_new_corrected
            k = v_new / d_ref
            qmin_new = qmin_ref * k
            qmax_new = qmax_ref * k
            # Recompute head at target
            q_ref_target = q_target / k
            h_target_scaled = (k**2) * poly_val(coeffs_ref, q_ref_target)

    # Generate points for the new curve (100 points)
    num_points = 100
    q_ref_points = np.linspace(qmin_ref, qmax_ref, num_points)
    q_new_points = q_ref_points * k
    h_new_points = (k**2) * np.array([poly_val(coeffs_ref, qr) for qr in q_ref_points])

    # Fit a cubic polynomial to the new points (optional, for consistency with method1)
    coeffs_new = np.polyfit(q_new_points, h_new_points, 3).tolist()

    return {
        'variant': v_new,
        'qh_coeffs': coeffs_new,
        'qmin': qmin_new,
        'qmax': qmax_new,
        'eff_coeffs': None,  # efficiency not scaled here; will be taken from method1
        'label': f'Method2 (physical scaling) {v_new:.1f}{"mm" if var_key=="diameter" else "stages"}'
    }

def get_values_at_q(curve, q, npsh_ref=None, d_ref=None, exponent=1.7, eff_coeffs=None):
    """
    Evaluate head, NPSH (if reference provided), and efficiency (if coefficients provided) at flow q.
    """
    h = poly_val(curve['qh_coeffs'], q)
    npsh = None
    if npsh_ref is not None and d_ref is not None:
        base_npsh = poly_val(npsh_ref['coefficients'], q)
        k = curve['variant'] / d_ref
        npsh = base_npsh * (k ** exponent)
    eff = None
    if eff_coeffs is not None:
        eff = poly_val(eff_coeffs, q)
    return h, npsh, eff

def hydraulic_power(q, h):
    """Compute hydraulic power in kW for water at 20°C (density 1000 kg/m³, g=9.81)."""
    # q in m³/h, h in m -> P (kW) = (q * h * 1000 * 9.81) / (3600 * 1000) = q * h * 9.81 / 3600
    return q * h * 9.81 / 3600

def main():
    parser = argparse.ArgumentParser(description='Pump curve interpolation with physical scaling')
    parser.add_argument('--catalog', default='pump_catalog.json', help='JSON catalog file')
    parser.add_argument('--model', help='Pump model')
    parser.add_argument('--rpm', type=int, help='Rated speed')
    parser.add_argument('--Q', type=float, help='Target flow (m³/h)')
    parser.add_argument('--H', type=float, help='Target head (m)')
    parser.add_argument('--npsh_exp', type=float, default=1.7, help='Exponent for NPSH scaling (default 1.7)')
    args = parser.parse_args()

    # Interactive input if not provided
    catalog_file = args.catalog
    model = args.model
    rpm = args.rpm
    Q_target = args.Q
    H_target = args.H
    npsh_exp = args.npsh_exp

    if not model:
        model = input("Enter pump model: ").strip()
    if not rpm:
        rpm = int(input("Enter rpm: ").strip())
    if not Q_target:
        Q_target = float(input("Enter target flow Q (m³/h): ").strip())
    if not H_target:
        H_target = float(input("Enter target head H (m): ").strip())

    # Load catalog and find pump (case‑insensitive)
    catalog = load_catalog(catalog_file)
    pump = find_pump(catalog, model, rpm)
    if not pump:
        print(f"❌ Pump model '{model}' at {rpm} rpm not found in catalog.")
        return

    print(f"\n✅ Found pump: {pump['model']} at {rpm} rpm (type: {pump.get('type', 'single')})")

    variants = pump['impeller_options']
    if len(variants) < 2:
        print("❌ Need at least two variants (diameters/stages) for interpolation.")
        return

    # Determine variant key and unit
    if 'diameter' in variants[0] and variants[0]['diameter'] is not None:
        var_key = 'diameter'
        unit = 'mm'
    else:
        var_key = 'stages'
        unit = 'stages'

    variants_sorted = sorted(variants, key=lambda x: x[var_key])

    # Evaluate head at target Q for each variant
    heads = []
    for v in variants_sorted:
        coeffs = v['curves']['q_h']['coefficients']
        qmin = v['curves']['q_h']['q_min']
        qmax = v['curves']['q_h']['q_max']
        if Q_target < qmin or Q_target > qmax:
            print(f"⚠️ Q_target {Q_target:.1f} outside range [{qmin:.1f}, {qmax:.1f}] for {v[var_key]}{unit}. Extrapolating.")
        h = poly_val(coeffs, Q_target)
        heads.append((v[var_key], h))

    # Find two variants that bracket H_target (linear interpolation in head space)
    heads_sorted = sorted(heads, key=lambda x: x[1])
    below = heads_sorted[0]
    above = heads_sorted[-1]
    for val, h in heads_sorted:
        if h <= H_target:
            below = (val, h)
        if h >= H_target:
            above = (val, h)
            break

    var_below = next(v for v in variants_sorted if v[var_key] == below[0])
    var_above = next(v for v in variants_sorted if v[var_key] == above[0])

    H_below = below[1]
    H_above = above[1]
    if H_above == H_below:
        factor = 0.5
    else:
        factor = (H_target - H_below) / (H_above - H_below)
    factor = max(0.0, min(1.0, factor))

    # Interpolate using both methods
    curve1 = interpolate_method1(var_below, var_above, factor, var_key)
    curve2 = method2_physical_scaling(var_below, var_above, factor, var_key, Q_target, H_target)

    # Efficiency coefficients and range from method1
    eff_coeffs = curve1.get('eff_coeffs')
    eff_qmin = curve1.get('eff_qmin')
    eff_qmax = curve1.get('eff_qmax')

    # Reference NPSH curve (assume largest diameter)
    max_var = max(variants_sorted, key=lambda x: x[var_key])
    if 'npsh' in max_var['curves']:
        npsh_ref = max_var['curves']['npsh']
        d_ref = max_var[var_key]
        npsh_qmin_ref = npsh_ref.get('q_min', max_var['curves']['q_h']['q_min'])
        npsh_qmax_ref = npsh_ref.get('q_max', max_var['curves']['q_h']['q_max'])
    else:
        npsh_ref = None
        d_ref = None
        npsh_qmin_ref = npsh_qmax_ref = None

    H1, N1, E1 = get_values_at_q(curve1, Q_target, npsh_ref, d_ref, npsh_exp, eff_coeffs)
    H2, N2, _ = get_values_at_q(curve2, Q_target, npsh_ref, d_ref, npsh_exp, None)

    print(f"\n📊 Interpolated variant: {curve1['variant']:.2f} {unit}")
    print(f"   At Q = {Q_target:.1f} m³/h:")
    print(f"     Method1 head: {H1:.2f} m (target {H_target:.2f} m)")
    print(f"     Method2 head: {H2:.2f} m")
    if npsh_ref:
        print(f"     NPSH (scaled, exp={npsh_exp}): {N1:.2f} m")
    if E1 is not None:
        print(f"     Efficiency (linear interp): {E1:.2f}%")

    # Difference metrics (common flow range)
    qmin_common = max(curve1['qmin'], curve2['qmin'])
    qmax_common = min(curve1['qmax'], curve2['qmax'])
    if qmin_common < qmax_common:
        q_common = np.linspace(qmin_common, qmax_common, 100)
        h1_common = [poly_val(curve1['qh_coeffs'], qi) for qi in q_common]
        h2_common = [poly_val(curve2['qh_coeffs'], qi) for qi in q_common]
        mae = np.mean(np.abs(np.array(h1_common) - np.array(h2_common)))
        rmse = np.sqrt(np.mean((np.array(h1_common) - np.array(h2_common))**2))
        max_diff = np.max(np.abs(np.array(h1_common) - np.array(h2_common)))
        print("\n📈 Difference metrics over common flow range:")
        print(f"   MAE  = {mae:.4f}")
        print(f"   RMSE = {rmse:.4f}")
        print(f"   Max  = {max_diff:.4f}")
    else:
        print("\n⚠️ No common flow range for difference calculation.")

    # --- Prepare data for plotting with proper scaled ranges ---
    # Method 2 Q-H curve
    x2 = np.linspace(curve2['qmin'], curve2['qmax'], 100)
    y2 = [poly_val(curve2['qh_coeffs'], xi) for xi in x2]

    # NPSHr curve (scaled from reference)
    if npsh_ref is not None and d_ref is not None:
        k_npsh = curve2['variant'] / d_ref
        npsh_qmin_new = npsh_qmin_ref * k_npsh
        npsh_qmax_new = npsh_qmax_ref * k_npsh
        npsh_qmin_plot = max(npsh_qmin_new, curve2['qmin'])
        npsh_qmax_plot = min(npsh_qmax_new, curve2['qmax'])
        if npsh_qmin_plot < npsh_qmax_plot:
            x_npsh = np.linspace(npsh_qmin_plot, npsh_qmax_plot, 100)
            y_npsh = [poly_val(npsh_ref['coefficients'], xi / k_npsh) * (k_npsh ** npsh_exp) for xi in x_npsh]
        else:
            x_npsh = y_npsh = None
            print("⚠️ NPSH curve has no overlap with Q‑H range after scaling – not plotted.")
    else:
        x_npsh = y_npsh = None

    # Efficiency curve (from Method 1, using its own range)
    if eff_coeffs is not None and eff_qmin is not None and eff_qmax is not None:
        eff_qmin_plot = max(eff_qmin, curve2['qmin'])
        eff_qmax_plot = min(eff_qmax, curve2['qmax'])
        if eff_qmin_plot < eff_qmax_plot:
            x_eff = np.linspace(eff_qmin_plot, eff_qmax_plot, 100)
            y_eff = [poly_val(eff_coeffs, xi) for xi in x_eff]
        else:
            x_eff = y_eff = None
            print("⚠️ Efficiency curve has no overlap with Q‑H range – not plotted.")
    else:
        x_eff = y_eff = None

    # Hydraulic power at target point
    power = hydraulic_power(Q_target, H2)

    # --- Interactive axis limit adjustment with final save option ---
    head_lim = None
    eff_lim = None
    npsh_lim = None
    final_fig = None   # store the last figure for saving

    while True:
        # Create new figure
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_xlabel('Flow rate (m³/h)')
        ax1.set_ylabel('Head (m)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)

        # Plot original curves (gray) on main axis
        for v in variants_sorted:
            coeffs = v['curves']['q_h']['coefficients']
            qmin = v['curves']['q_h']['q_min']
            qmax = v['curves']['q_h']['q_max']
            x = np.linspace(qmin, qmax, 100)
            y = [poly_val(coeffs, xi) for xi in x]
            label = f"{v[var_key]}{unit}"
            if v[var_key] in (below[0], above[0]):
                ax1.plot(x, y, '--', linewidth=1.8, label=label, alpha=0.8)
            else:
                ax1.plot(x, y, 'gray', linewidth=0.8, alpha=0.4)

        # Plot Method 2 Q-H curve (solid red) on main axis
        ax1.plot(x2, y2, 'r-', linewidth=2.5, label=curve2['label'])
        ax1.legend(loc='upper left')

        # Twin axis for efficiency (right side)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Efficiency (%)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        if x_eff is not None:
            ax2.plot(x_eff, y_eff, 'g--', linewidth=2, label='Efficiency (interp)')
            ax2.legend(loc='upper right')

        # Create a second twin axis for NPSHr by offsetting its spine
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('NPSHr (m)', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        if x_npsh is not None:
            ax3.plot(x_npsh, y_npsh, 'b:', linewidth=2, label='NPSHr (scaled)')
            ax3.legend(loc='upper center', bbox_to_anchor=(0.9, 0.9))

        # Mark target point on main axis
        ax1.plot(Q_target, H_target, 'k+', markersize=15, markeredgewidth=3, label='Target point')
        # Add annotation on main axis
        annot_text = (f'Q = {Q_target:.1f} m³/h\n'
                      f'H = {H2:.2f} m\n'
                      f'NPSHr = {N2:.2f} m\n'
                      f'η = {E1:.1f}%\n'
                      f'P₂ = {power:.2f} kW')
        ax1.annotate(annot_text,
                     xy=(Q_target, H_target),
                     xytext=(15, -40),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', color='black'))

        # Apply any previously set limits
        if head_lim:
            ax1.set_ylim(head_lim)
        if eff_lim:
            ax2.set_ylim(eff_lim)
        if npsh_lim:
            ax3.set_ylim(npsh_lim)

        ax1.set_title(f'{model} at {rpm} rpm')
        fig.tight_layout()

        # Store this figure as the final one (will be overwritten each loop)
        final_fig = fig

        # Show the plot (non-blocking)
        plt.show(block=False)
        plt.pause(0.1)

        # Ask user for action
        print("\n📊 Current plot displayed.")
        adjust = input("Adjust axis limits? (y/n): ").strip().lower()
        if adjust != 'y':
            # User is done – keep figure open for saving
            break

        # User wants to adjust, get new limits using current axes
        print("\nEnter new axis limits (leave blank to keep current):")

        # Head axis
        h_min = input(f"  Head min (current: {ax1.get_ylim()[0]:.1f}): ").strip()
        h_max = input(f"  Head max (current: {ax1.get_ylim()[1]:.1f}): ").strip()
        new_head_lim = list(ax1.get_ylim())
        if h_min:
            new_head_lim[0] = float(h_min)
        if h_max:
            new_head_lim[1] = float(h_max)
        head_lim = tuple(new_head_lim)

        # Efficiency axis
        eff_min = input(f"  Efficiency min (current: {ax2.get_ylim()[0]:.1f}): ").strip()
        eff_max = input(f"  Efficiency max (current: {ax2.get_ylim()[1]:.1f}): ").strip()
        new_eff_lim = list(ax2.get_ylim())
        if eff_min:
            new_eff_lim[0] = float(eff_min)
        if eff_max:
            new_eff_lim[1] = float(eff_max)
        eff_lim = tuple(new_eff_lim)

        # NPSH axis
        npsh_min = input(f"  NPSH min (current: {ax3.get_ylim()[0]:.1f}): ").strip()
        npsh_max = input(f"  NPSH max (current: {ax3.get_ylim()[1]:.1f}): ").strip()
        new_npsh_lim = list(ax3.get_ylim())
        if npsh_min:
            new_npsh_lim[0] = float(npsh_min)
        if npsh_max:
            new_npsh_lim[1] = float(npsh_max)
        npsh_lim = tuple(new_npsh_lim)

        # Close current figure before next iteration
        plt.close(fig)

    # After loop, final_fig holds the last figure
    if final_fig:
        save_plot = input("\nSave plot to PNG file? (y/n): ").strip().lower()
        if save_plot == 'y':
            filename = input("Enter filename (default: pump_curve.png): ").strip()
            if not filename:
                filename = "pump_curve.png"
            if not filename.endswith('.png'):
                filename += '.png'
            final_fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {filename}")
        else:
            print("⏭️ Plot not saved.")
        plt.close(final_fig)
    else:
        print("⚠️ No figure to save.")

    print("\n✅ Done.")

if __name__ == '__main__':
    main()