"""
Supercapacitor GCD analysis
---------------------------
Calculates specific capacitance (Cs) from galvanostatic charge-discharge data
at multiple current densities.

    Cs = j * dt / dV        (j in A/g, dt in s, dV in V  ->  F/g)

Input CSV columns:
    current_density_Ag, time_s, potential_V

Usage:
    python gcd_analysis.py
    python gcd_analysis.py my_data.csv --plot --out results.csv
"""

import argparse

import numpy as np
import pandas as pd


def analyse_discharge(g, lo=0.2, hi=0.8):
    """Analyse the first discharge curve in one current-density group.

    The IR drop is removed by fitting a straight line to the linear middle part
    of the discharge (between `lo` and `hi` of the voltage window) and
    extrapolating it back to the moment discharge starts.
    """
    g = g.reset_index(drop=True)

    peak = g["potential_V"].idxmax()                       # discharge starts here
    if peak >= len(g) - 2:
        return None
    trough = g.loc[peak:, "potential_V"].idxmin()          # end of FIRST discharge
    dis = g.loc[peak:trough]
    if len(dis) < 5:
        return None

    t = dis["time_s"].to_numpy()
    v = dis["potential_V"].to_numpy()

    v_peak, v_end = v[0], v[-1]
    window = v_peak - v_end

    # linear middle part of the discharge curve
    mask = (v <= v_peak - lo * window) & (v >= v_peak - hi * window)
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(t[mask], v[mask], 1)     # slope is negative

    v_start = slope * t[0] + intercept                     # voltage after IR drop
    ir_drop = v_peak - v_start

    dt = t[-1] - t[0]
    dV = v_start - v_end

    return {
        "dt (s)": dt,
        "dV (V)": dV,
        "IR drop (V)": ir_drop,
        "t_dis": t,
        "v_dis": v,
    }


def main():
    parser = argparse.ArgumentParser(description="Specific capacitance from GCD data")
    parser.add_argument("file", nargs="?", default="GCD_sample_data.csv",
                        help="CSV file (default: GCD_sample_data.csv)")
    parser.add_argument("--plot", action="store_true",
                        help="plot the selected discharge segments")
    parser.add_argument("--out", help="save the results table to this CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    needed = {"current_density_Ag", "time_s", "potential_V"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV: {', '.join(sorted(missing))}")

    rows, curves = [], {}
    for j, g in df.groupby("current_density_Ag"):
        res = analyse_discharge(g)
        if res is None:
            print(f"Warning: could not analyse j = {j} A/g (too few points?)")
            continue
        curves[j] = (res.pop("t_dis"), res.pop("v_dis"))
        rows.append({"j (A/g)": j, **res, "Cs (F/g)": j * res["dt (s)"] / res["dV (V)"]})

    if not rows:
        raise SystemExit("No discharge curves could be analysed.")

    out = pd.DataFrame(rows).sort_values("j (A/g)")
    print(out.round(2).to_string(index=False))
    print("\nNote: Cs is per the mass used to define j. For a two-electrode cell,"
          "\nconvert to single-electrode (x4 for total mass, x2 for one electrode).")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Saved: {args.out}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for j, (t, v) in curves.items():
            ax1.plot(t - t[0], v, label=f"{j} A/g")
        ax1.set(xlabel="Discharge time (s)", ylabel="Potential (V)",
                title="Selected discharge segments")
        ax1.legend()

        ax2.plot(out["j (A/g)"], out["Cs (F/g)"], "o-")
        ax2.set(xlabel="Current density (A/g)", ylabel="Cs (F/g)",
                title="Cs vs current density")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
