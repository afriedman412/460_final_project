import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp as pl


def lp_relaxation_duals(sched, msg=False, time_limit=None, fix_ft_offsets=True,
                        restore_mip_values=True):
    """
    Run LP relaxation to get duals/reduced costs. Optionally *fix FT offsets*
    to the MIP-chosen ones to avoid the '0.5 every day' smear.

    Returns: (lp_status, duals_dict, reduced_costs_dict, lp_under_dict, lp_over_dict)
    """
    if sched.model is None:
        sched.build()

    # snapshot current solution and var metadata so we can restore afterward
    vars_ = sched.model.variables()
    snap_val = {
        v.name: (None if v.varValue is None else float(v.varValue)) for v in vars_}
    snap_cat = {v.name: v.cat for v in vars_}
    snap_lb = {v.name: v.lowBound for v in vars_}
    snap_ub = {v.name: v.upBound for v in vars_}

    # Optionally fix FT offsets to the MIP-chosen offset
    fixed_names = set()
    if fix_ft_offsets:
        for (d, s), z in sched.zft.items():
            val = z.varValue
            if val is None:
                continue
            chosen = 1.0 if val > 0.5 else 0.0
            z.lowBound = chosen
            z.upBound = chosen
            fixed_names.add(z.name)

    # Relax integrality for everything (bounds above keep FT offsets fixed)
    for v in vars_:
        v.cat = pl.LpContinuous

    # Solve LP
    solver = pl.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    sched.model.solve(solver)
    status = pl.LpStatus[sched.model.status]

    # Collect duals / reduced costs
    duals = {name: c.pi for name, c in sched.model.constraints.items()}
    rc = {v.name: v.dj for v in sched.model.variables()}

    # Also capture LP under/over values for sanity checks
    lp_under = {(h, t): float(sched.u[(h, t)].varValue or 0.0)
                for h in sched.hospitals for t in range(sched.days)}
    lp_over = {(h, t): float(sched.o[(h, t)].varValue or 0.0)
               for h in sched.hospitals for t in range(sched.days)}

    # Restore original categories/bounds and (optionally) MIP var values
    for v in vars_:
        v.cat = snap_cat[v.name]
        v.lowBound = snap_lb[v.name]
        v.upBound = snap_ub[v.name]
        if restore_mip_values and snap_val[v.name] is not None:
            v.varValue = snap_val[v.name]
            v.setInitialValue(snap_val[v.name])

    return status, duals, rc, lp_under, lp_over


def shadow_price_df(sched, duals):
    Hs, T = list(sched.hospitals), range(sched.days)
    data = np.zeros((len(Hs), sched.days))
    for i, h in enumerate(Hs):
        for t in T:
            data[i, t] = float(duals[f"balance_{h}_{t}"])
    return pd.DataFrame(data, index=Hs, columns=[f"Day {t+1}" for t in T])


def min_reduced_cost_df(sched, reduced_costs, clip=1e5):
    """
    Min reduced cost over doctors for each (hospital, day). We clip extremes to
    make plots readable.
    """
    Hs, T = list(sched.hospitals), range(sched.days)
    data = np.full((len(Hs), sched.days), np.nan)
    for i, h in enumerate(Hs):
        for t in T:
            vals = []
            for d in sched.doc_names:
                key = (d, h, t)
                if key in sched.x:
                    vname = sched.x[key].name
                    rc = reduced_costs.get(vname, np.nan)
                    if np.isfinite(rc):
                        # clip huge magnitudes (numerical artifacts)
                        if rc < -clip:
                            rc = -clip
                        if rc > clip:
                            rc = clip
                        vals.append(rc)
            if vals:
                data[i, t] = np.min(vals)
    return pd.DataFrame(data, index=Hs, columns=[f"Day {t+1}" for t in T])


def plot_sensitivity(sched, sp_df, mr_df, normalize=True, show=True, figsize=(14, 8), cbar=True, suptitle=None):
    # Normalize shadow prices to [-1,1] by penalties for a clean legend
    sp = sp_df.values.copy()
    if normalize:
        for i in range(sp.shape[0]):
            for j in range(sp.shape[1]):
                v = sp[i, j]
                sp[i, j] = (
                    v / sched.pen_under) if v >= 0 else (v / (-sched.pen_over))
                sp[i, j] = max(-1.0, min(1.0, sp[i, j]))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    im1 = axes[0].imshow(sp, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
    axes[0].set_yticks(np.arange(len(sp_df.index)))
    axes[0].set_yticklabels(sp_df.index)
    axes[0].set_title("Shadow prices")
    if cbar:
        cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.025, pad=0.02)
        cb1.set_label("Shadow")

    mr = mr_df.values.copy()
    finite = mr[np.isfinite(mr)]
    if np.any(finite < 0):
        scale = np.percentile(-finite[finite < 0], 90)
        vmin, vmax = -max(scale, 1e-6), 0.0
    else:
        vmin, vmax = -1.0, 0.0
    im2 = axes[1].imshow(mr, aspect='auto', cmap='Greens',
                         vmin=vmin, vmax=vmax)
    axes[1].set_yticks(np.arange(len(mr_df.index)))
    axes[1].set_yticklabels(mr_df.index)
    axes[1].set_xticks(np.arange(sched.days))
    axes[1].set_xticklabels(
        [f"Day {t+1}" for t in range(sched.days)], rotation=90)
    axes[1].set_title("Min reduced cost per (hospital, day)")
    if cbar:
        cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.025, pad=0.02)
        cb2.set_label("Reduced cost (min)")

    if suptitle is None:
        suptitle = "Sensitivity overview (LP with FT offsets fixed to MIP)"

    fig.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig, axes
