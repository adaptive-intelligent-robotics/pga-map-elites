import os

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch

#############
# Find files


def get_files(
    *, data_path, variant, env, evals="", filetype="*.csv", prefixe="", verbose=False
):
    """
    Return all files name with the env and variant names in it, that contains the
    prefixe and end with the correct filetype (i.e. extension).
    """
    return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(data_path)
        for name in files
        if (
            variant in name
            and env in name
            and evals in name
            and name.endswith(filetype)
            and prefixe in name
        )
    ]


##############################
# Find min or max of dataframe


def find_min_max_prefixe(data, exp, prefixe, use_min=True):
    name = "Min" if use_min else "Max"
    if (prefixe + name + " Fitness") not in data.columns or data[
        data["Experiment"] == exp
    ][prefixe + name + " Fitness"].isnull().values.all():
        return None
    if use_min:
        return data[data["Experiment"] == exp][prefixe + name + " Fitness"].min()
    return data[data["Experiment"] == exp][prefixe + name + " Fitness"].max()


def find_min_max(data, exp, use_min=True):
    """
    Find the min or max fitness of the datas, considering reeval and robust stats.
    """

    extr_fit = find_min_max_prefixe(data, exp, "", use_min=use_min)
    extr_reeval_fit = find_min_max_prefixe(data, exp, "Reeval ")
    extr_fit = (
        extr_fit
        if extr_reeval_fit is None
        else (
            extr_reeval_fit
            if extr_fit is None
            else (
                min(extr_fit, extr_reeval_fit)
                if use_min
                else max(extr_fit, extr_reeval_fit)
            )
        )
    )
    extr_robust_fit = find_min_max_prefixe(data, exp, "Robust ")
    extr_fit = (
        extr_fit
        if extr_robust_fit is None
        else (
            extr_robust_fit
            if extr_fit is None
            else (
                min(extr_fit, extr_robust_fit)
                if use_min
                else max(extr_fit, extr_robust_fit)
            )
        )
    )
    if extr_fit is not None:
        print("Min" if use_min else "Max", "fitness for", exp, "is", extr_fit)
    return extr_fit


##############################
# Find archive-related files


def find_n_niches(env_name):
    if env_name in [
        "QDHalfCheetahBulletEnv-v0",
        "QDWalker2DBulletEnv-v0",
        "QDDeterministicHalfCheetahBulletEnv-v0",
        "QDDeterministicWalker2DBulletEnv-v0",
    ]:
        return 1024
    elif env_name in ["QDAntBulletEnv-v0", "QDDeterministicAntBulletEnv-v0"]:
        return 1296
    else:
        return 1000


def find_centroids(env_name):
    return "./CVT/centroids_" + str(find_n_niches(env_name)) + "_2.dat"


def find_cell_files(filename, variant):
    path = filename[: filename.rfind("/") + 1]
    if "reeval_" in filename:
        if "MAP-Elites-ES" in variant:
            return [
                path + "archive_reeval/cell_ids.pk",
                path + "archive_reeval/cell_boundaries.pk",
            ]
        return [path + "reeval_cell_ids.pk", path + "reeval_cell_boundaries.pk"]
    if "robust_" in filename:
        if "MAP-Elites-ES" in variant:
            return [
                path + "archive_robust/cell_ids.pk",
                path + "archive_robust/cell_boundaries.pk",
            ]
        return [path + "robust_cell_ids.pk", path + "robust_cell_boundaries.pk"]
    if "MAP-Elites-ES" in variant:
        return [path + "archive/cell_ids.pk", path + "archive/cell_boundaries.pk"]
    return [path + "cell_ids.pk", path + "cell_boundaries.pk"]


def get_variant(filename, exp):
    variant = filename[filename.rfind("/") + 1 :]
    if "reeval_" in variant or "robust_" in variant:
        variant = variant[variant.find("_") + 1 :]
        variant = variant[variant.find("_") + 1 :]
    else:
        variant = variant[variant.find("_") + 1 :]
    variant = variant[: variant.find(exp) - 1]
    return variant


#################
# Plot functions


def first_second_third_quartile(self, vals, grouper, units=None):
    """
    Plot first second third quantile in a plot.
    """
    # Group and get the aggregation estimate
    grouped = vals.groupby(grouper, sort=self.sort)
    est = grouped.agg("median")
    min_val = grouped.quantile(0.25)
    max_val = grouped.quantile(0.75)
    cis = pd.DataFrame(
        np.c_[min_val, max_val], index=est.index, columns=["low", "high"]
    ).stack()
    # Unpack the CIs into "wide" format for plotting
    if cis.notnull().any():
        cis = cis.unstack().reindex(est.index)
    else:
        cis = None
    return est.index, est, cis


def customize_axis(ax):
    """
    Customise axis for plots
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis="y", length=0)
    # ax.get_yaxis().tick_left()

    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="--", linewidth=1.5)


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """
    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for line in ax.lines:
                    if np.all(line.get_xdata() == [xmin, xmax]):
                        line.set_xdata([xmin_new, xmax_new])
