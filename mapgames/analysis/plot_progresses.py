import os
import traceback

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ranksums
from seaborn.relational import _LinePlotter

from mapgames.analysis.collect_data import collect_data
from mapgames.analysis.utils import (
    adjust_box_widths,
    customize_axis,
    first_second_third_quartile,
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

#####################
# Main-plot function


def plot_progress(
    *,
    data_path,
    save_path,
    case_name,
    hue="Algorithm",
    variant_names=["MAP-Elites-PG"],
    environment_names=["QDWalker2DBulletEnv-v0"],
    batch_size=100,
    stat_period=20000,
    verbose=False,
    filename="progress",
    p_values=False,
    limit_x=None,
    stackplot=False,
):
    """
    Plot values of different metrics over the time (time must be understood
    as number of evaluations). Done for each algo variant and for each environment.
    Inputs:
        - data_path {str}
        - save_path {str}
        - case_name {str}: complement to name for the graphs (often indicate max evals)
        - hue {str}: hue for graph
        - variant_names {list}: variants to plot (can be a subset of existing ones)
        - environment_names {list}
        - verbose {bool}
        - filename {str}: select files to plot depending on their prefixe
        - p_values {bool}: if True, compute p-values
        - limit_x {float}: if not Nont, limit max x-axis value
        - stackplot {bool}: if True plot stacked data, if False, plot lineplot
    Outputs: /
    """

    # Collect the datas
    verbose and print("\nCollecting data.")
    data = collect_data(
        data_path=data_path,
        variant_names=variant_names,
        environment_names=environment_names,
        batch_size=batch_size,
        stat_period=stat_period,
        filename=filename,
        verbose=verbose,
    )
    if data.empty:
        print("No progress graphs.")
        return
    verbose and print("Data collected \nPlotting graphs.")

    # Plot parameters
    params = {
        "lines.linewidth": 4,
        "axes.labelsize": 32,
        "legend.fontsize": 32,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "font.size": 32,
        "figure.figsize": [12, 7],
    }
    mpl.rcParams.update(params)

    # Create color frame
    labels = data[hue].drop_duplicates().values
    colors = sns.color_palette("husl", 6)
    color_frame = pd.DataFrame(
        data={"Label": labels, "Color": colors[: len(labels)][:]}
    )

    # Create replicaiton frame
    replication_frame = pd.DataFrame(
        columns={"Experiment", "Algorithm", "Max Evaluations", "Replications"}
    )
    loss_frame = pd.DataFrame(
        columns={"Experiment", "Algorithm", "Max Evaluations", "Stat", "Prop Loss"}
    )

    # Print all graphs
    for exp in environment_names:
        exp_data = data[data["Experiment"] == exp]
        exp_labels = exp_data[hue].drop_duplicates().values
        if len(exp_labels) == 0:
            continue

        # Set corresponding palette
        exp_palette = color_frame[color_frame["Label"].isin(exp_labels)]["Color"].values
        sns.set_palette(exp_palette)

        # Env name for plots
        name_exp = exp.replace("BulletEnv-v0", "").replace("2D", "")

        # For each stat of this environment
        for stat in exp_data.columns:

            # Ensure the stat exist
            if (
                stat == hue
                or stat == "Evaluations"
                or stat == "Algorithm"
                or stat == "Experiment"
                or stat == "Replication"
            ):
                continue
            if exp_data[stat].empty:
                continue
            if exp_data[stat].isnull().all():
                continue
            if exp_data[stat].dtype == list:
                continue

            # Select dataframe
            if hue != "Algorithm":
                columns = ["Algorithm", "Replication", "Evaluations", hue, stat]
            else:
                columns = ["Algorithm", "Replication", "Evaluations", stat]
            stat_data = exp_data[exp_data[stat] == exp_data[stat]][
                columns
            ]  # Remove NaN + keep only columns

            # Fill in replication frame
            replication_frame = fill_in_replication_frame(
                replication_frame, stat_data, exp
            )

            # Fill in loss frame
            loss_frame = fill_in_loss_frame(
                loss_frame, name_exp, exp_data, stat, stat_data
            )

            # Stat name for plots
            name_stat = stat.replace(" ", "_")
            verbose and print("Plotting", stat, "for", exp)

            try:
                # Stackplot
                if stackplot:
                    # Stack the dataframe for plot
                    values = stack_plot_dataframe(stat_data, hue, stat)

                    # Plot
                    fig = plt.figure()
                    ax = sns.lineplot(
                        x="Evaluations", y=stat, data=values, hue=hue, style=hue
                    )
                    stack_plot_error(values, hue, stat, ax, exp_palette)

                    # Print additional analysis
                    # variation_usage_prints(hue, exp, stat, stat_data)

                # Lineplot
                else:
                    # If ablation plot, use associated color palette
                    if all(
                        [
                            a == "1.00"
                            or a == "0.00"
                            or a == "0.25"
                            or a == "0.50"
                            or a == "0.75"
                            for a in stat_data["Algorithm"].drop_duplicates().values
                        ]
                    ):
                        stat_data["Algorithm"] = stat_data["Algorithm"].astype(float)

                    fig = plt.figure()
                    _LinePlotter.aggregate = first_second_third_quartile
                    ax = sns.lineplot(
                        x="Evaluations",
                        y=stat,
                        data=stat_data,
                        hue=hue,
                        estimator=np.median,
                        ci=None,
                        style=hue,
                    )
                # Customise axis
                customize_axis(ax)
                plt.title(name_exp + " - " + stat)
                if limit_x is not None:
                    plt.xlim([0, limit_x])

                # Save figure
                fig.savefig(
                    os.path.join(
                        save_path, f"{filename}_{name_exp}_{name_stat}_{case_name}.svg"
                    ),
                    bbox_inches="tight",
                )
                plt.close()

            except BaseException:
                print("\n!!!WARNING!!! Error when plotting", stat)
                print(traceback.format_exc(-1))

            # Computting p-values if needed
            try:
                if p_values:
                    verbose and print("Computing p-values of", stat, "for", exp)
                    compute_p_values(
                        stat_data,
                        hue,
                        save_path,
                        stat,
                        name_exp,
                        name_stat,
                        case_name,
                        filename=filename,
                    )
            except BaseException:
                print("\n!!!WARNING!!! Error when computting p-values for", stat)
                print(traceback.format_exc(-1))

    verbose and print("Graphs plotted.")

    # Write replication frame in a file
    print(replication_frame)
    replication_frame.to_csv(
        os.path.join(save_path, f"{filename}_replications_{case_name}.csv"),
        header=None,
        index=None,
        sep=",",
        mode="a",
    )

    # Plot reevaluation loss
    loss_plot(
        loss_frame,
        save_path,
        filename,
        case_name,
        color_frame,
        p_values=p_values,
    )
    loss_frame.to_csv(
        os.path.join(save_path, f"{filename}_losss_{case_name}.csv"),
        header=None,
        index=None,
        sep=",",
        mode="a",
    )
    return data


#########################################
# Variation usage prints helper function


def variation_usage_prints(hue, exp, stat, stat_data):
    """
    Print some stats about variation stats.
    """
    sub_stat_data = stat_data.groupby([hue, "Evaluations"], as_index=False).median()
    print("\nFor", exp, "and", stat, ":")
    total_max_value = 0
    for label in sub_stat_data[hue].drop_duplicates().values:
        print("    For", label, ":")
        max_value = max_number(sub_stat_data, hue, stat, label)
        total_max_value += max_value
        print("        maximum value is", max_value)
        max_value *= 0.01
        sub_sub_stat_data = sub_stat_data[sub_stat_data[hue] == label]
        line_idx = sub_sub_stat_data.shape[0] - 1
        while sub_sub_stat_data.iloc[line_idx][stat] < max_value and line_idx > 0:
            line_idx -= 1
        min_eval_0 = sub_sub_stat_data.iloc[line_idx]["Evaluations"]
        print("        inferior to", max_value, "for evals greater than", min_eval_0)
        for label in sub_stat_data[hue].drop_duplicates().values:
            if line_idx > 0:
                frame = stat_data[
                    (stat_data["Evaluations"] < min_eval_0) & (stat_data[hue] == label)
                ]
                frequency = frame[stat].sum() / frame.shape[0]
                print(
                    "        for",
                    label,
                    "in average",
                    frequency,
                    "indivs are added per gen, during the first phase",
                )
            if line_idx < sub_sub_stat_data.shape[0] - 1:
                frame = stat_data[
                    (stat_data["Evaluations"] > min_eval_0) & (stat_data[hue] == label)
                ]
                frequency = frame[stat].sum() / frame.shape[0]
                print(
                    "        for",
                    label,
                    "in average",
                    frequency,
                    "indivs are added per gen, during the second phase",
                )
    print("The total maximum value is", total_max_value)


####################################
# Replication frame helper function


def fill_in_replication_frame(replication_frame, stat_data, exp):
    """
    Fill in the replication frame.
    """
    for algo in stat_data["Algorithm"].drop_duplicates().values:
        if not (replication_frame.empty) and not (
            replication_frame[
                (replication_frame["Experiment"] == exp)
                & (replication_frame["Algorithm"] == algo)
            ].empty
        ):
            continue
        max_eval = min_number(stat_data, "Algorithm", "Evaluations", algo)
        replication = len(
            stat_data[
                (stat_data["Algorithm"] == algo)
                & (stat_data["Evaluations"] == max_eval)
            ]["Replication"]
            .drop_duplicates()
            .values
        )
        replication_frame = replication_frame.append(
            {
                "Experiment": exp,
                "Algorithm": algo,
                "Max Evaluations": max_eval,
                "Replications": replication,
            },
            ignore_index=True,
        )
        max_eval = max_number(stat_data, "Algorithm", "Evaluations", algo)
        replication = len(
            stat_data[
                (stat_data["Algorithm"] == algo)
                & (stat_data["Evaluations"] == max_eval)
            ]["Replication"]
            .drop_duplicates()
            .values
        )
        replication_frame = replication_frame.append(
            {
                "Experiment": exp,
                "Algorithm": algo,
                "Max Evaluations": max_eval,
                "Replications": replication,
            },
            ignore_index=True,
        )
    return replication_frame


####################################
# Reeval-loss-plots helper function


def fill_in_loss_frame(loss_frame, exp, exp_data, stat, stat_data):
    """
    Fill in the reevaluations losses dataframe for loss plots.
    """

    if not ("Reeval" in stat):
        return loss_frame

    # Find name of the original stat
    non_reeval_stat = ""
    non_reeval_stat_name = stat[7:]
    for other_stat in exp_data.columns:
        if (
            non_reeval_stat_name in other_stat
            and not ("Robust" in other_stat)
            and not (other_stat == stat)
        ):
            if non_reeval_stat != "":
                print(
                    "!!!WARNING!!! two stats correspond to non-reeval version of",
                    stat,
                    ":",
                    other_stat,
                    "and",
                    non_reeval_stat,
                )
            else:
                non_reeval_stat = other_stat

    # If connot find name of the original stat
    if non_reeval_stat == "":
        print("!!!Warning!!! Cannot find non-reeval stat for", stat)
        return loss_frame

    # Compute the loss between reeval and original stat for each algo
    non_reeval_stat_data = exp_data[
        exp_data[non_reeval_stat] == exp_data[non_reeval_stat]
    ]
    for algo in stat_data["Algorithm"].drop_duplicates().values:

        # Add the loss for each replication to the frame
        stat_value = []
        for replication in (
            stat_data[stat_data["Algorithm"] == algo]["Replication"]
            .drop_duplicates()
            .values
        ):
            frame = stat_data[
                (stat_data["Algorithm"] == algo)
                & (stat_data["Replication"] == replication)
            ]
            max_eval = max_number(frame, "Algorithm", "Evaluations", algo)
            stat_value.append(max(frame[frame["Evaluations"] == max_eval][stat].values))
        non_reeval_stat_value = []
        for replication in (
            non_reeval_stat_data[non_reeval_stat_data["Algorithm"] == algo][
                "Replication"
            ]
            .drop_duplicates()
            .values
        ):
            non_reeval_frame = non_reeval_stat_data[
                (non_reeval_stat_data["Algorithm"] == algo)
                & (non_reeval_stat_data["Replication"] == replication)
            ]
            max_eval = max_number(non_reeval_frame, "Algorithm", "Evaluations", algo)
            non_reeval_stat_value.append(
                max(
                    non_reeval_frame[non_reeval_frame["Evaluations"] == max_eval][
                        non_reeval_stat
                    ].values
                )
            )

        for i in range(len(stat_value)):
            if non_reeval_stat_value[i] == 0:
                if stat_value[i] == 0:
                    loss = 0
                else:
                    loss = 100
            else:
                loss = (
                    (non_reeval_stat_value[i] - stat_value[i])
                    / non_reeval_stat_value[i]
                    * 100
                )
            if loss != loss or (loss < 0 and ("Coverage" in stat or "Score" in stat)):
                print(
                    "!!!!WARNING!!!! Incorrect loss value:",
                    loss,
                    "for",
                    stat,
                    "on",
                    exp,
                    "and",
                    algo,
                )
            loss_frame = loss_frame.append(
                {
                    "Experiment": exp,
                    "Algorithm": algo,
                    "Max Evaluations": max_eval,
                    "Stat": stat,
                    "Prop Loss": loss,
                },
                ignore_index=True,
            )
    return loss_frame


def loss_plot(loss_frame, save_path, filename, case_name, color_frame, p_values):
    """
    Plot the reevaluation losses as box plots.
    """

    # For each stat stat one plot
    for stat in loss_frame["Stat"].drop_duplicates().values:
        sub_frame = loss_frame[loss_frame["Stat"] == stat]

        # Set color palette
        if all(
            [
                a == "1.00" or a == "0.00" or a == "0.25" or a == "0.50" or a == "0.75"
                for a in sub_frame["Algorithm"].drop_duplicates().values
            ]
        ):
            sub_frame["Algorithm"] = sub_frame["Algorithm"].astype(float)
            sns.set_palette(sns.cubehelix_palette())
        else:
            exp_labels = sub_frame["Algorithm"].drop_duplicates().values
            exp_palette = color_frame[color_frame["Label"].isin(exp_labels)][
                "Color"
            ].values
            sns.set_palette(exp_palette)

        # Plot
        try:
            # Box plot
            fig = plt.figure(figsize=(35, 10))
            ax = sns.boxplot(
                x="Experiment",
                y="Prop Loss",
                hue="Algorithm",
                data=sub_frame,
                width=0.6,
            )
            customize_axis(ax)
            adjust_box_widths(fig, 0.8)
            ax.set(xlabel=None)
            ax.set(ylabel="Proportion of loss in " + stat + " (%)")
            plt.title("All exp - Loss " + stat)
            name_stat = stat.replace(" ", "_")
            fig.savefig(
                os.path.join(
                    save_path, f"{filename}_loss_boxplot_{name_stat}_{case_name}.svg"
                ),
                bbox_inches="tight",
            )
            plt.close()

            # Computting p-values if needed
            if p_values:
                for exp in sub_frame["Experiment"].drop_duplicates().values:
                    name_exp = exp.replace("BulletEnv-v0", "").replace("2D", "")
                    sub_sub_frame = sub_frame[sub_frame["Experiment"] == exp]
                    sub_sub_frame["Evaluations"] = sub_sub_frame["Max Evaluations"]
                    print(sub_sub_frame)
                    try:
                        compute_p_values(
                            sub_sub_frame,
                            "Algorithm",
                            save_path,
                            "Prop Loss",
                            name_exp,
                            name_stat,
                            case_name,
                            filename=filename + "_loss",
                        )
                    except BaseException:
                        print(
                            "\n!!!WARNING!!! Error computing p-values for loss in",
                            stat,
                            "in",
                            exp,
                        )
                        print(traceback.format_exc(-1))

        except BaseException:
            print("\n!!!WARNING!!! Error when plotting loss plot for", stat)
            print(traceback.format_exc(-1))


##############################
# Stack-plot helper functions


def stack_plot_dataframe(frame, hue, stat):
    labels = frame[hue].drop_duplicates().values
    x = frame["Evaluations"].drop_duplicates().values
    values = pd.DataFrame(columns=["Evaluations", stat, hue])
    for i_stack in range(len(labels)):
        sub_frame = frame[frame[hue] == labels[i_stack]]
        if i_stack == 0 or values.empty:
            for stack_x in x:
                value = sub_frame[sub_frame["Evaluations"] == stack_x][stat].median()
                value = value if value == value else 0
                values = values.append(
                    {"Evaluations": stack_x, stat: value, hue: labels[i_stack]},
                    ignore_index=True,
                )
        else:
            for stack_x in x:
                value = (
                    values[
                        (values["Evaluations"] == stack_x)
                        & (values[hue] == labels[i_stack - 1])
                    ][stat].values[0]
                    + sub_frame[sub_frame["Evaluations"] == stack_x][stat].median()
                )
                value = value if value == value else 0
                values = values.append(
                    {"Evaluations": stack_x, stat: value, hue: labels[i_stack]},
                    ignore_index=True,
                )
    return values


def stack_plot_error(frame, hue, stat, ax, exp_palette):
    labels = np.array(frame[hue].drop_duplicates().values)
    x = np.array(frame["Evaluations"].drop_duplicates().values).astype(np.float)
    ax.fill_between(
        x=x,
        y1=[0 for _ in range(len(x))],
        y2=np.array(frame[frame[hue] == labels[0]][stat].values).astype(np.float),
        color=exp_palette[0],
        alpha=0.3,
    )
    for i_stack in range(1, len(labels)):
        ax.fill_between(
            x=x,
            y1=np.array(frame[frame[hue] == labels[i_stack - 1]][stat].values).astype(
                np.float
            ),
            y2=np.array(frame[frame[hue] == labels[i_stack]][stat].values).astype(
                np.float
            ),
            color=exp_palette[i_stack],
            alpha=0.3,
        )


####################
# p-value functions


def get(frame, column, value):
    return frame[frame[column] == value].reset_index(drop=True)


def get_no(frame, column, value):
    return frame[frame[column] != value].reset_index(drop=True)


# Return the maximum number of a column for a given label
def max_number(frame, hue, column, label):
    return get(frame, hue, label)[column].max()


def min_number(frame, hue, column, label):
    return get(frame, hue, label)[column].min()


# Compute one p-value
def p_value_ranksum(frame, hue, reference_label, compare_label, stat):
    _, p = ranksums(
        get(
            get(frame, hue, reference_label),
            "Evaluations",
            max_number(frame, hue, "Evaluations", reference_label),
        )[stat].to_numpy(),
        get(
            get(frame, hue, compare_label),
            "Evaluations",
            max_number(frame, hue, "Evaluations", compare_label),
        )[stat].to_numpy(),
    )
    return p


def compute_p_values(
    frame, hue, save_path, stat, name_exp, name_stat, case_name, filename="progress"
):

    p_frame = pd.DataFrame(columns=["Reference label", "Label", "p-value"])
    labels = frame[hue].drop_duplicates().values

    # For each labels-couple
    for reference_label in labels:
        for compare_label in labels:
            p_frame = p_frame.append(
                {
                    "Reference label": reference_label,
                    "Label": compare_label,
                    "p-value": p_value_ranksum(
                        frame, hue, reference_label, compare_label, stat
                    ),
                },
                ignore_index=True,
            )
    p_frame = p_frame.pivot(index="Reference label", columns="Label", values="p-value")
    p_file = open(
        os.path.join(
            save_path, f"{filename}_pvalue_{name_exp}_{name_stat}_{case_name}.md"
        ),
        "a",
    )
    p_file.write(p_frame.to_markdown())
    p_file.close()
