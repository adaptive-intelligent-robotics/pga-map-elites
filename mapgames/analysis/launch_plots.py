import os

import pandas as pd

from mapgames.analysis.plot_archives import plot_archive
from mapgames.analysis.plot_progresses import plot_progress
from mapgames.analysis.plot_visualisations import plot_visualisation

pd.set_option("display.max_rows", None)  # Plot full and not partial dataframe
pd.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)


def launch_plots(
    exp_path,
    algo_names,
    max_evals,
    batch_size,
    stat_period,
    env_names,
    env_actors,
    save_path="",
    min_fit=False,
    max_fit=False,
    archive=True,
    progress=True,
    variation=True,
    visualisation=True,
    save_videos=False,
    p_values=False,
    verbose=False,
):
    """
    Launch all map and progress plots analysis in exp_path for algo_path.
    Inputs:
        - exp_path {str}: path in which looking for results files
        - algo_names {list}: list of algo names
        - max_evals {int}: maximum eval, used to select archive files to plot
        - batch_size {int}: nr individuals per generation
        - stat_period {int}: period to save metrics
        - env_names {list}: list of environments
        - env_actors {list}: list of actors structure for each environment
        - save_path {str}: path to save the resulting graphs
        - min_fit {float}: minimum fitness used to plot archives with same scale
        - max_fit {float}: maximum fitness used to plot archives with same scale
        - archive {bool}: plot archives
        - progress {bool}: plot progress graphs
        - variation {bool}: plot variation graphs for PGA only
        - visualisation {bool}: plot visualisation
        - save_videos {bool}: save visualisation videos
        - p_values {bool}: compute p-values
        - verbose {bool}
    """

    if save_path == "":
        save_path = exp_path
    elif not os.path.exists(f"{save_path}"):
        os.mkdir(f"{save_path}")

    print("\nAnalysis results in:", exp_path)
    print("\nContent:", os.listdir(exp_path))
    print("\nSaving plots in:", save_path, "\n")

    # Plot archives
    if archive:
        print("\nPlotting archives.")
        plot_archive(
            "archive_",
            exp_path,
            save_path,
            env_names,
            algo_names,
            max_evals,
            min_fit,
            max_fit,
            verbose=verbose,
        )
        print("\nPlotted all archives.\n")

    # Plot progress and variation plots
    if progress:
        print("Reading and printing progress graphs.")
        data = plot_progress(
            data_path=exp_path,
            save_path=save_path,
            case_name=str(max_evals),
            hue="Algorithm",
            variant_names=algo_names,
            environment_names=env_names,
            batch_size=batch_size,
            stat_period=stat_period,
            verbose=verbose,
            filename="progress",
            p_values=p_values,
            limit_x=max_evals,
        )
        print("\nPlotted all progress graphs.\n")

        if variation:
            print("Reading and printing variation graphs.")

            # Plot variation plots for PGA-only
            variation_variants = (
                data[data["Algorithm"].str.contains("PGA-MAP-Elites")]["Algorithm"]
                .drop_duplicates()
                .values
            )
            for variation_variant in variation_variants:
                data = plot_progress(
                    data_path=exp_path,
                    save_path=save_path,
                    case_name=str(max_evals)
                    + "_"
                    + variation_variant.replace(" ", "_"),
                    hue="Label",
                    variant_names=[variation_variant],
                    environment_names=env_names,
                    batch_size=batch_size,
                    stat_period=stat_period,
                    verbose=verbose,
                    filename="variation",
                    p_values=p_values,
                    stackplot=True,
                )
            print("\nPlotted all variation graphs.\n")

    # Plot visualisation
    if visualisation:
        print("Reading and saving visualisations.")
        data = plot_visualisation(
            data_path=exp_path,
            save_path=save_path,
            variant_names=algo_names,
            env_names=env_names,
            env_actors=env_actors,
            save_videos=save_videos,
            verbose=verbose,
        )
        print("\nSaved all visualisations.\n")
