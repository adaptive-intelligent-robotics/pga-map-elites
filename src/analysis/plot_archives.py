import os

import matplotlib as mpl

from src.analysis.plot_cartesian_maps import plot_cartesian_map
from src.analysis.plot_cvt_maps import plot_cvt_map
from src.analysis.utils import (
    find_cell_files,
    find_centroids,
    get_files,
    get_variant,
)

#######################################
# Global variable for archive plotting

CARTESIAN_ARCHIVES = [
    "QD-RL",
    "QDRL",
    "MAP-Elites-ES",
]  # List of variants that have cartesian archives
ENV_ARCHIVES = [
    "QDHalfCheetahBulletEnv-v0",
    "QDWalker2DBulletEnv-v0",
    "QDDeterministicHalfCheetahBulletEnv-v0",
    "QDDeterministicWalker2DBulletEnv-v0",
]  # List of tasks that are plottable (ie 2D)
BD_AXIS = [
    ["Leg 1 proportion of contact-time", "Leg 2 proportion of contact-time"],
    ["Leg 1 proportion of contact-time", "Leg 2 proportion of contact-time"],
    ["Leg 1 proportion of contact-time", "Leg 2 proportion of contact-time"],
    ["Leg 1 proportion of contact-time", "Leg 2 proportion of contact-time"],
]  # List of BD dimension names for plots


#############################
# Main archive plot function


def plot_archive(
    prefixe,
    exp_path,
    save_path,
    env_names,
    algo_names,
    max_evals,
    min_fit=False,
    max_fit=False,
    verbose=False,
):
    """
    Find all archives files and plot them.
    """
    archive_plotable = True

    # Find all archive plots
    (
        archive_data_files,
        archive_data_types,
        archive_data_additional_files,
        archive_bd_axis,
    ) = find_archives(prefixe, exp_path, save_path, env_names, algo_names, max_evals)
    print(
        "\nReading and printing {} archive data files:".format(len(archive_data_files))
    )
    print(archive_data_files)

    # Plot all archives
    for idx, archive_file in enumerate(archive_data_files):
        archive_plot_filename = archive_file[archive_file.rfind("/") + 1 :].replace(
            ".dat", ""
        )
        archive_plot_filename = archive_plot_filename.replace(
            "BulletEnv-v0", ""
        ).replace("2D", "")
        if archive_data_types[idx]:
            if os.path.isfile(archive_data_additional_files[idx][0]) and os.path.isfile(
                archive_data_additional_files[idx][1]
            ):
                plotable = plot_cartesian_map(
                    archive_file,
                    archive_data_additional_files[idx][0],
                    archive_data_additional_files[idx][1],
                    save_path,
                    archive_plot_filename,
                    min_fit=min_fit,
                    max_fit=max_fit,
                    verbose=verbose,
                    colormap=mpl.cm.viridis,
                    bd_axis=archive_bd_axis[idx],
                )
            else:
                plotable = True
                print(
                    "Cells files",
                    archive_data_additional_files[idx][0],
                    "or",
                    archive_data_additional_files[idx][1],
                    "does not exist.",
                )
        else:
            if os.path.isfile(archive_data_additional_files[idx]):
                plotable = plot_cvt_map(
                    archive_file,
                    archive_data_additional_files[idx],
                    save_path,
                    archive_plot_filename,
                    min_fit=min_fit,
                    max_fit=max_fit,
                    verbose=verbose,
                    colormap=mpl.cm.viridis,
                    bd_axis=archive_bd_axis[idx],
                )
            else:
                plotable = True
                print(
                    "Centroid file",
                    archive_data_additional_files[idx],
                    "does not exist.",
                )
        if not plotable:
            archive_plotable = False
            break
    return archive_plotable


#############################
# Sub-function to find files


def find_archives(prefixe, exp_path, save_path, env_names, algo_names, max_evals):
    """
    Find archive files
    """
    archive_data_files = []
    archive_bd_axis = []
    archive_data_types = []
    archive_data_additional_files = []
    for env_name in env_names:
        if env_name in ENV_ARCHIVES:
            centroid_file = find_centroids(env_name)
            env_bd_axis = BD_AXIS[ENV_ARCHIVES.index(env_name)]
            if algo_names == []:
                # These files
                archive_data_file = get_files(
                    data_path=exp_path,
                    variant="",
                    env=env_name,
                    evals="_" + str(max_evals),
                    prefixe=prefixe,
                    filetype=".dat",
                )
                variants = [get_variant(f, env_name) for f in archive_data_file]
                archive_data_type = [v in CARTESIAN_ARCHIVES for v in variants]
                bd_axis = [env_bd_axis for _ in variants]

                # Add to global files
                archive_data_files += archive_data_file
                archive_bd_axis += bd_axis
                archive_data_types += archive_data_type
                archive_data_additional_files += [
                    find_cell_files(archive_data_file[i], variants[i])
                    if archive_data_type[i]
                    else centroid_file
                    for i in range(len(archive_data_file))
                ]
            for algo_name in algo_names:
                # These files
                archive_data_file = get_files(
                    data_path=exp_path,
                    variant=algo_name,
                    env=env_name,
                    evals="_" + str(max_evals),
                    prefixe=prefixe,
                    filetype=".dat",
                )
                archive_data_type = [
                    algo_name in CARTESIAN_ARCHIVES
                    for _ in range(len(archive_data_file))
                ]
                bd_axis = [env_bd_axis for _ in range(len(archive_data_file))]

                # Add to global files
                archive_data_files += archive_data_file
                archive_bd_axis += bd_axis
                archive_data_types += archive_data_type
                archive_data_additional_files += [
                    find_cell_files(archive_data_file[i], algo_name)
                    if archive_data_type[i]
                    else centroid_file
                    for i in range(len(archive_data_file))
                ]
    assert len(archive_data_files) == len(archive_bd_axis)
    assert len(archive_data_files) == len(archive_data_types)
    assert len(archive_data_files) == len(archive_data_additional_files)
    return (
        archive_data_files,
        archive_data_types,
        archive_data_additional_files,
        archive_bd_axis,
    )
