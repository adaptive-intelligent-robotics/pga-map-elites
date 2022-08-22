from functools import partial

from mapgames.mapping.cell import Cell
from mapgames.mapping.genotype import genotype_to_actor
from mapgames.mapping.grid import add_to_archive
from mapgames.mapping.individual import Individual

##############
# Save Archive


def save_archive(
    archive,
    n_evals,
    archive_name,
    save_path,
    save_models=False,
    prefixe="archive",
    suffixe="",
):
    """
    Save the archive status
    Input:
    - archive {dict} - archive to write
        - n_evals {int} - number of evaluations for file name
        - archive_name {str} - main file name
        - save_path {str}
        - save_models {bool} - also save the archive for resume as a model
        - prefixe {str} - prefixe for file name
        - suffixe {str} - prefixe for file name
    Output: /
    """

    filename = (
        f"{save_path}/{prefixe}_{archive_name}_" + str(n_evals) + f"{suffixe}.dat"
    )

    with open(filename, "w") as f:
        if len(archive) > 0:
            for k in archive.values():
                f.write(str(k.fitness) + " ")
                write_array(k.centroid, f)
                write_array(k.desc, f)
                f.write(str(k.x.id) + " ")
                f.write("\n")
                if save_models:
                    k.x.save(f"{save_path}/models/{archive_name}_actor_" + str(k.x.id))


############################################
# Alternative save for archive: save actors


def save_actor(s, n_evals, actors_file):
    actors_file.write(
        "{} {} {} {} {} {}\n".format(
            n_evals,
            s.x.id,
            s.fitness,
            str(s.desc).strip("[]"),
            str(s.centroid).strip("()"),
            s.x.novel,
        )
    )
    actors_file.flush()


def save_actors(archive, n_evals, actors_file):
    if len(archive) > 0:
        for s in archive.values():
            save_actor(s, n_evals, actors_file)


########
# Utils


def make_hashable(array):
    return tuple(map(float, array))


def write_array(a, f):
    for i in a:
        f.write(str(i) + " ")


###################
# Utils for PyRibs


def get_archive_stat(archive, dim_gen, dim_map, actor_fn, kdt, cell_fn, pyribs):
    """
    Return the archive to use to compute and write stats.
    Inputs:
        - archive {dict or ribs.archives.CVtArchive} - archive object
        - dim_gen {int}
        - dim_map {int}
        - actor_fn {Actor} - actor default architecture
        - kdt {KDTree} - archive addition mechanism for archive
        - cell_fn {Cell} - cell structure inside the archive
        - pyribs {bool} - 1 the archive is a PyRibs object or 0 it is a simple dict
    Ouputs: archive_stat {dict} - the archive to use for stats
    """
    if pyribs:
        archive_stat = {}
        dataframe = archive.as_pandas(True, False)
        if dataframe.shape[0] > 0:
            for line in range(dataframe.shape[0]):
                desc = [
                    dataframe.at[line, "behavior_" + str(i)] for i in range(dim_map)
                ]
                actor = genotype_to_actor(
                    [dataframe.at[line, "solution_" + str(i)] for i in range(dim_gen)],
                    actor_fn(),
                )
                indiv = Individual(
                    actor,
                    desc,
                    dataframe.at[line, "objective"],
                    dataframe.at[line, "objective"],
                )
                add_to_archive(indiv, desc, archive_stat, kdt, cell_fn)
            assert (
                len(archive_stat) == dataframe.shape[0]
            ), "!!!ERROR!!! Not all indivs from Pyribs archive added."
        return archive_stat
    else:
        return archive


#################################
# Archive-reevaluation functions


def evaluate_archive(archive, kdt, envs, nb_reeval=50):
    """
    Reevaluate archive to get a better idea of the impact of stochasticity
    Input:
        - archive {dict} - archive to reevaluate
        - envs {ParallelEnv} - env to used to reevaluate
        - nb_reeval {int} - number of reevaluations
    Output:
        - new_archive {dict} - archive with individuals reevaluated nb_reeval
                               times and placed where they truly belong
        - new_robust_archive {dict} - archive with individuals reevaluated
                                      nb_reeval times and placed only if they
                                      were robust in term of descriptor
    """
    # create a new empty archive
    new_archive = {}
    new_robust_archive = {}

    # get the number of controllers in the original archive
    nb_controllers = len(archive)

    # create an array to store the individuals to evaluate
    to_evaluate = []
    old_niches = []

    # loop
    for niche, cell in archive.items():
        old_niches += [niche]
        # individuals -> we extract the actor, used by eval_policy()
        for _ in range(nb_reeval):
            indiv = cell.select()
            to_evaluate += [indiv.x]
    assert len(old_niches) == nb_controllers

    # evaluate all the controllers - in eval mode (don't store transitions)
    evaluations = envs.eval_policy(to_evaluate, eval_mode=True)

    # collect results for each individual before reevaluation
    for indiv_idx in range(nb_controllers):
        evaluations_indiv = evaluations[
            indiv_idx * nb_reeval : (indiv_idx + 1) * nb_reeval
        ]
        estim_fitness = 0
        estim_bd = 0

        # loop through the re-evaluations to store results
        for evaluation in evaluations_indiv:
            fitness, descriptor, alive, time = evaluation
            estim_fitness += fitness
            estim_bd += descriptor

        # get the estimated fitness, bd
        estim_fitness /= nb_reeval
        estim_bd /= nb_reeval

        # try to add to new archives
        individual = Individual(
            to_evaluate[indiv_idx * nb_reeval], estim_bd, estim_fitness
        )
        cell_fn = partial(Cell, 1)  # Reeval archives use vanilla MAP-Elites grids
        add_to_archive(
            individual, individual.desc, new_archive, kdt, cell_fn, reeval=True
        )
        add_on_robustness(individual, new_robust_archive, kdt, old_niches[indiv_idx])

    print("    -> Archive has ", nb_controllers, " controllers.")
    print("    -> New archive re-estimated has ", len(new_archive), " controllers.")
    print(
        "    -> New archive re-estimated on robustness has ",
        len(new_robust_archive),
        " controllers.",
    )

    return new_archive, new_robust_archive


def add_on_robustness(individual, archive, kdt, old_niche):
    """
    Add individual to robust archive
    Input:
        - individual {Individual} - individual to add
        - archive {dict} - archive to add to
        - kdt {KDTree} - used to add to archive
        - old_niche {int} - former individual niche index
    Output: archive {dict} - updated archive
    """
    # Find individual niche
    niche_index = kdt.query([individual.desc], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = make_hashable(niche)

    # Add niche information to the individual
    individual.centroid = n

    if old_niche == n:  # if individual is robust, add to archive
        archive[n] = Cell(1)
        return archive[n].add(individual)
    return 0  # else, do not add
