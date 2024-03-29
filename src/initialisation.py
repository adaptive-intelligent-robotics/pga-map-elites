# external imports
import os
from functools import partial
from itertools import count
from math import floor

import gym
import numpy as np
import QDgym_extended
import torch
import time

# PyRibs
from ribs.archives import CVTArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from sklearn.neighbors import KDTree

# local imports
from src.analysis.launch_plots import launch_plots  # Double check analysis import
from src.learning.actor import Actor
from src.learning.critic import Critic
from src.learning.replay_buffer import ReplayBuffer
from src.mapping.archive_selector import ArchiveSelector
from src.mapping.cell import Cell, DeepGridCell
from src.mapping.genotype import get_dim_gen
from src.mapping.grid import cvt
from src.metrics.progress_metrics import AllProgressMetrics
from src.metrics.variation_metrics import AllVariationMetrics
from src.process.critic_process import CriticProcess
from src.process.critic_worker import critic_worker, td3_critic_worker
from src.process.utils import CloudpickleWrapper, make_env
from src.process.vectorized_env import ParallelEnv
from src.variation.variation_scheduler import VariationScheduler


def init_factory(args):
    """
    Initialise all processes and objects for the main loop.
    Inputs: /
    Outputs:
        - actor_fn {Actor} - actor default architecture
        - dim_gen {int} - number of genotype dimensions
        - dim_map {int} - number of descriptor dimensions
        - envs {Vector of envs} - Parallel envs for evaluation
        - critc_proc {CriticProcess}
        - variation_scheduler {VariationScheduler} - variations
        - optimizer {ribs.optimizer.Optimizer} - optimizer for CMA-ME
        - kdt {KDTree} - archive addition mechanism for archive
        - archive {dict}
        - cell_fn {Cell} - cell structure inside the archive
        - n_evals {int} - total number of evals
        - b_evals {int} - number of evals since last statistic save
        - a_evals {int} - number of evals since last archive save
        - counter {int} - counter to give unique id to all individuals
        - all_variation_metrics {AllVariationMetrics}
        - all_progress_metrics {AllProgressMetrics}
    """
    n_emitters = 5

    ##################
    # Create folders #
    if not os.path.exists(f"{args.save_path}"):
        os.mkdir(f"{args.save_path}")
        os.mkdir(f"{args.save_path}/models/")
        os.mkdir(f"{args.save_path}/checkpoint/")
    if not os.path.exists(f"{args.save_path}/models"):
        os.mkdir(f"{args.save_path}/models/")
    if not os.path.exists(f"{args.save_path}/checkpoint"):
        os.mkdir(f"{args.save_path}/checkpoint/")

    #############
    # Set seeds #
    print("Seed for Pybullet, torch and numpy:", args.seed * int(1e6))
    torch.manual_seed(args.seed * int(1e6))
    np.random.seed(args.seed * int(1e6))

    ############
    # Set envs #

    print("\nInitialising the environments.")
    time.sleep(0.5)

    # Temp env to get env parameters
    temp_env = gym.make(args.env)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    max_action = float(temp_env.action_space.high[0])
    dim_map = temp_env.desc.shape[0]
    temp_env.close()

    # For TD3 keep one cpu to run the algorithms and all others cpu for evaluation
    if "TD3" in args.algo:
        td3_env = CloudpickleWrapper(partial(make_env, args.env))
        args.num_cpu = args.num_cpu - 1

    # General env object
    make_fns = [
        partial(make_env, args.env) for _ in range(args.num_cpu)
    ]  # function for paralell eval
    envs = ParallelEnv(
        make_fns,
        args.seed,
        default_eval_mode=(
            args.algo in ["Deep-grid", "MAP-Elites", "CMA-ME", "TD3"]
        ),  # Do not use eval replay buffer
    )
    time.sleep(0.5)

    ###########################
    # Set actors architecture #

    print("\nInitialising the policies.")

    actor_fn = partial(
        Actor,
        state_dim,
        action_dim,
        max_action,
        args.neurons_list,
        normalise=False,
        affine=False,
    )
    dim_gen = get_dim_gen(actor_fn())

    ########################################
    # Set critics architecture and process #

    # All standard QD algos do not have critic processes
    if args.algo in ["Deep-grid", "MAP-Elites", "CMA-ME"]:
        print("!!!WARNING!!! This algorithm does not use any critic.")
        critic_proc = None
        args.proportion_evo = 1

    # DRL and QD-DRL algo do have critic processes
    else:
        print("\nInitialising the critic process.")

        replay_fn = partial(ReplayBuffer, state_dim, action_dim, dim_map)
        critic_fn = partial(
            Critic,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            expl_noise=0.2,
            policy_noise=0.2 * max_action,
            noise_clip=0.5 * max_action,
            policy_freq=2,
            lr=args.lr_crit,
        )
        # For TD3 the worker run the main algorithm in the background
        if "TD3" in args.algo:
            worker_fn = partial(
                td3_critic_worker,
                CloudpickleWrapper(replay_fn),
                CloudpickleWrapper(critic_fn),
                actor_fn,
                td3_env,
                args.random_init,
                args.num_cpu,
            )
            args.exclude_greedy_actor = False
            args.random_init = 0
        # For other baseline, the worker train the critic
        else:
            worker_fn = partial(
                critic_worker,
                CloudpickleWrapper(replay_fn),
                CloudpickleWrapper(critic_fn),
                args.nr_of_steps_crit,
                args.nr_of_steps_act,
                args.random_init,
                envs.transitions_queue,
            )
        # Main critic process, wrapping the worker
        critic_proc = CriticProcess(worker_fn)

    ##########################
    # Set selection operator #

    print("\nInitialising the algorithm.")
    selection_op = ArchiveSelector("uniform")

    ##################
    # Set variations #

    variations_scheduler = VariationScheduler(selection_op, args.proportion_evo)
    use_ga_variation = not (args.algo in ["TD3", "CMA-ME"])
    use_pg_variation = (args.algo == "PGA-MAP-Elites") and (
        args.proportion_evo < 1 or args.proportion_update < 1
    )
    variations_scheduler.initialise_variations(
        crossover_op=args.crossover_op if use_ga_variation else None,  # Crossover (GA)
        mutation_op=args.mutation_op if use_ga_variation else None,  # Mutation (GA)
        pg_op=use_pg_variation,  # Policy gradient (PG)
        num_cpu=args.num_cpu_var,
        lr=args.lr_act,
        nr_of_steps_act=args.nr_of_steps_act,
    )

    ###############
    # Set metrics #
    all_variation_metrics = AllVariationMetrics(
        variations_scheduler,
        args.save_path,
        args.file_name,
        args.eval_batch_size,
        args.save_stat_period,
    )
    all_progress_metrics = AllProgressMetrics(
        args.save_path, args.file_name, args.nb_reeval
    )

    ###############
    # Set archive #

    # KDTree for cvt archives
    c = cvt(args.n_niches, dim_map, 100000, True)
    kdt = KDTree(c, leaf_size=30, metric="euclidean")  # k-nn for achive addition

    # CMA-ME uses PyRibs archive
    if args.algo == "CMA-ME":
        archive = CVTArchive(
            args.n_niches,
            [(0, 1) for _ in range(dim_map)],
            samples=100000,
            custom_centroids=c,
        )

    # Other algo uses standard archive
    else:
        archive = {}  # init archive (empty)

    # Deep-grid is the only algo using specific cell structures
    if args.algo == "Deep-grid":
        cell_fn = partial(DeepGridCell, args.depth)
    else:
        cell_fn = partial(Cell, args.depth)

    ##############################
    # Set optimiser (for PyRibs) #
    if args.algo == "CMA-ME":
        # ImprovementEmitter, which starts from the search point 0 in N_dim dims space
        emitter_batch = floor(float(args.eval_batch_size) / float(n_emitters))
        emitters = [
            ImprovementEmitter(archive, [0.0] * dim_gen, 0.5, batch_size=emitter_batch)
            for _ in range(n_emitters)
        ]
        print("!!!WARNING!!! INverting matrices of size:", emitters[0].opt.cov.cov.size)
        print(
            "!!!WARNING!!! To compute covariance inversion, use lazy gap eval:",
            emitters[0].opt.lazy_gap_evals,
        )
        # Optimizer that combines the archive and emitter together
        optimizer = Optimizer(archive, emitters)
        args.exclude_greedy_actor = False
    else:
        optimizer = None

    ###################################
    # Set other counter and variables #
    n_evals = 0  # number of evaluations since the beginning
    b_evals = 0  # number of evaluations since the last stat save
    a_evals = 0  # number of evaluations since the last archive save
    counter = count(0)

    print("\nFinished initialisation.\n")
    print("-" * 80 + "\n\n")

    return (
        actor_fn,
        dim_gen,
        dim_map,
        envs,
        critic_proc,
        variations_scheduler,
        optimizer,
        kdt,
        archive,
        cell_fn,
        n_evals,
        b_evals,
        a_evals,
        counter,
        all_variation_metrics,
        all_progress_metrics,
    )
