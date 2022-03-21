# external imports
import numpy as np
import time as lib_time
import copy

# local imports
from . import mapping


def map_elites(actor_fn, dim_gen, dim_map, envs, critic_proc, variations_scheduler, optimizer,
               kdt, archive, cell_fn, n_evals, b_evals, a_evals, counter,
               all_variation_metrics, all_progress_metrics, 
               max_evals, nb_resample, random_init, init_batch_size, eval_batch_size, 
               train_batch_size, nr_of_steps_act, discard_dead, save_stat_period, save_archive_period, nb_reeval,
               archive_filename, save_path, exclude_greedy_actor, best_greedy_actor):
    """
    Algorithm main loop.
    Inputs: 
	    - actor_fn {Actor} - actor default architecture
            - dim_gen {int} - number of genotype dimensions
            - dim_map {int} - number of descriptor dimensions
	    - envs {Vector of envs} - Parallel envs for evaluation
	    - critic_proc {CriticProcess} 
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
	    - max_evals {int} - Nr of evaluations
	    - nb_resample {int} - Resampling before adding to the archive to handle noise
	    - random_init {int} - Nr of init evaluations
            - init_batch_size {int} -  Nr individuals per init generation
            - eval_batch_size {int} - Nr individuals per generation
            - train_batch_size {int} - Batch size for both actor and critic
            - nr_of_steps_act {int} - Nr of training steps for local search
            - discard_dead {bool} - do not keep solutions that do not last all simu
            - save_stat_period {int} - period to save metrics
            - save_archive_period {int} - period to save archive
            - nb_reeval {int} - Nb evals used for better estimation of fitness/bd
            - archive_filename {str} - filename prefixe to save archives
            - save_path {str} - path to save archives
            - exclude_greedy_actor {bool} - Do not consider critic's actor for archive addition
            - best_greedy_actor {bool} - Update critic's greedy actor based on archive's actor
    Outputs:
	    - archive {dict}
	    - greedy_indiv {Individual}
	    - cell_fn {Cell} - cell structure inside the archive
	    - n_evals {int} - total number of evals
	    - b_evals {int} - number of evals since last statistic save
	    - a_evals {int} - number of evals since last archive save
	    - envs {Vector of envs} - Parallel envs for evaluation
	    - critic_proc {CriticProcess} 
	    - variation_scheduler {VariationScheduler} - variations
    """

    
    mapping.Individual._ids = counter
    greedy_indiv = None
    send_greedy = True
    actors = []
    if optimizer != None:
        n_cov_updates = [0 for emitter in optimizer._emitters]

    # Main loop
    while (n_evals < max_evals):

        print(f"\n[{n_evals}/{int(max_evals)}]")
        print(f"Number of solutions: {len(archive)}")
        start = lib_time.time()

        to_evaluate = [] # Offspring to evaluate
        variation_types = [] # Mutation used

        #######################
        # Generate offsprings # 

        # If using PyRibs
        if optimizer != None:
            print("Optimizer loop")
            solutions = optimizer.ask()
            n_cov_updates = [n_cov_updates[i] + int(optimizer._emitters[i].opt.current_eval == optimizer._emitters[i].opt.cov.updated_eval) for i in range (len(optimizer._emitters))]
            for solution in solutions:
                to_evaluate += [mapping.genotype_to_actor(solution, actor_fn())]
                variation_types += ["CMA"]

        # If not using PyRibs and random initialisation
        elif n_evals < random_init: 
            print("Random Loop")
            for i in range(0, init_batch_size):
                to_evaluate += [actor_fn()] # Randomly initialised actors
                variation_types += ["random"]

        # If not using PyRibs and Variation/selection loop
        else: 
            print("Selection/Variation Loop")

            if critic_proc != None:
                critic, actors, states, train_time = critic_proc.get_critic()
            else:
                critic, actors, states, train_time = None, [], None, 0.0
            
            # Add to offsprings
            if exclude_greedy_actor: actors = []
            to_evaluate = copy.deepcopy(actors)
            variation_types = ["greedy" for _ in range(len(actors))]

            # Apply variation
            variations_scheduler.update(n_evals)
            new_controllers, controllers_variations = variations_scheduler.evolve(
                archive,
                eval_batch_size - len(actors),
                critic=critic,
                states=states
            )

            # Add to offsprings
            to_evaluate += new_controllers
            variation_types += controllers_variations

        print(variation_types)

        #######################
        # Evaluate offsprings #

        evaluations = envs.eval_policy(to_evaluate)

        # If resampling 
        if nb_resample > 0 :

            # Initialise matrices for fit and bd with first evaluation
            fitness_evals = [[f] for f, _, _, _ in evaluations]
            bd_evals = [[bd] for _, bd, _, _ in evaluations]

            # Re-evaluate nb_resample times
            for _ in range(nb_resample):

                new_evaluations = envs.eval_policy(to_evaluate, eval_mode=True) # Do not add to replay buffer

                # Fill-in matrices
                for idx, evaluation in enumerate(new_evaluations):
                    fitness, descriptor, alive, time = evaluation
                    fitness_evals[idx] += [fitness]
                    bd_evals[idx] += [descriptor]

            # Use mean of fit and bd as estimate
            for idx in range(len(evaluations)):
                fitness, descriptor, alive, time = evaluations[idx]
                estim_fitness = np.mean(fitness_evals[idx], 0)
                estim_bd = np.mean(bd_evals[idx], 0)
                evaluations[idx] = estim_fitness, estim_bd, alive, time # Update evaluations

        # Update evals
        n_evals += len(to_evaluate) * (nb_resample + 1)
        b_evals += len(to_evaluate) * (nb_resample + 1)
        a_evals += len(to_evaluate) * (nb_resample + 1)

        ##################
        # Add to archive #

        # If using PyRibs
        if optimizer != None:
            fits, bds = [], []
            for evaluation in evaluations:
                fitness, descriptor, alive, time = evaluation
                fits.append(fitness)
                bds.append(descriptor)
            optimizer.tell(fits, bds)
            evaluations = []

        # If not using PyRibs
        for idx, evaluation in enumerate(evaluations):

            fitness, descriptor, alive, time = evaluation

            # If need to be added
            if alive or not(discard_dead):

                # Initialise individual
                s = mapping.Individual(to_evaluate[idx], descriptor, fitness)
				
                # Add to archive
                added_main = mapping.add_to_archive(s, s.desc, archive, kdt, cell_fn)

                # Update metrics
                all_variation_metrics.update(n_evals, variation_types[idx], int(added_main), int(s.x.novel), 
                                             float(s.x.delta_f), float(s.parent_delta_f), float(s.parent_delta_bd))

                # Update greedy actor if relevant
                if greedy_indiv == None:
                    greedy_indiv = s
                elif best_greedy_actor and greedy_indiv.fitness < s.fitness:
                    greedy_indiv = s
                    send_greedy = True
                elif not(best_greedy_actor) and actors != [] and idx == len(actors)-1:
                    greedy_indiv = s
            
            # If no need to be added, still update variation metrics
            else:
                all_variation_metrics.update(n_evals, variation_types[idx], 0, 0, 0, 0, 0)

        # Update greedy actor
        if critic_proc != None and send_greedy:
            critic_proc.update_greedy(greedy_indiv.x)
        if send_greedy:
            send_greedy = False

        # Print iteration time
        end = lib_time.time()
        print("Iteration took:", end - start)


        ############################
        # Save states and archives #

        if b_evals < save_stat_period or save_stat_period == -1:
            continue
        print("\nWritting stats")
        start_stat = lib_time.time()

        # Create the archive to compute stats
        archive_stat = mapping.get_archive_stat(archive, dim_gen, dim_map, actor_fn, kdt, cell_fn, optimizer != None)

        # Write current archive
        if a_evals >= save_archive_period and save_archive_period != -1:
            print("  -> Saving archives")
            mapping.save_archive(archive_stat, n_evals, archive_filename, save_path) # Archive

        # Reevaluate archive
        if nb_reeval > 0 and n_evals > 0:
            print("  -> Reevaluating archives")
            new_archive, new_robust_archive = mapping.evaluate_archive(archive_stat, kdt, envs, nb_reeval)
            
            # Write reeval archive
            if a_evals >= save_archive_period and save_archive_period != -1:
                 print("  -> Saving reeval archives")
                 mapping.save_archive(new_archive, n_evals, archive_filename, save_path, 
                                      save_models=False, suffixe = "re_eval")
                 mapping.save_archive(new_robust_archive, n_evals, archive_filename, save_path,
                                      save_models=False, suffixe="robust")

            print("  -> Saving reeval stats")
            # Write progress
            for a_label, a_archive in [("reeval", new_archive), ("robust", new_robust_archive)]:
                all_progress_metrics.update(a_label, n_evals, a_archive)
                all_progress_metrics.write(a_label)
                all_progress_metrics.reset(a_label)
           
        print("  -> Saving stats")

        # Write progress
        all_progress_metrics.update("classic", n_evals, archive_stat) # update classic progress metrics
        all_progress_metrics.write("classic")
        all_progress_metrics.reset("classic")

        # Update all stat and archive periods
        b_evals = 0
        if a_evals >= save_archive_period and save_archive_period != -1:
            a_evals = 0
        end_stat = lib_time.time()
        print("Finished writting stats, took:", end_stat - start_stat)


    # End of Main loop
    all_variation_metrics.write()
    archive_stat = mapping.get_archive_stat(archive, dim_gen, dim_map, actor_fn, kdt, cell_fn, optimizer != None)
    return archive_stat, greedy_indiv, n_evals, b_evals, a_evals, envs, critic_proc, variations_scheduler
