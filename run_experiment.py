# external imports
import numpy as np
import argparse
import multiprocessing
import copy

# internal import
import mapgames

ENV_LIST = [ 
            "QDHalfCheetahBulletEnv-v0", "QDWalker2DBulletEnv-v0", \
            "QDAntBulletEnv-v0", "QDHopperBulletEnv-v0", \
            "QDDeterministicHalfCheetahBulletEnv-v0", "QDDeterministicWalker2DBulletEnv-v0", \
            "QDDeterministicAntBulletEnv-v0", "QDDeterministicHopperBulletEnv-v0"
           ]

ALGO_LIST = [ "PGA-MAP-Elites", "MAP-Elites", "Deep-grid", "TD3", "CMA-ME" ]

############# Read inputs

class LoadFromFile(argparse.Action):
	"""
	Read a config file and import it in the arg parser.
	"""
	def __call__ (self, parser, namespace, values, option_string = None):
		with values as f:
			parser.parse_args([s.strip("\n") for s in f.readlines()], namespace)

def read_input():
	"""
	Read the inputs as an arg parser.
	"""

	parser = argparse.ArgumentParser()

    	#### Run-related inputs

	parser.add_argument('--config_file', type=open, action=LoadFromFile, help="Config file to load args")
	parser.add_argument("--save_path", default="none", type=str, help="Path where to save results")
	parser.add_argument("--save_stat_period", default=1000, type=int, help="Period to save metrics")
	parser.add_argument("--save_archive_period", default=10000, type=int, help="Period to save archives")
	parser.add_argument("--num_cpu", default=48, type=int, help="Nr. of CPUs to use in parallel evaluation")
	parser.add_argument("--num_cpu_var", default=48, type=int, help="Nr. of CPUs to use in parallel variation")

    	#### Task-related inputs

	parser.add_argument("--env", default="QDAntBulletEnv-v0", type=str, help="Env name from "+str(ENV_LIST))
	parser.add_argument("--algo", default="PGA-MAP-Elites", type=str, help="Algo name from "+str(ALGO_LIST))
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--max_evals", default=1000000, type=int, help="Nr of evaluations")
	parser.add_argument("--neurons_list", default="128 128", type=str, help="Actor NN: [neurons_list + [action dim]]")
	parser.add_argument("--nb_reeval", default=50, type=int, help="Nb evals used for better estimation of fitness/bd")

	#### QD-related inputs

	parser.add_argument("--depth", default=1, type=int, help="Depth of cells")
	parser.add_argument("--nb_resample", default=0, type=int, help="Naive sampling to fight noise")
	parser.add_argument("--min_genotype", default=False, help="Minimum value of genes (if False no limit)")
	parser.add_argument("--max_genotype", default=False, help="Maximum value a genes (if False no limit)")
	parser.add_argument("--no_cached_cvt", action="store_true", help="Do not use cached centroids if available")
	parser.add_argument("--cvt_samples", default=100000, type=int, help="Nr of samples to approximate centroid")
	parser.add_argument("--eval_batch_size", default=100, type=int, help="Nr individuals per generation")
	parser.add_argument("--random_init", default=500, type=int, help="Nr of init evaluations")
	parser.add_argument("--init_batch_size", default=100, type=int, help="Nr individuals per init generation")
	parser.add_argument("--discard_dead", action="store_true", help="Do not keep solutions that do not last all simu")
	parser.add_argument("--selector", default="uniform", type=str, help="Selection-type")

	#### Mutation-related inputs

	parser.add_argument("--mutation_op", default=None, help="Mutation operator")
	parser.add_argument("--crossover_op", default="iso_dd", help="Crossover operator (iso_dd = line = mut + cross)")
	parser.add_argument("--mutation_rate", default=0.05, type=float, help="Probablity of mutation")
	parser.add_argument("--crossover_rate", default=0.75, type=float, help="Probablity of crossover")
	parser.add_argument("--eta_m", default=5.0, type=float, help="Parameter for polynomial mutation")
	parser.add_argument("--eta_c", default=10.0, type=float, help="Parameter for Simulated Binary Crossover")
	parser.add_argument("--sigma", default=0.2, type=float, help="Sandard deviation for gaussian mutation")
	parser.add_argument("--iso_sigma", default=0.005, type=float, help="Gaussian parameter in line mutation (iso_dd)")
	parser.add_argument("--line_sigma", default=0.05, type=float, help="Line parameter in line mutation (iso_dd)")
	parser.add_argument("--max_uniform", default=0.1, type=float, help="Max mutation for uniform mutation")

	#### PGA-related inputs

	parser.add_argument("--proportion_evo", default=0.5, type=float, help="Proportion of batch to use GA variation")
	parser.add_argument("--exclude_greedy_actor", action="store_true", help="Do not consider critic's actor for archive addition")
	parser.add_argument("--best_greedy_actor", action="store_true", help="Update critic's actor based on archive's actors")
	parser.add_argument('--nr_of_steps_act', default=50, type=int, help="Nr of training steps for local search (PG variation)")
	parser.add_argument('--nr_of_steps_crit', default=300, type=int, help="Nr of training steps for global search")

	parser.add_argument("--stop_pg", default=100000000, type=int, help="Number of evals before stopping using pg")
	parser.add_argument("--proportion_update", default=1.0, type=float, help="Reduction of p_evo per update")
	parser.add_argument("--proportion_update_period", default=10000000, type=int, help="Period of p_evo reduction")
	parser.add_argument("--lr_update", default=1.0, type=float, help="Reduction of lr per update")
	parser.add_argument("--lr_update_period", default=10000000, type=int, help="Period of lr reduction")
	parser.add_argument("--nr_of_steps_act_update", default=1.0, type=float, help="Reduction of nr_of_steps_act per update")
	parser.add_argument("--nr_of_steps_act_update_period", default=10000000, type=int, help="Period of nr_of_steps_act reduction")

	#### RL-related inputs

	parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size for both actor and critic")
	parser.add_argument("--discount", default=0.99, type=float, help="Discount factor for critic")
	parser.add_argument("--tau", default=0.005, type=float, help="Target networks update rate")
	parser.add_argument("--policy_noise", default=0.2, type=float, help="Noise added to target during critic update")
	parser.add_argument("--expl_noise", default=0.2, type=float, help="Noise added for exploration")
	parser.add_argument("--noise_clip", default=0.5, type=float, help="Range to clip target noise")
	parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed actor updates")
	parser.add_argument("--normalise", action="store_true", help="Use layer norm")
	parser.add_argument("--affine", action="store_true", help="Use affine transormation with layer norm")
	parser.add_argument("--lr_crit", default=3e-4, type=float, help="Learning rate critic")
	parser.add_argument("--lr_act", default=0.005, type=float, help="Learning rate actor")

	args = parser.parse_args()

	# Ensure algorithm building choices are possible 
	assert args.save_stat_period >= args.eval_batch_size, "!!!Error!!! Can't save stats multiple times per gen"
	assert args.save_archive_period >= args.eval_batch_size, "!!!Error!!! Can't save stats multiple times per gen"
	assert args.env in ENV_LIST, "!!!Error!!! Not a valid environment"
	assert args.algo in ALGO_LIST, "!!!Error!!! Not a valid algorithm"
	assert args.proportion_evo == 0 or args.crossover_op != None or args.mutation_op != None, \
           "!!!Error!!! No mutation nor crossover and n_evo set to non-zero."
	assert not(args.algo == "PGA-MAP-Elites" and args.proportion_evo == 1 and \
		   args.proportion_update == 1 and args.exclude_greedy_actor), \
		   "!!!Error!!! Algorithm equivalent to MAP-Elites."

	# Set algo name based on inputs
	args.algo_type = args.algo
	if args.algo == "Deep-grid":
		args.algo_type += "_" + str(args.depth) 
	elif args.algo == "PGA-MAP-Elites":
		args.algo_type += "_" + str(args.proportion_evo)
		if args.exclude_greedy_actor: args.algo_type += "_wo_greedy"
	if args.best_greedy_actor: args.algo_type += "_best_greedy"
	if args.nb_resample > 0: args.algo_type += "_sample-" + str(args.nb_resample + 1)
	if args.algo != "PGA-MAP-Elites" and args.crossover_op == "iso_dd": args.algo_type += "-line"
	if args.iso_sigma != 0.005 and args.crossover_op == "iso_dd": args.algo_type += "_iso" + str(args.iso_sigma)
	if args.line_sigma != 0.05 and args.crossover_op == "iso_dd": args.algo_type += "_line" + str(args.line_sigma)

	# Set grid dimension based on inputs
	if args.env in ["QDHopperBulletEnv-v0", "QDDeterministicHopperBulletEnv-v0"]:
		args.n_niches = 1000 # 1 BD dim
	elif args.env in ["QDAntBulletEnv-v0", "QDDeterministicAntBulletEnv-v0"]:
		args.n_niches = 1296 # 4 BD dims
	else:
		args.n_niches = 1024 # 2 BD dims (most frequent)

        # Pre-process rest of inputs
	args.neurons_list = [int(x) for x in args.neurons_list.split()]
	if args.save_path == "none": args.save_path = "./test_" + args.env + "_" + args.algo_type
	args.file_name = f"{args.algo_type}_{args.env}_{args.seed}"

	return args


############# Main

if __name__ == "__main__":

	# Read inputs
	args = read_input()

	# Plot infos
	print("\n\n" + "-"*80)
	print("Available Envs:")
	for env in ENV_LIST: print("  " + env)
	print("-"*80)
	print(f"Policy: {args.algo_type} \nEnv: {args.env} \nSeed: {args.seed}")
	print("-"*80)
	num_cores = multiprocessing.cpu_count()
	print(f"Number of found_cores: {num_cores}")
	print("-"*80, "\n\n")
	
	# Define and initialise all the import elements
	print("\n\n-------------------- INITIALISATION -------------------\n")

	actor_fn, dim_gen, dim_map, envs, critic_proc, variations_scheduler, optimizer, \
	kdt, archive, cell_fn, n_evals, b_evals, a_evals, counter,\
	all_variation_metrics, all_progress_metrics = mapgames.init_factory(args)

	# Main part : the map-elites algorithm is launched
	print("\n\n---------- BEGINNING OF THE MAP ELITES ALGO ----------\n")

	archive, greedy, n_evals, b_evals, a_evals, envs, critic_proc, variations_scheduler = mapgames.map_elites(
		actor_fn, dim_gen, dim_map, envs, critic_proc, variations_scheduler, optimizer,
		kdt, archive, cell_fn, n_evals, b_evals, a_evals, counter, 
		all_variation_metrics, all_progress_metrics, 
		args.max_evals, args.nb_resample, args.random_init, args.init_batch_size, 
		args.eval_batch_size, args.train_batch_size, args.nr_of_steps_act, 
		args.discard_dead, args.save_stat_period, args.save_archive_period, args.nb_reeval,
		args.file_name, args.save_path, args.exclude_greedy_actor, args.best_greedy_actor
	)

	# Save the final state of archive AND the models in it !
	print("\nSaving archive.")
	mapgames.mapping.save_archive(archive, n_evals, args.file_name, args.save_path, save_models=True)
	mapgames.mapping.save_actors(archive, n_evals, open(f"{args.save_path}/actors_{args.file_name}.dat", 'w'))
	if greedy != None:
		mapgames.mapping.save_actor(greedy, n_evals, open(f"{args.save_path}/greedy_actor_{args.file_name}.dat", 'w'))
	print("Saved archive.")

	# End env evaluation parallel processes
	print("\nClosing envs.")
	env_rng_states = envs.close()
	print("Closed envs.")
	
	# End critic process
	print("\nClosing critic process.")
	if critic_proc != None:
		critic, replay_buffer = critic_proc.close() #this seems to never end atm
		critic.save(f"{args.save_path}/models/{args.file_name}_critic_" + str(n_evals))
	else:
		critic, replay_buffer = False, False
	print("Closed critic process and saved it.")

	# End Variation scheduler
	print("\nClosing variation scheduler.")
	variations_scheduler.close() 
	print("Closed variation scheduler.")
	
	print("\nSaving checkpoint file.")
	mapgames.process.save(args, 
                              n_evals, 
			      b_evals, 
			      a_evals, 
			      archive,
			      greedy,
			      critic, 
			      replay_buffer, 
			      env_rng_states, 
			      kdt, 
			      mapgames.mapping.Individual.get_counter())
	print("Saved checkpoint file.")

	# end of the map elites algo
	print("\n---------- END OF THE MAP ELITES ALGO ----------\n")

