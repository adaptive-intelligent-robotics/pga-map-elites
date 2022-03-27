# external imports
import argparse
import gym


# internal import
import mapgames
from run_experiment import ENV_LIST


############# Read inputs
def read_input():
    """
    Read the inputs as an arg parser.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", default="none", type=str, help="Path where to save results")
    parser.add_argument("--results_path", default=".", type=str, help="Path to store analysis results")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--env", default="all", type=str, help="Env names separated with | (default: all)")
    parser.add_argument("--algo", default="all", type=str, help="Algo names separated with | (default: all)")

    parser.add_argument("--max_evals", default=1000000, type=int, help="Nr of evaluations for the archive to plot")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="Nr individuals per generation")
    parser.add_argument("--save_stat_period", default=20000, type=int, help="Period to save metrics")

    parser.add_argument("--min_fitness", default="", type=str, help="Min-fitness for analysis plots (default: none)")
    parser.add_argument("--max_fitness", default="", type=str, help="Max-fitness for analysis plots (default: none)")

    parser.add_argument("--neurons_list", default="128 128", type=str, help="Actor NN: [neurons_list + [action dim]]")
    parser.add_argument("--normalise", action="store_true", help="Use layer norm")
    parser.add_argument("--affine", action="store_true", help="Use affine transormation with layer norm")

    parser.add_argument("--archive", action="store_true", help="Plot the archives")
    parser.add_argument("--progress", action="store_true", help="Plot the progresss graphs")
    parser.add_argument("--variation", action="store_true", help="Plot the variations graphs")
    parser.add_argument("--visualisation", action="store_true", help="Plot the visualisation graphs")

    parser.add_argument("--save_videos", action="store_true", help="Save visualisation videos")
    parser.add_argument("--p_values", action="store_true", help="Compute p-values")
    
    args = parser.parse_args()
    args.neurons_list = [int(x) for x in args.neurons_list.split()]

    return args

############# Main

if __name__ == "__main__":

    args = read_input()
    if args.variation and not(args.progress):
        print("!!!WARNING!!! Cannot plot variaiton without progress")

    # Read algos and envs
    algo_list = [] if args.algo == "all" else args.algo.rstrip().split('|')
    env_list = ENV_LIST if args.env == "all" else args.env.rstrip().split('|')

    # Create Actor instance for each environment
    env_actors = {}
    for env_name in env_list:
        assert env_name in ENV_LIST, "!!!ERROR!!! Unvalid environment name" + str(env_name)
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        env_actors[env_name] = mapgames.learning.Actor(state_dim, action_dim, max_action, args.neurons_list, \
                                                       normalise=args.normalise, affine=args.affine)
        env.close()


    # Start plots
    mapgames.analysis.launch_plots(
            args.save_path, 
            algo_list,
	    args.max_evals, 
            args.eval_batch_size,
            args.save_stat_period,
            env_list,
            env_actors,
	    save_path = args.results_path,
	    min_fit = None if args.min_fitness == "" else float(args.min_fitness),
	    max_fit = None if args.max_fitness == "" else float(args.max_fitness),
	    archive = args.archive,
	    progress = args.progress,
	    variation = args.variation,
	    visualisation = args.visualisation,
            save_videos = args.save_videos,
	    p_values = args.p_values,
	    verbose = args.verbose
            )
