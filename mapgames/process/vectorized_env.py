# external imports
import numpy as np
from multiprocessing import Process, Queue, Event, Pipe
import copy
import time

# internal imports
from . import utils

class ParallelEnv(object):
    def __init__(self, env_fns, seed, default_eval_mode = False):
        """
        Initialies envs for parallel evaluations.
	Input:
	    - env_fns {make_env} - function to create envs instances
	    - seed {float} - seed for envs
	    - default_eval_mode {bool} - do not store the transition for the replay buffer by default
	Output: /
        """

        # Create queues
        self.eval_in_queue = Queue() # Actors to evaluate
        self.eval_out_queue = Queue() # Actors evaluated
        self.transitions_queue = Queue() # Transitions (for replay buffer)

        # connexions: n_processes firsts are for envs - last for critic
        self.n_processes = len(env_fns)
        self.remotes, self.locals = zip(*[Pipe() for _ in range(self.n_processes)])

        # Other attributes
        self.close_processes = Event()
        self.steps = None
        self.seed = seed
        self.evaluation_id = 0
        self.default_eval_mode = default_eval_mode

        # Create the env-related processes
        self.processes = [Process(
                target=evaluation_worker, 
                args=(
                    process_id,
                    utils.CloudpickleWrapper(env_fn),
                    self.eval_in_queue,
                    self.eval_out_queue,
                    self.transitions_queue,
                    self.close_processes,
                    self.remotes[process_id],
                    self.seed)
            ) for process_id, env_fn in enumerate(env_fns)]

        # Start them
        for p in self.processes:
            p.daemon = True
            p.start()



    def eval_policy(self, actors, eval_mode=False):
        """
        Put actors in the queue for being evaluated by the function evaluation_worker
        used in the env processes.
	Inputs:
            - actors {list of Actor} - actors to evaluate
	    - eval_mode {bool} - do not store the transition for the replay buffer
	Ouputs: results {list} - results of evaluation 
        """

        self.steps = 0
        N = len(actors)
        results = [None] * N

        # Put the actors in the queue
        for idx, actor in enumerate(actors):
            self.evaluation_id += 1
            self.eval_in_queue.put((idx, copy.deepcopy(actor), self.evaluation_id, eval_mode or self.default_eval_mode))

        # Retrieve the actors from the out queue
        for _ in range(N):
            idx, result = self.eval_out_queue.get()
            fitness, behav_desc, is_alive, nb_steps = result
            for bd in behav_desc:
                assert (bd >= 0) and (bd <= 1), "BD out of range: " + str(bd)
            # Follow number of steps
            self.steps += nb_steps
            # Store the new result
            results[idx] = fitness, behav_desc, is_alive, nb_steps

        # Retrieve all the results
        return results


    def close(self):
        """
        Close all the processes.
        """
        # Set the close_processes flag to True
        self.close_processes.set()
        rng_states = []
        for local in self.locals:
            rng_states.append(local.recv())
        
        # Terminate all the processes
        for p in self.processes:
            p.terminate()
        return [x[1] for x in sorted(rng_states, key=lambda element: element[0])]



def evaluation_worker(process_id,
                    env_fn_wrapper,
                    eval_in_queue,
                    eval_out_queue,
                    transitions_queue,
                    close_processes,
                    remote,
                    master_seed):
    
    '''
    Function that runs the paralell processes for the evaluation
    Inputs:
        - process_id {int} - ID of the process so it can be identified
        - env_fn_wrapper {make_env} - function that when called starts a new environment
        - eval_in_queue {Queue} - queue for incoming actors
        - eval_out_queue {Queue} - queue for outgoing actors
        - transitions_queue {Queue} - queue for outgoing transitions
    '''

    # Start environment simulation
    env = env_fn_wrapper.x()

    # Begin process loop (closed by the close() method)
    while True:
        try:
            # Get a new actor to evaluate
            try:
                # Get next actor in queue
                idx, actor, evaluation_id, eval_mode = eval_in_queue.get_nowait()

                # Set a seed for the episode
                env.seed(int((master_seed + 100) * evaluation_id))

                # Reset the environment and start an evaluation episode
                state = env.reset()
                done = False

                # Eval loop
                while not done:
                    action = actor.select_action(np.array(state)) 
                    next_state, reward, done, _ = env.step(action)
                    done_bool = float(done) if env.T < env._max_episode_steps else 0
                    # First step
                    if env.T == 1:
                        state_array = state
                        action_array = action
                        next_state_array = next_state
                        reward_array = reward
                        done_bool_array = done_bool
                    # Any other step
                    else:
                        state_array = np.vstack((state, state_array))
                        action_array = np.vstack((action, action_array))
                        next_state_array = np.vstack((next_state, next_state_array))
                        reward_array = np.vstack((reward, reward_array))
                        done_bool_array = np.vstack((done_bool, done_bool_array))

                    state = next_state

                # Retrieve data computed by the environment - correspond to the evaluation of the controller
                eval_out_queue.put((idx, (env.tot_reward, env.desc, env.alive, env.T)))

                # If not in eval_mode
                if not eval_mode:
                    # Send all the transitions encountered during the episode
                    l = len(state_array)
                    bd_array = np.ones((l, 1)) * env.desc
                    transitions_queue.put((idx, (state_array, 
                                                 action_array, 
						 next_state_array, 
						 reward_array, 
						 done_bool_array, 
						 bd_array)))
            except:
                pass

            # If close
            if close_processes.is_set():
                print(f"Close Eval Process nr. {process_id}")
                remote.send((process_id, env.np_random.get_state()))
                # Close the gym env
                env.close()
                time.sleep(10)
                break

        # Handle keyboard interruption
        except KeyboardInterrupt:
            env.close()
            break




