import os
import gym
from gym import wrappers
import copy
import numpy as np
import time
import traceback
from . import get_files

def plot_visualisation(*, 
                       data_path, 
                       save_path, 
                       variant_names, 
                       env_names, 
                       env_actors, 
                       save_videos=False,
                       delay=1/60,
                       resolution=(1280, 960),
                       cam_dist = 6,
                       cam_yaw = 45,
                       cam_pitch = -30,
                       verbose=False):

    """
    Plot genotype stats.
    Inputs:
    	- data_path {str}: path in which looking for results files
	- save_path {str}: path to save the results
	- variant_names {list}
	- env_names {list}
	- env_actors {list} - actors associated with each enviroment
        - save_videos {bool} - save as video or display 
        - delay {float}: delay between frames if displaying
	- resolution {dic}: render image size
        - cam_dist {float}
        - cam_yaw {float}
        - cam_pitch {float}
	- verbose {bool}
    """

    # Collect model subfolder
    replication_folders = []
    replication_algos = []
    replication_envs = []
    for root, dirs, files in os.walk(data_path):
        if "/models" in root:
            if len(files) == 0:
                print("!!!WARNING!!! Empty models folder:", root)
                continue
            env_name = None
            for env in env_names:
                if env in files[0]:
                    env_name = env
            if env_name != None:
                variant_name = files[0]
                variant_name = variant_name[:variant_name.find(env_name)-1]
                if variant_name in variant_names or variant_names == []:
                    replication_folders.append(root)
                    replication_algos.append(variant_name)
                    replication_envs.append(env_name)

    # Find actor files for each replication
    for idx, folder in enumerate(replication_folders):

        variant_name = replication_algos[idx]
        env_name = replication_envs[idx]
        env_actor = env_actors[env_name]

        # Load file
        actor_files = get_files(data_path = folder, variant = variant_name, \
                                env = env_name, filetype = "", prefixe = "_actor_")
        if len(actor_files) == 0: 
            print("No candidate actor files for", env_name, "and", variant_name, "in", folder)
            continue
        verbose and print("Candidates:", actor_files)

        # Create env
        if not save_videos:
            env = gym.make(env_name, render=True)
            env.env._render_width, env.env._render_height = resolution
            env.env._cam_dist = cam_dist
            env.env._cam_yaw = cam_yaw
            env.env._cam_pitch = cam_pitch

        # Read actors loop 
        for idx, actor_file in enumerate(actor_files):

            # Preprocess
            actor_id = actor_file[actor_file.rfind("_")+1:]
            print("Running actor", actor_id, "found with", variant_name, "on", env_name)
            print("In file:", actor_file)

            # Create policy
            policy = copy.deepcopy(env_actor)
            policy.load(actor_file)

            # Visualise
            if save_videos:
                try:
                    video_path = os.path.join(save_path, "video_" + variant_name + "_" + env_name + "_" + actor_id)
                    fitness, desc = video_policy(env_name, policy, video_path, resolution, cam_dist, cam_yaw, cam_pitch)
                    print("Final performance:", fitness)
                    print("Final desc:", desc)
                except:
                    print("\n!!!WARNING!!! Error when visualising")
                    print(traceback.format_exc(-1))
            if not save_videos:
                try:
                    fitness, desc = visualise_policy(env, policy, delay)
                    print("Final performance:", fitness)
                    print("Final desc:", desc)
                except:
                    print("\n!!!WARNING!!! Error when visualising")
                    print(traceback.format_exc(-1))

        if not save_videos:
            env.close()
    return


def visualise_policy(env, policy, delay):
    """
    Launch visualisation
    Inputs:
    	- env {Env}
	- policy {Actor}
        - delay {float}: delay between frames if displaying
    Outputs: 
    	- tot_reward {float}
	- mean_desc {list}
    """
    T = 0
    tot_reward = 0.0
    state, done = env.reset(), False
    while not done:
        action = policy.select_action(np.array(state))
        state, reward, done, _ = env.step(action)
        time.sleep(delay)
        tot_reward += reward
        T += 1
    return tot_reward, env.desc


def video_policy(env_name, policy, video_path, resolution, cam_dist, cam_yaw, cam_pitch):
    """
    Launch visualisation
    Inputs:
        - env_name {str}
	- policy {Actor}
    	- video_path {str}
	- resolution {dic}: render image size
        - cam_dist {float}
        - cam_yaw {float}
        - cam_pitch {float}
    Outputs: 
    	- tot_reward {float}
	- mean_desc {list}
    """
    env = gym.make(env_name, render=True)
    env.env._render_width, env.env._render_height = resolution
    env.env._cam_dist = cam_dist
    env.env._cam_yaw = cam_yaw
    env.env._cam_pitch = cam_pitch
    env = wrappers.Monitor(env, video_path, force=False)

    T = 0
    tot_reward = 0.0
    state, done = env.reset(), False
    while not done:
        action = policy.select_action(np.array(state))
        state, reward, done, _ = env.step(action)
        tot_reward += reward
        T += 1

    desc = env.desc
    env.close()
    return tot_reward, desc
