import copy
import time

import numpy as np
import torch


def critic_worker(
    replay_fn,
    critic_fn,
    nr_of_steps_crit,
    nr_of_steps_act,
    random_init,
    transitions_queue,
    remote,
    global_sync,
    close_process,
    train_batch_size=256,
):

    """
    Critic worker - run the critic training process.
    Inputs:
        - replay_fn {partial ReplayBuffer} - function that initialises the replay buffer
        - critic_fn {partial Critic} - function that initialises the critic
        - nr_of_steps_crit {int} - nr of steps to train critic per generation
        - nr_of_steps_act {int} - nr of steps to train actor per generation
        - random_init {int} - number of init steps
        - transitions_queue {Queue} - queue to receive transitions
        - remote {Pipe} - pipe to receive greedy actor
        - global_sync {Event} - event to trigger synch
        - close_process {Event}
        - train_batch_size {int} - batch size for training critic
    """

    # initialisation - .x() is to extract from CloudpickleWrapper
    replay_buffer = replay_fn.x()
    critic = critic_fn.x()
    waiting = False

    # start loop for process
    while True:
        try:

            if close_process.is_set():
                print("Close Critic Process")
                break

            # collect new transitions
            while transitions_queue.qsize() > 0:
                try:
                    idx, transitions = transitions_queue.get_nowait()
                    replay_buffer.add(transitions)
                except BaseException:
                    pass

            # get new greedy actor
            if remote.poll():
                new_actor = remote.recv()
                critic.update_greedy(new_actor)

            # training critic
            if (
                replay_buffer.additions > random_init * 0.9
                and critic.has_greedy()
                and not waiting
            ):
                start = time.time()
                critic_loss = critic.train(
                    replay_buffer, nr_of_steps_crit, train_batch_size=train_batch_size
                )
                train_time = time.time() - start
                waiting = True  # hack

            # synch
            if global_sync.is_set() and waiting:
                # print("Parallel worker received the True flag of global sync")
                out_actor = copy.deepcopy(critic.actor)
                for param in out_actor.parameters():
                    param.requires_grad = False

                # sample states from the replay buffer
                states = replay_buffer.sample_state(train_batch_size, nr_of_steps_act)

                # send the latest critic, the latest actors, sampled states,
                # and the last critic loss/training time
                remote.send(
                    (
                        critic.critic,
                        [out_actor],
                        states,
                        critic_loss.detach(),
                        train_time,
                    )
                )

                global_sync.clear()
                # print("Flag back to false")
                waiting = False  # hack

        except KeyboardInterrupt:
            break


def td3_critic_worker(
    replay_fn,
    critic_fn,
    actor_fn,
    env_fn,
    random_init,
    num_cpu,
    remote,
    global_sync,
    close_process,
    train_batch_size=256,
):

    """
    Critic worker for TD3 algorithm - run the critic training process.
    Inputs:
        - replay_fn {partial ReplayBuffer} - function that initialises the replay buffer
        - critic_fn {partial Critic} - function that initialises the critc
        - actor_fn {partial Actor} - function that initialises the actor
        - env_fn {Partial Env}
        - random_init {int} - number of init steps
        - num_cpu {int} - number of parallel envs used to regulate Pipe exchanges
        - remote {Pipe} - pipe to receive greedy actor
        - global_sync {Event} - event to trigger synch
        - close_process {Event}
        - train_batch_size {int} - batch size for training critic
    """

    # initialisation - .x() is to extract from CloudpickleWrapper
    replay_buffer = replay_fn.x()
    critic = critic_fn.x()
    critic.update_greedy(actor_fn())
    env = env_fn.x()
    out_actors = []
    critic_loss = torch.zeros(1)

    # initialisetion of episode
    t = 0
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start = time.time()

    # start loop (stop when main loop finishes
    while True:
        try:
            # if closing process
            if close_process.is_set():
                print("Close Critic Process")
                remote.send((critic, replay_buffer))
                time.sleep(10)
                break

            episode_timesteps += 1
            t += 1

            # Select action randomly or according to _olicy
            if t < random_init:
                action = env.action_space.sample()
            else:
                action = (
                    critic.actor.select_action(np.array(state))
                    + np.random.normal(
                        0, critic.max_action * critic.expl_noise, size=critic.action_dim
                    )
                ).clip(-critic.max_action, critic.max_action)

            # Perform action
            next_state, reward, done, info = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add((state, action, next_state, reward, done_bool, env.desc))

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if replay_buffer.size >= random_init:
                critic_loss = critic.train(replay_buffer, 1, train_batch_size)

            if done:
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Add current actor to the actors list
            if t % critic.policy_freq == 0:
                out_actor = copy.deepcopy(critic.actor)
                for param in out_actor.parameters():
                    param.requires_grad = False
                out_actors.append(out_actor)

            if len(out_actors) > 3 * num_cpu and not (global_sync.is_set()):
                print("\n[WORKER] Waiting for the evaluation process\n")
                wait_time = time.time()
                while not (global_sync.is_set()):
                    time.sleep(1)
                print("\n[WORKER] Resuming after", time.time() - wait_time, "\n")

            # Send the actors to the main process
            if global_sync.is_set() and len(out_actors) > num_cpu:
                remote.send(
                    (
                        critic.critic,
                        out_actors,
                        None,
                        critic_loss.detach(),
                        time.time() - start,
                    )
                )
                start = time.time()
                out_actors = []
                global_sync.clear()

        except KeyboardInterrupt:
            break
