
# Empirical analysis of PGA-MAP-Elites for Neuroevolution in Stochastic Domains

This repository contains the code associated with [Empirical analysis of PGA-MAP-Elites for Neuroevolution in Stochastic Domains](https://www.imperial.ac.uk/adaptive-intelligent-robotics).

This code proposes an implementation of several algorithms: 
+ MAP-Elites algorithm from [Illuminating search spaces by mapping elites](https://arxiv.org/pdf/1504.04909.pdf).
+ TD3 proposed in [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf).
+ PGA-MAP-Elites first introduced in [Policy Gradient Assisted MAP-Elites](https://dl.acm.org/doi/pdf/10.1145/3449639.3459304?casa_token=6iKdMjVJXLoAAAAA:lzuKlS8Gm_-g_AFIBYFA_g790NVOX6Y615n9SoG5zGG9fHQH7xf0RSKE__5a3qsOSswKRM1cErTymg).
+ CMA-MAP-Elites from [Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space](https://dl.acm.org/doi/pdf/10.1145/3377930.3390232?casa_token=84WaWgtEOHwAAAAA:J01FdWPGmWq0Y5iwTIo1QB6nL41JHyNjlPFpZ3f4AwZMGlHVbjJDdFjZAxT_Bfft6IPB3vupERM-9w).
+ Deep-grid-MAP-Elites introduced in [Fast and stable MAP-Elites in noisy domains using deep grids](https://watermark.silverchair.com/isal_a_00316.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsswggLHBgkqhkiG9w0BBwagggK4MIICtAIBADCCAq0GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM3iP7kURpUbDa-XAbAgEQgIICfigb4OIm1zO3sLG0EY96SXx3pQblCAEC2gDQryW9XMLErwaDHaAojOW8PuO3iB4bugnXxTpXS7uCobvKSYBzYPOnF7PEUYwpGiVNpjiRCi6LXOIOtVhbrMdHuwz7zlLKwA_wTH_3QyDN8nuUn3xLLxeM9dTwdehi7Rg3iX83WkI3PS4CR6SeRLwpSn-_nJLz_l2mP8hPlaKJcaGSN7R_ZfRhEg_w64wUHKEG9AuvtZkf7YYMr7c-dmR29TXRSBJw5iUOrYBZkSI1yry1I5MHZVdy9qYJTNqxl3rV7Do__e2RS6mYEEvOSb1UG1OJMjrvsQ50yAOGC7cjk_J91zwXa27KvFU2nRAG6KZ38W1uq3gghRzMmo1wHPBArIaVmN1firRhACJJxKqrdn_yg4xT2eH6FkmG9RIXfba2B5dw2LgsHQ7HwzZoETZ-JFFUdafp5GQHduWfJ2dDCZ7avUIMCR34ELArLISmoL5i-Ygb-Hshkax1qIKZoIiK9tFBpWY7e1P5p07PZoubdpFLbhqZ-OkFd9Jv4kazbCEk0yGZVZhN38xGeJUoSH_VdFVOWeJGN18y-EZOp9DYlOJgbcxtc6PUUJ-cCAHkggIz_H1SG3OHrtTxSvuWNWJqOB4XA8K9sQhEgj_R1ij1jbezSUCbd2PhFeWvbDC2QqwBukWd-aabwDydXaZTucDAoq13bDOtaB6kxmbmetugTyzRwg-WaPVeAUPnq8KHmZpUlzABzPlFk3UgBhdqgmd-g2mnxu6XtNIixbdSX9Fd4fc-GuCq6VsWuYSxQ7ZKCdL1eyljBjE3FCbnAFZ-XYX2Ol8Q2QMDP6chEyvovFVQ1xchF9rs).
+ MAP-Elites-sampling as used in previous works such as [Hierarchical Behavioral Repertoires with Unsupervised Descriptors](https://dl.acm.org/doi/pdf/10.1145/3205455.3205571?casa_token=n8XprBI79jcAAAAA:IwwUqHH9dNCZc9GgbPFc2Xp8Ox5O4CeRoG7J0Xb_YpiyPR57NrlAhNmH1b9kqESzi85IIPaqEMZVpA).

It allows comparing these approaches on the [QDgym_extended tasks](https://github.com/adaptive-intelligent-robotics/QDgym_extended) based on the [QDgym tasks](https://github.com/ollenilsson19/QDgym).
It is based on the original repository of [PGA-MAP-Elites](https://github.com/ollenilsson19/PGA-MAP-Elites), which merged the original repository of [TD3](https://github.com/sfujim/TD3) with the [PyMAP-Elites](https://github.com/resibots/pymap_elites) implementation of MAP-Elites.


# Libraries and dependencies

The implementation of all tasks and algorithms is done in Python, it requires the standard Python libraries as well as [Numpy](https://numpy.org/).
All the tasks used in the experiments are implemented in the [QDgym_extended library](https://github.com/adaptive-intelligent-robotics/QDgym_extended) based on the [QDgym library](https://github.com/ollenilsson19/QDgym) introduced in [Policy Gradient Assisted MAP-Elites](https://dl.acm.org/doi/pdf/10.1145/3449639.3459304?casa_token=6iKdMjVJXLoAAAAA:lzuKlS8Gm_-g_AFIBYFA_g790NVOX6Y615n9SoG5zGG9fHQH7xf0RSKE__5a3qsOSswKRM1cErTymg). They rely on [Gym](https://gym.openai.com/) and [Pybullet](https://pybullet.org/wordpress/).
The algorithms require [Scikit-learn](https://pypi.org/project/scikit-learn/) for the implementation of the CVT archives, and CMA-MAP-Elites is based on the implementation in [PyRibs](https://pyribs.org/) developed by its authors. 
All algorithms are used to learn Deep Neural Network controllers requiring the [PyTorch](https://pytorch.org/) library.
Furthermore, the analysis of the results is based on [Panda](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/index.html) libraries.
We also propose a containerised version of our environment to replicate our experiment and results in [Singularity](https://singularity-docs.readthedocs.io).


# Structure

The main part of the code contains the following files and folders:
+ `run_experiment.py` is the main file, used to run any experiments.
+ `run_plots.py` allows to analyse the results of a run and generate the corresponding graphs.
+ `configure_experiment` contains one config file to run each algorithm in this repository with its default parameters.
+ `CVT` saves the CVT archives to avoid recomputing them at each run.
+ `mapgames` contains the main structure of all algorithms based on the structure of [PyMAP-Elites](https://github.com/resibots/pymap_elites). 
	+ `initialisation.py` is called first to initialise the chosen algorithm.
	+ `mapelites.py` runs the main loop of the algorithm.
	+ `mapping` contains all the classes and functions needed to handle the main component of MAP-Elites: the archive.
	+ `learning` contains all the classes for Deep Neural Networks objects: the controllers for all algorithms and the critics and replay-buffers for Deep Reinforcement Learning approaches.
	+ `process` defines the processes parallel to the main loop: the critic training and the parallelisation of the evaluations.
	+ `variation` allows defining the MAP-Elites mutations and the policy-gradient mutations.
	+ `metrics` contains the definition of all metrics used in the paper.
	+ `analysis` allows analysing a finalised run.
+ `singularity` contains all files needed to compile the Singularity container for this experiment.


# Execution

## Run

The results from the experiments can be reproduced by running `run_experiment.py`. 
This script takes a range of arguments that are easiest to pass as a .txt by using the `--config_file` argument. You can find the config files for all algorithms in the `configure_experiment` folder. For example to run PGA-MAP-Elites with the parameters from the paper:
```shell script
python3 run_experiment.py --config_file configure_experiment/pga_config.txt
```

These config files do not define the experiment you want to run, by default it runs the QDAntBulletEnv-v0 environment, to change it to QDWalker2DBulletEnv-v0 for example, you can run:
```shell script
python3 run_experiment.py --config_file configure_experiment/pga_config.txt --env QDWalker2DBulletEnv-v0
```

If you want to run a short-time test for a smaller number of evaluations, for example 100000 evaluations instead of 1000000, you can use the following command:
```shell script
python3 run_experiment.py --config_file configure_experiment/pga_config.txt --max_evals 100000
```

You can find the list of possible environments and more details about the arguments passed by the config files in `run_experiment.py`.
Be careful that the seed is fixed for a given run and needs to be passed as an argument using `--seed`. For the results presented in the paper, we used a different seed for each run of each algorithm. 

## Analysis

The output of the code will be saved in the location specified by the `--save_path` argument.
It contains stats files that can be analysed and plotted using `run_plots.py`. To execute it, use:

```shell script
python3 run_plots.py --save_path [path_to_stats_files] --results_path *path_to_save_graphs* --max_evals *max_evals_of_run* --progress --archive
```

The `--max_evals` input allows selecting which archive should be plotted as the final archive. If there is no archive corresponding to this value, no archive plots will be displayed.
The `--progress` input chose to plot the progress graphs and `--archive` to plot the final archives, you may want to have a look at the other possible analysis listed in `run_plots.py`.


# Execution from the final Singularity container image

The results of the paper can be reproduced by running the Singularity container image of the experiment. 
Executing the container image directly with the following command: `./*image_name*` will re-generate the results for PGA-MAP-Elites on the QDAnt task with the parameters from the paper.
Other parameters can be specified as followed: `./*image_name* *additional_parameters*`, these parameters can be alternative algorithms or environments.
Unless otherwise specified, the results of the execution are solved in a `results_pga-map-elites` folder, outside of the image, at the same location.

An additional app named Analysis allows running the analysis of a given folder only. 
It can be run with the following command: `singularity run --app Analysis *image_name* *results_folder_name* *additional_analysis_parameters*`

This container can be recompiled using the files in the singularity folder: `start_container` to build a sandbox container and `build_final_image` to build the final container image.
