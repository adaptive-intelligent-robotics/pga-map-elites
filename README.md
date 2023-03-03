# Empirical analysis of PGA-MAP-Elites for Neuroevolution in Stochastic Domains

This repository contains the code associated with [Empirical analysis of PGA-MAP-Elites for Neuroevolution in Stochastic Domains]().

It proposes an implementation of several algorithms compared within the paper:
+ MAP-Elites algorithm from [Illuminating search spaces by mapping elites](https://arxiv.org/pdf/1504.04909.pdf).
+ TD3 proposed in [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf).
+ PGA-MAP-Elites first introduced in [Policy Gradient Assisted MAP-Elites](https://dl.acm.org/doi/pdf/10.1145/3449639.3459304?casa_token=6iKdMjVJXLoAAAAA:lzuKlS8Gm_-g_AFIBYFA_g790NVOX6Y615n9SoG5zGG9fHQH7xf0RSKE__5a3qsOSswKRM1cErTymg).
+ CMA-MAP-Elites from [Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space](https://dl.acm.org/doi/pdf/10.1145/3377930.3390232?casa_token=84WaWgtEOHwAAAAA:J01FdWPGmWq0Y5iwTIo1QB6nL41JHyNjlPFpZ3f4AwZMGlHVbjJDdFjZAxT_Bfft6IPB3vupERM-9w).
+ Deep-grid-MAP-Elites introduced in [Fast and stable MAP-Elites in noisy domains using deep grids](https://watermark.silverchair.com/isal_a_00316.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsswggLHBgkqhkiG9w0BBwagggK4MIICtAIBADCCAq0GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM3iP7kURpUbDa-XAbAgEQgIICfigb4OIm1zO3sLG0EY96SXx3pQblCAEC2gDQryW9XMLErwaDHaAojOW8PuO3iB4bugnXxTpXS7uCobvKSYBzYPOnF7PEUYwpGiVNpjiRCi6LXOIOtVhbrMdHuwz7zlLKwA_wTH_3QyDN8nuUn3xLLxeM9dTwdehi7Rg3iX83WkI3PS4CR6SeRLwpSn-_nJLz_l2mP8hPlaKJcaGSN7R_ZfRhEg_w64wUHKEG9AuvtZkf7YYMr7c-dmR29TXRSBJw5iUOrYBZkSI1yry1I5MHZVdy9qYJTNqxl3rV7Do__e2RS6mYEEvOSb1UG1OJMjrvsQ50yAOGC7cjk_J91zwXa27KvFU2nRAG6KZ38W1uq3gghRzMmo1wHPBArIaVmN1firRhACJJxKqrdn_yg4xT2eH6FkmG9RIXfba2B5dw2LgsHQ7HwzZoETZ-JFFUdafp5GQHduWfJ2dDCZ7avUIMCR34ELArLISmoL5i-Ygb-Hshkax1qIKZoIiK9tFBpWY7e1P5p07PZoubdpFLbhqZ-OkFd9Jv4kazbCEk0yGZVZhN38xGeJUoSH_VdFVOWeJGN18y-EZOp9DYlOJgbcxtc6PUUJ-cCAHkggIz_H1SG3OHrtTxSvuWNWJqOB4XA8K9sQhEgj_R1ij1jbezSUCbd2PhFeWvbDC2QqwBukWd-aabwDydXaZTucDAoq13bDOtaB6kxmbmetugTyzRwg-WaPVeAUPnq8KHmZpUlzABzPlFk3UgBhdqgmd-g2mnxu6XtNIixbdSX9Fd4fc-GuCq6VsWuYSxQ7ZKCdL1eyljBjE3FCbnAFZ-XYX2Ol8Q2QMDP6chEyvovFVQ1xchF9rs).
+ MAP-Elites-sampling as used in previous works such as [Hierarchical Behavioral Repertoires with Unsupervised Descriptors](https://dl.acm.org/doi/pdf/10.1145/3205455.3205571?casa_token=n8XprBI79jcAAAAA:IwwUqHH9dNCZc9GgbPFc2Xp8Ox5O4CeRoG7J0Xb_YpiyPR57NrlAhNmH1b9kqESzi85IIPaqEMZVpA).

This code compares these approaches on the [QDgym_extended tasks](https://github.com/adaptive-intelligent-robotics/QDgym_extended) based on the [QDgym tasks](https://github.com/ollenilsson19/QDgym).
It is based on the original repository of [PGA-MAP-Elites](https://github.com/ollenilsson19/PGA-MAP-Elites), merging the repository of [TD3](https://github.com/sfujim/TD3) with the [PyMAP-Elites](https://github.com/resibots/pymap_elites) implementation of MAP-Elites.
We provide the data used to generate the paper results in zip files in `2022_results`.
We provide an executable precompiled Singularity container [here](https://drive.google.com/file/d/1XsapzTxorN5GVBcd3C2f-yWkn5odhEIv/view?usp=sharing).


# Libraries and dependencies

The implementation of all tasks and algorithms is in Python 3.8.
It requires the standard Python 3.8 libraries, [Numpy](https://numpy.org/), [Scikit-learn](https://pypi.org/project/scikit-learn/) for the implementation of the CVT archives, and [PyRibs](https://pyribs.org/) for CMA-MAP-Elites.
All algorithms learn Deep Neural Network controllers, requiring the [PyTorch](https://pytorch.org/) library, and all the tasks are implemented in the [QDgym_extended library](https://github.com/adaptive-intelligent-robotics/QDgym_extended) that relies on [Gym](https://gym.openai.com/) and [Pybullet](https://pybullet.org/wordpress/).
Furthermore, the analysis of the results is based on [Panda](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/index.html) libraries.

We also propose a containerised version of our environment to replicate our experiment and results in [Singularity](https://singularity-docs.readthedocs.io). We provide an executable precompiled Singularity container for this repository [here](https://drive.google.com/file/d/1XsapzTxorN5GVBcd3C2f-yWkn5odhEIv/view?usp=sharing).


# Structure

+ `run_experiment.py` is the main file, used to run any experiments.
+ `run_plots.py` allows for analysing the results of a run and generating the corresponding graphs.
+ `configure_experiment` contains one config file to run each algorithm with its default parameters.
+ `CVT` saves the CVT archives to avoid recomputing them at each run.
+ `src` contains the definition of all algorithms based on the structure of [PyMAP-Elites](https://github.com/resibots/pymap_elites).
	+ `initialisation.py` is called first to initialise the chosen algorithm.
	+ `mapelites.py` runs the main loop of the algorithm.
	+ `mapping` contains all the classes and functions needed to handle the main component of MAP-Elites: the archive.
	+ `learning` contains all the classes for Deep Neural Networks objects: the controllers for all algorithms and the critics and replay-buffers for Deep Reinforcement Learning approaches.
	+ `process` defines the processes parallel to the main loop: the critic training and the parallelisation of the evaluations.
	+ `variation` allows defining the MAP-Elites mutations and the policy-gradient mutations.
	+ `metrics` contains the definition of all metrics used in the paper.
	+ `analysis` defines all the functions needed to analyse the data output by a run.
+ `singularity` contains all files needed to compile the Singularity container for this experiment. You can find a precompiled container for this repository [here](https://drive.google.com/file/d/1XsapzTxorN5GVBcd3C2f-yWkn5odhEIv/view?usp=sharing).
+ `2022_results` contains the data used to generate the paper results.


# Execution from source

## Dependencies to run form source

Running this code from source requires Python 3.8, and the libraries given in `requirements.txt` (Warning: preferably use a virtual environment for this specific project, to avoid breaking the dependencies of your other projects).
In Ubuntu, installing the dependencies can be done using the following command:

```shell script
pip install -r requirements.txt
```

### Troubleshooting 
Depending on your setup, you might get an error when installing scikit-learn from the `requirements.txt`, this is due to pip trying to install the package in the wrong order. You can enforce it by first running:

```shell script
pip3 install numpy==1.19.5 Cython==0.29.21 GPUtil==1.4.0 psutil==5.9.0
```

You can then run the `requirements.txt` installation safely.

## Run from source

The results from the experiments can be reproduced by running `run_experiment.py`.
This script takes a range of arguments and the `configure_experiment` folder contains `.txt` configuration files with default arguemnt values for each algorithms. Configuration files can be passed using the `--config_file` argument.
For example, to run PGA-MAP-Elites with the parameters from the paper:
```shell script
python3 run_experiment.py --config_file configure_experiment/pga_config.txt
```

By default, this would run the QDWalker2DBulletEnv-v0 environment. To change it, for example, to QDAntBulletEnv-v0, use the `--env` argument. To run a short-time test, for example, for 1000 evaluations instead of 1000000, use the `--max_evals` argument:
```shell script
python3 run_experiment.py --config_file configure_experiment/pga_config.txt --env QDAntBulletEnv-v0 --max_evals 1000
```
The output of the run will be saved in the location specified by the `--save_path` argument. Other arguments are detailed in `run_experiment.py`.
Be careful that the seed is fixed for a given run and needs to be passed as an argument using `--seed`, the results of the paper used a random different seed for each run.

## Analysis from source


The output of a run contains `.csv`, `.dat` and `.pickle` files that can be analysed and plotted using `run_plots.py`. To execute it, use:

```shell script
python3 run_plots.py --save_path *path_to_stats_files* --results_path *path_to_save_graphs* --max_evals *max_evals_of_run*
```

`--save_path`specifies where the path to the files output by the run, `--results_path` specifies where to save the resulting graphs and `--max_evals` allows selecting which archive should be plotted as the final archive. If there is no archive corresponding to this value, no archive plots will be displayed.
Other arguments are listed in `run_plots.py`.
We provide the data used to generate the paper results in zip files in `2022_results`.

`run_plots.py` also allows visualising the policies in the PyBullet environment using the `--visualisation` argument.


# Execution from the final Singularity container image

## Run with an existing container

The results of the paper can be reproduced by running the Singularity container of the experiment, for example, the precompiled container provided [here](https://drive.google.com/file/d/1XsapzTxorN5GVBcd3C2f-yWkn5odhEIv/view?usp=sharing).
A container image can be executed directly: `./*image_name*`. This will re-generate the results for PGA-MAP-Elites on the QDWalker task with the parameters from the paper.
Arguments, as detailed in previous sections, can be specified when executing the container. For example, to run PGA-MAP-Elites for 1000 evaluations on the QDAnt task using the provided image, one can execute the command:
```shell script
./pga_map_elites.sif --env QDAntBulletEnv-v0 --max_evals 1000
```

Unless otherwise specified, the results of the execution are solved in a `results_pga-map-elites` folder, outside of the image. Executing the container will also run the analysis of these results and generate the corresponding graphs as `.svg`and `.png`files.

## Analysis with an existing container

The Singularity container also provides a separate`Analysis` app to run the analysis only. It can be called using:
```shell script
singularity run --app Analysis *image_name* *results_folder_name* *additional_analysis_parameters*
```

## Compiling a new Singularity container

The `singularity` folder provides all the building blocks to recompile a new Singularity container:
+ `singularity.def` contains the "recipe" to build the container: it starts from a Ubuntu container, updates a existing libraries and installs the package in `requirements.txt`.
+ `start_container` allows to build a sandbox container, for development. It can be executed with `./start_container`.
+ `build_final_image` recompiles a final container similar to the one provided [here](https://drive.google.com/file/d/1XsapzTxorN5GVBcd3C2f-yWkn5odhEIv/view?usp=sharing). It can be executed with `./build_final_image`.
