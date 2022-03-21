This is the code for PGA-MAP-Elites Algorithm. 

### Introduction

Running the code requires installing `singularity`. Instructions for which can be found [here](https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps). The easiest way is to consult the [AIRL WIKI](https://gitlab.doc.ic.ac.uk/AIRL/AIRL_WIKI) which containes detailed instructions for installing `singularity` on different systems. 
To run the code locally first clone the repo:


```shell script
git clone --recurse-submodules git@gitlab.doc.ic.ac.uk:AIRL/research_projects/manon_flageat/pg-map-elites.git
```

#### Running in Sandbox

First build the sandbox image by:

```shell script
cd pg-map-elites/singularity
./start_container.sh
```
This will build an image that containes all the correct dependencies to run the code. Too see the dependencies you can inspect the `singularity.def` file located in `pg-map-elites/singularity`.


When inside the sandbox container the code can be run by:

```shell script
cd /git/sferes2/exp/pg-map-elites
python3 run_experiment.py
```
`run_experiment.py` takes a range of arguments which is easiest to pass as a .txt by using the `--config_file` argument. You can find the config files for all algorithms in the `configure_experiment` folder.
The files ending with `config_local` are meant to be used on local computer to test the code, the one ending with `config` are meant to be used on the HPC. For example:

```shell script
cd /git/sferes2/exp/pg-map-elites
python3 run_experiment.py --config_file configure_experiment/pga_config_local.txt
```

These config files do not define the experiment you want to run, by default it runs the QDAntBulletEnv-v0 environment, to change it to QDWalker2DBulletEnv-v0 for example, you can run:

```shell script
cd /git/sferes2/exp/pg-map-elites
python3 run_experiment.py --config_file configure_experiment/pga_config_local.txt --env QDWalker2DBulletEnv-v0
```

You can find the list of possible environments, and more details about the arguments passed by the config files in `run_experiment.py`.

#### Analysis

The output of the code will be saved in the location specified by the `--save_path` argument. 
It contains stats file that can be analysed and plotted using `run_plots.py`. To execute it, use:

```shell script
python3 run_plots.py --save_path [path_to_stats_files] --results_path [path_to_save_graphs] --max_evals [max_evals_of_run] --progress --archive
```

The `--max_evals` input allow to select which archive should be plotted as the final archive. If there is no archive corresponding to this value, no archive plots will be displayed. 
The `--progress` input chose to plot the progress graphs and `--archive` to plot the final archives, you may want to have a look at the other possible analysis listed in `run_plots.py`.

#### Visualisation

You can also visualise the agent in the PyBullet environment by running `run_visualisation.py`:

```shell script
python3 run_visualisation.py --save_path [path_to_stats_files] --results_path [path_to_save_videos]
```

#### Running Final Images

First build the final image by:

```shell script
cd singularity
./build_final_image.sh
```
This will build an image named `final_pg-map-elites_$(date +%Y-%m-%d_%H_%M_%S).sif` that containes all the correct dependencies to run the code.

Running this final container will automatically run the current `run_experiment.py`, followed by the current `run_plots.py`. You can specify additional parameters for `run_experiment.py` by adding them at the end as follow:
You can run the final container with the following command:

```shell script
./final_pg-map-elites_$(date +%Y-%m-%d_%H_%M_%S).sif [other_arguments_you_may_have]
```

All outputs of this run will be saved in a folder `results_pg_map_elites`. Each run will create its own new subfolder: `results_pg_map_elites/%Y-%m-%d_%H_%M_%S`. The analysis script will consider the full content of `results_pg_map_elites`.


#### Running on HPC

When running on the HPC, make sure to add the following lines in your job script to constrain the number of thread used by PyTorch (the evaluations and PG mutation are already spreading the cores through the multiprocessing library):

```shell script
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

If you use the gitlab notebook, use the the `no_containall` branch which already contains these lines.

