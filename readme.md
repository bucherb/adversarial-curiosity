# Adversarial Curiosity

Code for reproducing simulation experiments in our work [An Adversarial Objective for Scalable Exploration](https://arxiv.org/abs/2003.06082). Our [project page](https://sites.google.com/view/action-for-better-prediction) provides data and information regarding our robotics experiments.

Please cite our paper if you use our research or code in your work.

```
@misc{bucher2020adversarial,
    title={An Adversarial Objective for Scalable Exploration},
    author={Bernadette Bucher and Karl Schmeckpeper and Nikolai Matni and Kostas Daniilidis},
    year={2020},
    eprint={2003.06082},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

### Software Dependencies

We provide a Docker image with the required dependencies other than Mujoco to run our code. To build the Docker image and push to your Dockerhub account, run

```
./docker_build [dockerhub_username]
```

The OpenAI Half Cheetah simulation in which we execute our experimental evaluation requires Mujoco to run. For instructions on acquiring and installing a Mujoco license, see the [Mujoco website](http://www.mujoco.org/).

After Mujoco is properly installed, mujoco_py, the Python interface to Mujoco, needs to be installed.

```
pip3 install mujoco_py
```

### Reproducing Half Cheetah Experiments
Execute the commands listed below from the code directory to reproduce the results we report with our method as well as each of the baseline methods against which we compare.

* Adversarial Curiosity
```
python3 main.py with max_explore utility_measure=discrim env_noise_stdev=0.02 n_warm_up_steps=1024 m_loss_weight=1.0 a_loss_weight=1.0 utility_scale=10.0 n_layers=8
```

* MAX:
```
python3 main.py with max_explore env_noise_stdev=0.02
```

* Trajectory Variance Active Exploration (TVAX):
```
python3 main.py with max_explore utility_measure=traj_stdev policy_explore_alpha=0.2 env_noise_stdev=0.02
```

* Renyi Divergence Reactive Exploration (JDRX):
```
python3 main.py with max_explore exploration_mode=reactive env_noise_stdev=0.02
```

* Prediction Error Reactive Exploration (PERX):
```
python3 main.py with max_explore exploration_mode=reactive utility_measure=pred_err policy_explore_alpha=0.2 env_noise_stdev=0.02
```

* Random Exploration:
```
python3 main.py with random_explore env_noise_stdev=0.02
```

### Reproducing the Ant Experiments
Execute the commands listed below to reproduce the results we report for our method and the baseline methods.

You can repeat these commands with different ensemble sizes by changing the value of ENSEMBLE_SIZE.

* Adversarial Curiosity
```
python3 main.py with max_explore log_dir='/PATH/TO/LOGS' experiment_name=EXP_NAME env_noise_stdev=0.02 utility_measure=discrim m_loss_weight=1.0 a_loss_weight=1.0 utility_scale=30 threshold=0.75 env_name=MagellanAnt-v2 n_warm_up_steps=1024 ant_coverage=True ensemble_size=ENSEMBLE_SIZE n_layers=8 n_exploration_steps=10100
```

* MAX
```
python3 main.py with max_explore log_dir='/PATH/TO/LOGS' experiment_name=EXP_NAME env_noise_stdev=0.02 env_name=MagellanAnt-v2 n_warm_up_steps=1024 ant_coverage=True ensemble_size=ENSEMBLE_SIZE n_layers=8 n_exploration_steps=10100
```

### Acknowlegdements

The authors are grateful for support through the Curious Minded Machines project funded by the Honda Research Institute.

This repository was built off of a fork from [Model-Based Active Exploration (MAX) repository](https://github.com/nnaisense/max) from which we run baselines for comparison against our method.
