from __future__ import absolute_import, division, print_function

import argparse
import os

import warnings ; warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import os
import random
import tempfile
from argparse import RawTextHelpFormatter

import gfootball.env as football_env
import gym
import numpy as np
import ray
import torch
from gfootball import env as fe
from gym import wrappers
from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from rldm.utils import football_tools as ft
from rldm.utils import gif_tools as gt
from rldm.utils import system_tools as st
from rldm.scripts import group_football_tools as gft

EXAMPLE_USAGE = """
Example usage:

    Train one sample of independent policies
        python -m rldm.scripts.train_group_agents

    Train four samples of independent policies sequentially
        python -m rldm.scripts.train_group_agents -n 4

    Train four samples of independent policies, two at a time
        python -m rldm.scripts.train_group_agents -n 4 -m 2

    Train four samples of independent policies, all at once
        python -m rldm.scripts.train_group_agents -n 4 -o

    Train 50 samples of independent policies, using an ASHA scheduler
        python -m rldm.scripts.train_group_agents -n 50 -a

    Train one sample of independent policies, enable debug mode
        python -m rldm.scripts.train_group_agents -d

    To play around with the arguments and not start any training runs:
        python -m rldm.scripts.train_group_agents -r # add parameters
        For example:
            python -m rldm.scripts.train_group_agents -r -n 4 -o -d
            python -m rldm.scripts.train_group_agents -r -o -d -g 0

Checkpoints provided were trained with:

    python -m rldm.scripts.train_group_agents -n 100 -m 4 -t 50000000 -a -e -b
        This schedules 100 samples, 4 at a time, each up to 50M steps,
        but using the ASHA scheduler, and with callbacks collecting metrics

"""


def main(n_cpus, n_gpus, env_name,
         n_policies, n_timesteps, n_samples,
         sample_cpu, sample_gpu, use_scheduler,
         use_tune_config, use_callbacks, debug):
    ray.init(num_cpus=n_cpus, num_gpus=n_gpus, local_mode=debug)

    register_env(env_name, lambda _: gft.create_RllibGFootball_group_env(env_name))

    config = {
        'env': env_name,
        'framework': 'torch',
        'lr': 0.00022602718266055705,
        'gamma': 0.99,
        'train_batch_size': 512,
        'rollout_fragment_length': 16,
        'num_workers': sample_cpu-1,  # one goes to the trainer
        'num_envs_per_worker': 1,
        'num_gpus': sample_gpu,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'log_level': 'INFO' if not debug else 'DEBUG',
        'ignore_worker_failures': False,
        'horizon': 500,
        'model': {
            'lstm_cell_size': 256,
            'lstm_use_prev_action': 'true', 
            'lstm_use_prev_reward': 'true', 
            'max_seq_len': 13,
        }, 
    }
    if use_tune_config:
        tune_config = {
            'lr': tune.uniform(1e-4, 1e-6),
            'gamma': tune.uniform(0.9, 0.999),
        }
        config = {**config, **tune_config}
    if use_callbacks:
        config['callbacks'] = ft.FootballCallbacks

    PBTscheduler = PopulationBasedTraining(
        time_attr='timesteps_total',
        metric='episode_reward_mean',
        mode='max',
        perturbation_interval=50000, 
        hyperparam_mutations={
            'lr': tune.uniform(0.00005, 0.0005),
            'gamma': tune.uniform(0.99, 0.999),
            'train_batch_size': tune.randint(200, 1000),
            'model': {
                'lstm_cell_size': tune.choice([64, 128, 256]),
                'max_seq_len': tune.choice([10, 20, 30])
            }
        }
    )

    stop = {
        "timesteps_total": n_timesteps,
    }
    # if use_scheduler:
    #     scheduler = ASHAScheduler(
    #         time_attr='timesteps_total',
    #         metric='episode_reward_mean',
    #         mode='max',
    #         max_t=n_timesteps,
    #         grace_period=int(n_timesteps*0.10),
    #         reduction_factor=3,
    #         brackets=1)
    #     stop = None

    filename_stem = os.path.basename(__file__).split(".")[0]
    policy_type = 'search' if use_tune_config else 'shared'
    scheduler_type = 'asha' if use_scheduler else 'fifo'
    config_type = 'tune' if use_tune_config else 'fixed'
    experiment_name =f"{filename_stem}_{env_name}_{policy_type}_{n_timesteps}_{scheduler_type}_{config_type}"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_dir = os.path.join(script_dir, '..', '..', 'logs')
    a = tune.run(
        'QMIX', 
        name=experiment_name,
        reuse_actors=False,
        scheduler=PBTscheduler,
        raise_on_failed_trial=True,
        fail_fast=True,
        max_failures=0,
        num_samples=n_samples,
        stop=stop,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir=local_dir,
        config=config,
        verbose=1 if not debug else 3
    )

    checkpoint_path = a.get_best_checkpoint(a.get_best_trial("episode_reward_mean", "max"), "episode_reward_mean", "max")
    print('Best checkpoint found:', checkpoint_path)

    # Writing to a file
    with open('best_checkpoint.txt', 'w') as f:
        f.write('Best checkpoint found: ' + checkpoint_path)

    ray.shutdown()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Script for training RLDM's P3 baseline agents",
        formatter_class=RawTextHelpFormatter,
        epilog=EXAMPLE_USAGE)

    parser.add_argument('-c','--num-cpus', default=None, type=int,
                        help='Number of cpus to allocate for experiment.\
                            \nDefault: use all cpus on system.')
    parser.add_argument('-g','--num-gpus', default=None, type=int,
                        help='Number of gpus to allocate for experiment.\
                            \nDefault: use all gpus on system.')
    parser.add_argument('-n','--num-samples', default=1, type=int,
                        help='Number of training samples to run.\
                            \nDefault: 1 sample.')
    parser.add_argument('-o','--at-once', default=False, action='store_true',
                        help='Whether to run all samples at once.\
                            \nDefault: sequential trials.')
    parser.add_argument('-m','--simultaneous-samples', default=None, type=int,
                        help='Number of samples to run simultaneously.\
                            \nDefault: sequential trials.')
    parser.add_argument('-t','--num-timesteps', default=None, type=int,
                        help="Number of environment timesteps to train each sample.\
                            \nDefault: 5_000_000 steps per number of players,\
                            \nand an additional 25%% of the total for independent policies.")
    parser.add_argument('-a','--scheduler', default=False, action='store_true',
                        help='Use an ASHA scheduler to run only promising trials.\
                            \nDefault: ASHA scheduler disabled.')
    parser.add_argument('-b','--callbacks', default=False, action='store_true',
                        help='Enable callbacks to display metrics on TensorBoard.\
                            \nDefault: Callbacks disabled.')
    parser.add_argument('-e','--tune', default=False, action='store_true',
                        help='Use tune to search for best hyperparameter combination.\
                            \nDefault: Fixed hyperparameters.')
    parser.add_argument('-r','--dry-run', default=False, action='store_true',
                        help='Print the training plan, and exit.\
                            \nDefault: normal mode.')
    parser.add_argument('-d','--debug', default=False, action='store_true',
                        help='Set full script to debug.\
                            \nDefault: "INFO" output mode.')
    args = parser.parse_args()

    assert args.num_samples <= 100, "Can only train up-to 100 samples on a single run"
    assert args.num_samples >= 1, "Must train at least 1 sample"

    assert not args.at_once or args.num_samples <= 10, "Can only train up-to 10 samples with --at-once"
    assert not args.at_once or args.simultaneous_samples is None, "Use either --at-once or --simultaneous-samples"

    if args.simultaneous_samples:
        assert args.simultaneous_samples <= args.num_samples, "Cannot run more simultaneous than total samples"
        assert args.num_samples > 1, "Must train at least 2 samples if selecting simultaneous"

    assert not args.at_once or args.num_samples <= 4, "Cannot run all samples at once with more than 4 samples"
    assert not args.scheduler or args.num_samples > 2, "Scheduler only makes sense with more than 2 samples"

    n_cpus, n_gpus = st.get_cpu_gpu_count()
    assert args.num_cpus is None or args.num_cpus <= n_cpus, "Didn't find enough cpus"
    assert args.num_gpus is None or args.num_gpus <= n_gpus, "Didn't find enough gpus"
    n_cpus = n_cpus if args.num_cpus is None else args.num_cpus
    n_gpus = n_gpus if args.num_gpus is None else args.num_gpus

    num_players = 3 # hard-coding 3 here for now
    env_name = ft.n_players_to_env_name(num_players, True) # hard-coding auto GK
    n_policies = 1

    n_timesteps = args.num_timesteps if args.num_timesteps else num_players * 5_000_000

    sample_cpu = n_cpus if not args.at_once else n_cpus // args.num_samples
    assert sample_cpu >= 1, "Each sample needs at least one cpu to run"
    sample_gpu = 0 if not n_gpus else 1 if not args.at_once or args.num_samples == 1 else n_gpus / args.num_samples

    if args.simultaneous_samples:
        sample_cpu = n_cpus // args.simultaneous_samples
        assert sample_cpu >= 1, "Each sample needs at least one cpu to run"
        sample_gpu = 0 if not n_gpus else n_gpus / args.simultaneous_samples

    print("")
    print("Calling training code with following parameters (arguments affecting each):")
    print("\tEnvironment to load with GFootball:", env_name)
    print("\tNumber of cpus to allocate for RLlib (-c):", n_cpus)
    print("\tNumber of gpus to allocate for RLlib (-g):", n_gpus)
    print("\tNumber of environment timesteps to train (-t):", n_timesteps)
    print("\tNumber of samples to run (-n):", args.num_samples)
    print("\tNumber of simultaneous samples to run (-n, -m):", args.simultaneous_samples)
    print("\tNumber of cpus to allocate for each sample: (-c, -n, -o, -m)", sample_cpu)
    print("\tNumber of gpus to allocate for each sample: (-g, -n, -o, -m)", sample_gpu)
    print("\tActive scheduler (-a):", args.scheduler)
    print("\tActive callbacks (-b):", args.callbacks)
    print("\tTune hyperparameters (-e):", args.tune)
    print("\tIs this a dry-run only (-r):", args.dry_run)
    print("\tScript running on debug mode (-d):", args.debug)
    print("")

    if not args.dry_run:
        main(n_cpus, n_gpus, env_name,
             n_policies, n_timesteps, args.num_samples,
             sample_cpu, sample_gpu, args.scheduler,
             args.tune, args.callbacks, args.debug)
