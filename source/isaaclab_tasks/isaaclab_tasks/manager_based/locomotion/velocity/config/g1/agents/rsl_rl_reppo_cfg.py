# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlReppoActorQCfg,
    RslRlReppoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
)


@configclass
class HumanoidREPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 100000
    experiment_name = "humanoid"
    policy = RslRlReppoActorQCfg(
        init_noise_std=0.1,
        init_alpha_kl=0.01,
        init_alpha_temp=0.0001,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 512],
        critic_hidden_dims=[1024, 1024, 1024],
        activation="elu",
        noise_std_type="sigmoid",
        vmin=-20.0,
        vmax=20.0,
    )
    algorithm = RslRlReppoAlgorithmCfg(
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.1,
        max_grad_norm=1.0,
        target_entropy=-0.5,
        rnd_cfg=None,
        symmetry_cfg=None,
        scale_actions=True,
        action_upper_bound=1.0,
        action_lower_bound=-1.0,
    )
