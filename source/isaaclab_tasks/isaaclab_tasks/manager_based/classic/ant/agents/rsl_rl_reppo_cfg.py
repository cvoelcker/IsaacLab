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
class AntREPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 100000
    save_interval = 50
    experiment_name = "ant"
    policy = RslRlReppoActorQCfg(
        init_noise_std=1.0,
        init_alpha_kl=0.01,
        init_alpha_temp=0.01,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512, 512],
        activation="swish",
        vmin=-15.0,
        vmax=15.0,
    )
    algorithm = RslRlReppoAlgorithmCfg(
        num_learning_epochs=8,
        num_mini_batches=64,
        learning_rate=3.0e-4,
        gamma=0.99,
        lam=0.9,
        desired_kl=0.1,
        max_grad_norm=10.0,
        target_entropy=-0.5,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
