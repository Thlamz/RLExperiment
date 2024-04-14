from typing import Dict, Union, Optional

from ray import air, tune
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms import SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.stopper import TrialPlateauStopper

from environment import TheMindEnvironment

class CustomTheMindCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        if episode.last_info_for("player1").get("won", False) or episode.last_info_for("player2").get("won", False):
            episode.custom_metrics["win"] = 1
        else:
            episode.custom_metrics["win"] = 0


if __name__ == "__main__":
    register_env("themind", lambda config: TheMindEnvironment(config))

    config = (
        PPOConfig()
        .training()
        .resources(num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=7, rollout_fragment_length='auto',
                  num_envs_per_worker=8, batch_mode="complete_episodes")
        .environment("themind", env_config={
            "hand_size": 10,
            "intention_size": tune.grid_search([0, 1]),
            "number_of_players": 3 ,
            "stall_limit": 5
        })
        .multi_agent(
            policies={
                "p0"
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: "p0"),
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={"p0": SingleAgentRLModuleSpec()}
            )
        )
        .callbacks(CustomTheMindCallback)
    )

    results = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"episodes_total": 60_000},
        ),
        param_space=config
    ).fit()

    print(results)
