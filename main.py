from typing import Dict, Union, Optional

from ray import air, tune
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
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
        .environment("themind", env_config={
            "hand_size": 10,
            "intention_size": tune.grid_search([0, 1]),
            "stall_limit": 5
        })
        .resources(num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=2, batch_mode="complete_episodes")
        .evaluation(
            evaluation_interval=10_000,
            evaluation_duration=10,
            evaluation_num_workers=1
        )
        .multi_agent(
            policies={
                "player1": (None, None, None, {}),
                "player2": (None, None, None, {})
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .callbacks(CustomTheMindCallback)
    )

    results = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=TrialPlateauStopper(metric="custom_metric/win_mean")
        ),
        param_space=config
    ).fit()

    print(results)
