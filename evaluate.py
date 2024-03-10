from ray.rllib.algorithms import Algorithm
from ray.tune import register_env

from environment import TheMindEnvironment

register_env("themind", lambda config: TheMindEnvironment(config))
trained = Algorithm.from_checkpoint("C:/Users/thlam/ray_results/PPO_2024-03-10_09-56-50/PPO_themind_abc20_00001_1_intention_size=1_2024-03-10_09-56-54/checkpoint_000000")

test_env = TheMindEnvironment({"hand_size": 10, "intention_size": 1})

obs, _ = test_env.reset()

game_round = 0
while True:
    print(f"---- ROUND {game_round} ----")
    print("---- ENVIRONMENT ----")
    test_env.render()

    print("---- ACTION ----")
    action = trained.compute_single_action(obs[test_env.current_player], policy_id=test_env.current_player)
    print(action)

    obs, rewards, terminated, _, _ = test_env.step(action)

    print("---- REWARDS ----")
    print(rewards)

    if terminated["__all__"]:
        break

    game_round += 1
    print("\n\n")
