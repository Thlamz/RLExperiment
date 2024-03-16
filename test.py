import numpy as np

from environment import TheMindEnvironment

for i in range(20):
    print("Game", i)
    env = TheMindEnvironment({"hand_size": 1, "intention_size": 1, 'number_of_players': 4})
    print(env.decks)
    while True:
        intention = np.zeros(1)
        intention.fill(env.current_player_index)
        step = env.step({env.player(env.current_player_index): {"play": 1, "intention": intention}})
        print(step)
        if step[2]["__all__"]:
            break
    print("\n\n")

