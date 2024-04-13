import numpy as np

from environment import TheMindEnvironment

while True:
    env = TheMindEnvironment({"hand_size": 1, "intention_size": 1, 'number_of_players': 5})
    print(env.decks)
    while True:
        intention = np.zeros(1)
        intention.fill(env.current_player_index)
        step = env.step({env.player(env.current_player_index): np.array([1, *intention])})
        print(step)
        if step[2]["__all__"]:
            break

    input()
    print("\n\n")

