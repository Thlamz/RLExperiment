from environment import TheMindEnvironment

for i in range(20):
    print("Game", i)
    env = TheMindEnvironment({"hand_size": 2, "intention_size": 1})
    print(env.decks)
    while True:
        step = env.step({"player1": {"play": 1, "intention": 0}, "player2": {"play": 1, "intention": 0}})
        print(step)
        if step[2]["__all__"]:
            break
    print("\n\n")

