from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from gymnasium.vector.utils import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


TheMindObservation = Dict[str, Dict[str, Any]]


class TheMindEnvironment(MultiAgentEnv):
    decks: Dict[str, List[int]]
    current_player: str
    table: int
    pile_size: int
    stall_count: int
    finished: bool
    won: bool
    termination: Dict[str, bool]

    def __init__(self, config: dict):
        super().__init__()

        self.hand_size = config.get("hand_size", 5)
        self.intention_size = config.get("intention_size", 0)
        self.stall_limit = config.get("stall_limit", 2)

        self._agent_ids = ["player1", "player2"]

        # 101 because hand is set to 100 when the other player has no cards

        table_space = spaces.Box(0, 1, shape=(1,))
        hand_space = spaces.Box(0, 1, shape=(2,))
        intention_space = spaces.Box(0, 1, shape=(self.intention_size,))

        self.observation_space = spaces.Dict({
            "player1": spaces.Dict({"table": table_space,
                                    "hand": hand_space,
                                    "intention": intention_space}),
            "player2": spaces.Dict({"table": table_space,
                                    "hand": hand_space,
                                    "intention": intention_space}),
        })
        self.action_space = spaces.Dict({
            "player1": spaces.Dict({"play": spaces.Box(0, 1, shape=(1,)),
                                    "intention": spaces.Box(0, 1, shape=(self.intention_size,))}),
            "player2": spaces.Dict({"play": spaces.Box(0, 1, shape=(1,)),
                                    "intention": spaces.Box(0, 1, shape=(self.intention_size,))}),
        })

        self.reset_variables()

    def reset_variables(self):
        self.decks: Dict[str, List[int]] = {}
        self.draw_decks()
        self.current_player = "player1"
        self.table = 0
        self.pile_size = 0
        self.stall_count = 0
        self.finished = False
        self.won = False
        self.termination = {
            "player1": False,
            "player2": False,
            "__all__": False
        }

    @property
    def other_player(self):
        return "player1" if self.current_player == "player2" else "player2"

    def draw_decks(self):
        cards_drawn = np.random.choice(100, self.hand_size * 2, replace=False)

        self.decks = {
            "player1": list(sorted(cards_drawn[:self.hand_size])),
            "player2": list(sorted(cards_drawn[self.hand_size:]))
        }

    def get_top_cards(self, player):
        top_card = self.decks[player][0] if len(self.decks[player]) > 0 else 100
        second_card = self.decks[player][1] if len(self.decks[player]) > 1 else 100
        return top_card, second_card

    def get_top_cards_observation(self, player):
        top_card, second_card = self.get_top_cards(player)
        return np.array([top_card/100, second_card/100], dtype=np.float32)

    def get_table_observation(self):
        return np.array([self.table/100], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.reset_variables()

        observation: TheMindObservation = {
            "player1": {
                "table": self.get_table_observation(),
                "hand": self.get_top_cards_observation(self.current_player),
                "intention": np.zeros((self.intention_size,), dtype=np.float32)
            },
        }
        return observation, {self.current_player: {"won": self.won}}

    def compute_if_finished(self):
        if self.finished:
            return

        other_player_top, _ = self.get_top_cards(self.other_player)
        lost = self.table > other_player_top
        if lost:
            self.finished = True
            return

        # If one player has emptied his hand, the game is over
        won = len(self.decks[self.current_player]) == 0 or len(self.decks[self.other_player]) == 0
        if won:
            # Empty both decks and place all cards on the table
            self.table = max([*self.decks[self.current_player], *self.decks[self.other_player]])
            self.pile_size = self.hand_size * 2
            self.decks[self.current_player] = []
            self.decks[self.other_player] = []

            self.won = True
            self.finished = True
            return

        stalled = self.stall_count >= self.stall_limit
        if stalled:
            self.finished = True

    def read_action(self, action_dict) -> Tuple[bool, np.ndarray]:
        # Handling for cases when the action dict is received per agent and when it is received as a single dict
        if self.current_player in action_dict:
            action = action_dict[self.current_player]
        else:
            action = action_dict

        return action["play"] > 0.5, action["intention"]

    def step(self, action_dict):
        play, intention = self.read_action(action_dict)

        hand = self.decks[self.current_player]

        # Applying action
        if not self.finished and play and len(hand) > 0:
            self.table = hand.pop(0)
            self.stall_count = 0
            self.pile_size += 1
        else:
            self.stall_count += 1

        self.compute_if_finished()

        observation = {
            self.other_player: {
                "table": self.get_table_observation(),
                "hand": self.get_top_cards_observation(self.other_player),
                "intention": intention
            }
        }

        # Calculating rewards
        reward = {
            self.current_player: self.pile_size / (self.hand_size * 2) if self.finished else 0
        }

        # Calculating if terminated
        if self.finished:
            self.termination[self.current_player] = True
            self.termination["__all__"] = self.termination[self.current_player] and self.termination[self.other_player]

        # Calculating truncateds
        truncateds = {
            "player1": False,
            "player2": False,
            "__all__": False
        }

        self.current_player = self.other_player
        return observation, reward, self.termination, truncateds, {self.current_player: {"won": self.won}}

    def render(self):
        print("Table:", self.table)
        print("Player1:", self.decks["player1"])
        print("Player2:", self.decks["player2"])
        print("Current player:", self.current_player)