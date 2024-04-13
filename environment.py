from itertools import chain
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from gymnasium.vector.utils import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

TheMindObservation = Dict[str, Dict[str, Any]]


class TheMindEnvironment(MultiAgentEnv):
    decks: Dict[str, List[int]]
    intentions: np.array
    current_player_index: int
    table: int
    pile_size: int
    stall_count: int
    finished: bool
    won: bool
    termination: Dict[str, bool]

    def __init__(self, config: dict):
        super().__init__()

        self.number_of_players = config.get("number_of_players", 2)
        self.hand_size = config.get("hand_size", 5)
        self.intention_size = config.get("intention_size", 0)
        self.stall_limit = config.get("stall_limit", 2) * self.number_of_players

        self._agent_ids = list(self.all_players())

        observation_dict = {
                "table": spaces.Box(0, 1, shape=(1,)),
                "hand": spaces.Box(0, 1, shape=(2,)),
        }
        if self.intention_size > 0:
            observation_dict["intention"] = spaces.Box(0, 1, shape=(self.intention_size, self.number_of_players - 1))

        self.observation_space = spaces.Dict({
            player: spaces.Dict(observation_dict) for player in self.all_players()
        })

        action_dict = spaces.Box(0, 1, shape=(1,))
        if self.intention_size > 0:
            action_dict = spaces.Box(0, 1, shape=(1 + self.intention_size,))

        self.action_space = spaces.Dict({
            player: action_dict for player in self.all_players()
        })

        self.reset_variables()

    def reset_variables(self):
        self.decks: Dict[str, List[int]] = {}
        self.intentions = np.zeros((self.intention_size, self.number_of_players), dtype=np.float32)
        self.draw_decks()
        self.current_player_index = 0
        self.table = 0
        self.pile_size = 0
        self.stall_count = 0
        self.finished = False
        self.won = False
        self.termination = {
            player: False for player in self.all_players()
        }
        self.termination["__all__"] = False

    def all_players(self):
        for i in range(0, self.number_of_players):
            yield f"player{i+1}"

    def player(self, player):
        return f"player{player+1}"

    @property
    def next_player_index(self):
        return (self.current_player_index + 1) % self.number_of_players

    @property
    def next_player(self):
        return self.player(self.next_player_index)

    def draw_decks(self):
        cards_drawn = np.random.choice(100, self.hand_size * self.number_of_players, replace=False)

        self.decks = {
            player: list(sorted(cards_drawn[index * self.hand_size:index * self.hand_size + self.hand_size]))
            for index, player in enumerate(self.all_players())
        }

    def get_top_cards(self, player: str):
        top_card = self.decks[player][0] if len(self.decks[player]) > 0 else 100
        second_card = self.decks[player][1] if len(self.decks[player]) > 1 else 100
        return top_card, second_card

    def get_top_cards_observation(self, player: str):
        top_card, second_card = self.get_top_cards(player)
        return np.array([top_card / 100, second_card / 100], dtype=np.float32)

    def get_table_observation(self):
        return np.array([self.table / 100], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.reset_variables()

        observation_dict = {
            "table": self.get_table_observation(),
            "hand": self.get_top_cards_observation(self.player(self.current_player_index)),
        }
        if self.intention_size > 0:
            observation_dict["intention"] = np.zeros((self.intention_size, self.number_of_players - 1),
                                                     dtype=np.float32)

        observation: TheMindObservation = {
            self.player(self.current_player_index): observation_dict
        }
        return observation, {self.player(self.current_player_index): {"won": self.won}}

    def compute_if_finished(self):
        if self.finished:
            return

        min_card_on_hand = min(chain(*self.decks.values()))
        lost = self.table > min_card_on_hand
        if lost:
            self.finished = True
            return

        # If one player has emptied his hand, the game is over
        empty_decks = 0
        for deck in self.decks.values():
            if len(deck) == 0:
                empty_decks += 1
        won = empty_decks >= self.number_of_players - 1
        if won:
            # Empty all decks and place all cards on the table
            max_card_on_hand = max(chain(*self.decks.values()))
            self.table = max(self.table, max_card_on_hand)
            self.pile_size = self.hand_size * self.number_of_players
            for player in self.all_players():
                self.decks[player] = []

            self.won = True
            self.finished = True
            return

        stalled = self.stall_count >= self.stall_limit
        if stalled:
            self.finished = True

    def read_action(self, action_dict) -> Tuple[bool, np.ndarray]:
        # Handling for cases when the action dict is received per agent and when it is received as a single dict
        if self.player(self.current_player_index) in action_dict:
            action = action_dict[self.player(self.current_player_index)]
        else:
            action = action_dict

        return action[0] > 0.5, action[1:]

    def step(self, action_dict) -> Tuple[TheMindObservation, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        play, intention = self.read_action(action_dict)

        if self.intention_size > 0:
            self.intentions[:, self.current_player_index] = intention

        hand = self.decks[self.player(self.current_player_index)]

        # Applying action
        if not self.finished and play and len(hand) > 0:
            self.table = hand.pop(0)
            self.stall_count = 0
            self.pile_size += 1
        else:
            self.stall_count += 1

        self.compute_if_finished()

        # intention is the self.intentions matrix with every line but the next-player's line
        observation_dict = {
            "table": self.get_table_observation(),
            "hand": self.get_top_cards_observation(self.next_player),
        }
        if self.intention_size > 0:
            observation_dict["intention"] = np.delete(self.intentions, self.next_player_index, axis=1)

        observation = {
            self.next_player: observation_dict
        }

        # Calculating rewards
        reward = {
            self.player(self.current_player_index):
                self.pile_size / (self.hand_size * self.number_of_players) if self.finished else 0
        }

        # Calculating if terminated
        if self.finished:
            self.termination[self.player(self.current_player_index)] = True
            self.termination["__all__"] = self.termination[self.player(self.current_player_index)] and self.termination[self.next_player]

        # Calculating truncateds
        truncateds = {
            "__all__": False
        }

        self.current_player_index = self.next_player_index
        return (observation, reward, self.termination,
                truncateds, {self.player(self.current_player_index): {"won": self.won}})

    def render(self):
        print("Table:", self.table)
        for player in self.all_players():
            print(self.player(player), self.decks[player])
        print("Current player:", self.current_player_index)
