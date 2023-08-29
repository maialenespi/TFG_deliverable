""" Deck module """
import numpy as np


class Deck:
    """ Deck class """
    def __init__(self, name=None, reward=None, penalties=None):
        self.name = name
        self.chosen_cards = 0
        self.positive_cards = 0
        self.negative_cards = 0
        self.last_outcome = None
        self.worst_outcome = None
        self.best_outcome = None
        self.won_money = 0
        self.lost_money = 0
        self.decisions_not_choosing = 0
        self.reward = reward
        self.penalties = penalties

    def draw_card_from_deck(self):
        """ Draw card from deck """
        if self.chosen_cards < 100:
            reward = self.reward
            penalty = int(np.random.choice(self.penalties[0], 1, p=self.penalties[1]))
            self.update_deck(penalty, reward)
        else:
            raise Exception
        return [self.name, reward, penalty]

    def update_deck(self, penalty, reward):
        """ Update deck state after card draw """
        outcome = reward + penalty
        self.chosen_cards += 1
        if outcome >= 0:
            self.positive_cards += 1
        else:
            self.negative_cards += 1
        self.last_outcome = outcome
        if (self.worst_outcome is None) or (self.worst_outcome > self.last_outcome):
            self.worst_outcome = self.last_outcome
        if (self.best_outcome is None) or (self.best_outcome < self.last_outcome):
            self.best_outcome = self.last_outcome
        self.won_money += reward
        self.lost_money += -penalty
