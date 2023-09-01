""" IGT module """
from pathlib import Path

import pandas as pd

import igt_website.backend.deck

OUTPUT_FILES_PATH = str(Path.home()) + "/PycharmProjects/TFG/output_files/"
DATA_RECORDS_PATH = OUTPUT_FILES_PATH + "data_records.xlsx"


class IGT:
    """ IGT class """

    def __init__(self):

        deck_a = igt_website.backend.deck.Deck('A', reward=100,
                                               penalties=[[-350, -300, -250, -200, -150, 0],
                                                          [.1, .1, .1, .1, .1, .5]])
        deck_b = igt_website.backend.deck.Deck('B', reward=100,
                                               penalties=[[-1250, 0],
                                                          [.1, .9]])
        deck_c = igt_website.backend.deck.Deck('C', reward=50,
                                               penalties=[[-25, -50, -75, 0],
                                                          [.1, .1, .1, .7]])
        deck_d = igt_website.backend.deck.Deck('D', reward=50,
                                               penalties=[[-250, 0],
                                                          [.1, .9]])
        self.decks = {
            'A': deck_a,
            'B': deck_b,
            'C': deck_c,
            'D': deck_d
        }

        self.remaining_decisions = 100
        self.loan = 1000
        self.money = self.loan
        self.last_card = None
        self.game_over = False
        self.data_records = None
        self.card_sequence = []
        self.time_sequence = []

    def draw_card(self, selected_deck, timestamp):
        """ Draw card from selected deck """
        if self.remaining_decisions > 0 and not self.game_over:
            self.last_card = self.decks[selected_deck].draw_card_from_deck()
            self.time_sequence.append(timestamp)
            self.update_game_state(selected_deck)
        else:
            raise Exception
        return self.last_card

    def update_game_state(self, selected_deck):
        """ Update game state after card draw from deck """
        self.remaining_decisions -= 1
        self.money += (self.last_card[1] + self.last_card[2])
        self.card_sequence.append(selected_deck)
        for deck in self.decks.items():
            if deck[1].name == selected_deck:
                deck[1].decisions_not_choosing = 0
            else:
                deck[1].decisions_not_choosing += 1
        self.save_decision()
        if self.remaining_decisions == 0:
            self.save_game_result()

    def save_decision(self):
        """ Save info of each decision """
        decision = {
            "Remaining decisions": self.remaining_decisions,
            "Money accumulated": self.money,
            "Last outcome was positive": self.last_card[1] + self.last_card[2] >= 0,
            "Last outcome was negative": self.last_card[1] + self.last_card[2] < 0,
            "Money accumulated > Loan": self.money >= self.loan,
            "Money accumulated < Loan": self.money < self.loan,
            "Current decision": 100 - self.remaining_decisions,

            "Cards chosen from deck (A)": self.decks['A'].chosen_cards,
            "Positive last outcome (A)": self.decks['A'].last_outcome and self.decks['A'].last_outcome >= 0,
            "Negative last outcome (A)": self.decks['A'].last_outcome and self.decks['A'].last_outcome < 0,
            "Positive cards (A)": self.decks['A'].positive_cards,
            "Negative cards (A)": self.decks['A'].negative_cards,
            "Frequency of positive cards (A)":
                self.decks['A'].positive_cards / self.decks['A'].chosen_cards
                if self.decks['A'].chosen_cards != 0 else None,
            "Frequency of negative cards (A)":
                self.decks['A'].negative_cards / self.decks['A'].chosen_cards
                if self.decks['A'].chosen_cards != 0 else None,
            "Best outcome (A)": self.decks['A'].best_outcome,
            "Worst outcome (A)": self.decks['A'].worst_outcome,
            "decisions without choosing (A)": self.decks['A'].decisions_not_choosing,
            "Won money (A)": self.decks['A'].won_money,
            "Lost money (A)": self.decks['A'].lost_money,

            "Cards chosen from deck (B)": self.decks['B'].chosen_cards,
            "Positive last outcome (B)": self.decks['B'].last_outcome and self.decks['B'].last_outcome >= 0,
            "Negative last outcome (B)": self.decks['B'].last_outcome and self.decks['B'].last_outcome < 0,
            "Positive cards (B)": self.decks['B'].positive_cards,
            "Negative cards (B)": self.decks['B'].negative_cards,
            "Frequency of positive cards (B)":
                self.decks['B'].positive_cards / self.decks['B'].chosen_cards
                if self.decks['B'].chosen_cards != 0 else None,
            "Frequency of negative cards (B)":
                self.decks['B'].negative_cards / self.decks['B'].chosen_cards
                if self.decks['B'].chosen_cards != 0 else None,
            "Best outcome (B)": self.decks['B'].best_outcome,
            "Worst outcome (B)": self.decks['B'].worst_outcome,
            "decisions without choosing (B)": self.decks['B'].decisions_not_choosing,
            "Won money (B)": self.decks['B'].won_money,
            "Lost money (B)": self.decks['B'].lost_money,

            "Cards chosen from deck (C)": self.decks['C'].chosen_cards,
            "Positive last outcome (C)": self.decks['C'].last_outcome and self.decks['C'].last_outcome >= 0,
            "Negative last outcome (C)": self.decks['C'].last_outcome and self.decks['C'].last_outcome < 0,
            "Positive cards (C)": self.decks['C'].positive_cards,
            "Negative cards (C)": self.decks['C'].negative_cards,
            "Frequency of positive cards (C)":
                self.decks['C'].positive_cards / self.decks['C'].chosen_cards
                if self.decks['C'].chosen_cards != 0 else None,
            "Frequency of negative cards (C)":
                self.decks['C'].negative_cards / self.decks['C'].chosen_cards
                if self.decks['C'].chosen_cards != 0 else None,
            "Best outcome (C)": self.decks['C'].best_outcome,
            "Worst outcome (C)": self.decks['C'].worst_outcome,
            "decisions without choosing (C)": self.decks['C'].decisions_not_choosing,
            "Won money (C)": self.decks['C'].won_money,
            "Lost money (C)": self.decks['C'].lost_money,

            "Cards chosen from deck (D)": self.decks['D'].chosen_cards,
            "Positive last outcome (D)": self.decks['D'].last_outcome and self.decks['D'].last_outcome >= 0,
            "Negative last outcome (D)": self.decks['D'].last_outcome and self.decks['D'].last_outcome < 0,
            "Positive cards (D)": self.decks['D'].positive_cards,
            "Negative cards (D)": self.decks['D'].negative_cards,
            "Frequency of positive cards (D)":
                self.decks['D'].positive_cards / self.decks['D'].chosen_cards
                if self.decks['D'].chosen_cards != 0 else None,
            "Frequency of negative cards (D)":
                self.decks['D'].negative_cards / self.decks['D'].chosen_cards
                if self.decks['D'].chosen_cards != 0 else None,
            "Best outcome (D)": self.decks['D'].best_outcome,
            "Worst outcome (D)": self.decks['D'].worst_outcome,
            "decisions without choosing (D)": self.decks['D'].decisions_not_choosing,
            "Won money (D)": self.decks['D'].won_money,
            "Lost money (D)": self.decks['D'].lost_money
        }

        if self.data_records is None:
            self.data_records = pd.DataFrame.from_records([decision])
        else:
            row = pd.DataFrame.from_records([decision])
            self.data_records = pd.concat([self.data_records, row])

        return decision

    def print_state(self):
        """ Retrieve game state """
        game_state = {
            'decisions': self.remaining_decisions,
            'score': self.money,
            'deck_a': 100 - self.decks['A'].chosen_cards,
            'deck_b': 100 - self.decks['B'].chosen_cards,
            'deck_c': 100 - self.decks['C'].chosen_cards,
            'deck_d': 100 - self.decks['D'].chosen_cards
        }
        return game_state

    @staticmethod
    def save_player_info(form_data):
        """ Save player info """
        form_data_df = pd.DataFrame.from_records([form_data])
        return form_data_df

    def save_game_result(self):
        """ Save game data when game over """
        self.game_over = True
        self.data_records['Selected deck'] = self.card_sequence[1:] + [None]
        self.data_records['Time'] = [None] + [self.time_sequence[i + 1] - self.time_sequence[i] for i in
                                              range(len(self.time_sequence) - 1)]
        self.data_records.index = range(1, len(self.card_sequence) + 1)
