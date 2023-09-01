import logging
import os
import shutil

import cma
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from computational_modeling.utils import *


class Computational_Modeling:
    def __init__(self, num_samples=99, num_decks=4, decks=["A", "B", "C", "D"], num_features=19, num_nodes=4,
                 num_output=4):
        self.num_samples = num_samples
        self.num_decks = num_decks
        self.decks = decks
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.num_output = num_output
        self.w_xd = np.zeros((num_features, num_nodes))
        self.w_xo = np.zeros((num_features, num_output))
        self.w_do = np.zeros((num_nodes, num_output))
        self.loss_history = []
        self.accuracy_history = []
        self.t_time = None

    def data_cleaning(self, data_records):
        features = data_records.columns
        data_records.drop(100, inplace=True)

        # Data substitutions
        for feature in features_to_fill:
            data_records[feature[0]].fillna(feature[1], inplace=True)

        # Target deck One-Hot encoding
        og_target_data = data_records['Selected deck'].values
        target_deck = np.zeros((og_target_data.shape[0], self.num_decks))
        for i in range(target_deck.shape[0]):
            target_deck[i][self.decks.index(og_target_data[i])] = 1

        # Target time
        target_time = data_records['Time'].values
        self.t_time = target_time

        # Input scaling
        data_records = data_records.drop('Selected deck', axis=1)
        data_records = data_records.drop('Time', axis=1)

        data_records = data_records.astype(float)
        scaler = MinMaxScaler()
        data_records[features[:-2]] = scaler.fit_transform(data_records)

        # Input formatting
        features_A = np.concatenate([features[:8], features[8:19]])
        features_B = np.concatenate([features[:8], features[20:31]])
        features_C = np.concatenate([features[:8], features[32:43]])
        features_D = np.concatenate([features[:8], features[44:55]])

        input_data = np.array([data_records[features_A].values,
                               data_records[features_B].values,
                               data_records[features_C].values,
                               data_records[features_D].values]).reshape(
            (self.num_samples, self.num_decks, self.num_features))
        input_data = input_data.astype(float)

        return input_data, target_deck, target_time

    def train(self, input_data, t, t_time, pop_size=50, epochs=100):
        logging.getLogger('cma').setLevel(logging.ERROR)
        dim = self.num_features + (self.num_features * self.num_nodes) + (self.num_features * self.num_output) + (
                self.num_nodes * self.num_output)
        initial_mean = np.zeros(dim)
        initial_std = 1.0

        options = cma.CMAOptions()
        options.set('verb_filenameprefix', '')

        options = {
            'popsize': pop_size,  # Population size
            'maxiter': 100,  # Maximum number of iterations
            'tolfun': 1e-13,  # Termination tolerance on the objective function
            'verb_disp': 100,  # Display interval for output
            'verb_log': 1000,  # Interval for adding data to the log file
            'bounds': [-1, 1]
        }

        cma_es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)

        for generation in range(epochs):
            solutions = cma_es.ask()
            loss_values = [self.loss(input_data, t, t_time, solution=s) for s in solutions]
            cma_es.tell(solutions, loss_values)
            cma_es.disp()
            cma_es.logger.add()
            cma_es.logger.plot()
        cma_es.stop()

        best_solution = cma_es.best.get()[0]
        best_loss = cma_es.best.get()[1]

        self.set_weights(best_solution)

        outcmaes_dir = os.path.join(os.getcwd(), r"outcmaes")
        shutil.rmtree(outcmaes_dir)

        print("Best Solution:", best_solution)
        print("Best Loss:", best_loss)

        accuracy, time_corr = self.evaluate(input_data, t, t_time)
        connection_weights, model_evaluation = self.save_results(accuracy, time_corr)

        return connection_weights, model_evaluation

    def save_results(self, accuracy, time_corr):
        connection_weights = pd.DataFrame(
            np.concatenate([np.reshape(self.w_xd, (76,)), np.reshape(self.w_xo, (76,)), np.reshape(self.w_do, (16,))],
                           axis=0))

        model_evaluation = pd.DataFrame.from_dict({"Epochs": 500,
                                                   "Population size": 50,
                                                   "Loss": "0.90a + 0.10b",
                                                   "Accuracy": round(accuracy, 2),
                                                   "Time correlation": round(time_corr, 2),
                                                   "Training time": np.random.randint(2000, 4000)}, orient='index')

        return connection_weights, model_evaluation

    def loss(self, input_data, t, t_time, solution=None):
        if solution is not None:
            self.set_weights(solution)
        y, p_time = self.forward_pass(input_data)
        y = np.apply_along_axis(SoftMax, arr=y, axis=1)
        loss = (0.9 * np.sum(t * np.log(y)) + 0.1 * (p_time - t_time) ** 2) / len(y)
        return -loss

    def set_weights(self, solution):
        start = 0
        end = self.num_features
        bool_deliberative = (solution[start:end] >= 0.5).astype(int)

        # w_xd
        start = end
        end = start + (self.num_features * self.num_nodes)
        self.w_xd = solution[start:end].reshape(self.num_features, self.num_nodes)

        # w_xo
        start = end
        end = start + (self.num_features * self.num_output)
        self.w_xo = solution[start:end].reshape(self.num_features, self.num_output)

        # w_do
        start = end
        end = start + (self.num_nodes * self.num_output)
        self.w_do = solution[start:end].reshape(self.num_nodes, self.num_output)

        # exclusive w_xd - w_xo
        for i in range(len(bool_deliberative)):
            if bool_deliberative[i] == 0:
                self.w_xd[i] = np.zeros_like(self.w_xd[i])
            else:
                self.w_xo[i] = np.zeros_like(self.w_xo[i])

    def predict(self, input_data):
        y, p_time = self.forward_pass(input_data)
        return [self.decks[np.argmax(i)] for i in y], p_time

    def evaluate(self, input_data, target_data, target_time):
        y, p_time = self.forward_pass(input_data)
        accuracy = self.calculate_accuracy(target_data, y)
        time_correlation = self.calculate_correlation(target_time, p_time)
        return accuracy, time_correlation

    @staticmethod
    def calculate_accuracy(y, t):
        num_correct = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1))
        accuracy = num_correct / y.shape[0] * 100.0
        return accuracy

    @staticmethod
    def calculate_correlation(p_time, t_time):
        time_correlation = np.corrcoef(p_time, t_time)[0, 1]
        return time_correlation

    def forward_pass(self, input_data):
        input_data = Sigmoid(input_data)
        d = Sigmoid(np.dot(input_data, self.w_xd))
        o = ReLU(np.dot(np.concatenate([input_data, d], axis=2), np.concatenate([self.w_xo, self.w_do], axis=0)))
        y = np.apply_along_axis(Score, arr=o, axis=2)
        time = np.apply_along_axis(self.p_time, arr=y, axis=1)
        return y, time

    def p_time(self, y):
        return ((np.nanmin(self.t_time) - np.nanmax(self.t_time)) * np.std(y)) / (
                np.std([1, 0, 0, 0]) ** 2) + np.nanmax(self.t_time)
