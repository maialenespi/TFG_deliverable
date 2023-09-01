import numpy as np

features_to_fill = [["Positive last outcome (A)", False],
                    ["Positive last outcome (B)", False],
                    ["Positive last outcome (C)", False],
                    ["Positive last outcome (D)", False],
                    ["Negative last outcome (A)", False],
                    ["Negative last outcome (B)", False],
                    ["Negative last outcome (C)", False],
                    ["Negative last outcome (D)", False],
                    ["Frequency of positive cards (A)", 0.5],
                    ["Frequency of positive cards (B)", 0.5],
                    ["Frequency of positive cards (C)", 0.5],
                    ["Frequency of positive cards (D)", 0.5],
                    ["Frequency of negative cards (A)", 0.5],
                    ["Frequency of negative cards (B)", 0.5],
                    ["Frequency of negative cards (C)", 0.5],
                    ["Frequency of negative cards (D)", 0.5],
                    ["Best outcome (A)", 0],
                    ["Best outcome (B)", 0],
                    ["Best outcome (C)", 0],
                    ["Best outcome (D)", 0],
                    ["Worst outcome (A)", 0],
                    ["Worst outcome (B)", 0],
                    ["Worst outcome (C)", 0],
                    ["Worst outcome (D)", 0]]


def Score(row):
    return row[0] / (1 + row[1]) - row[2] / (1 + row[3])


def SoftMax(row):
    return np.exp(row) / np.sum(np.exp(row), axis=0)


def ReLU(arr):
    arr[arr < 0] = 0
    return arr


def Sigmoid(arr):
    return 1 / (1 + np.exp(-arr))
