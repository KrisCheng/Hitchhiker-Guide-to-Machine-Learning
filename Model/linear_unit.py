#!~/anaconda/bin/python

__author__ = "Kris Peng"

# a implement of linear unit, based on Python2.7
from perceptron import Perceptron

f = lambda x: x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    label = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, label

def train_linear_unit():
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 100, 0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    print("Work 3.4 years, monthly salary = %.2f'" % linear_unit.predict([3.4]))
    print("Work 15 years, monthly salary = %.2f'" % linear_unit.predict([15]))
    print("Work 6 years, monthly salary = %.2f'" % linear_unit.predict([6]))
    print("Work 27 years, monthly salary = %.2f'" % linear_unit.predict([27]))