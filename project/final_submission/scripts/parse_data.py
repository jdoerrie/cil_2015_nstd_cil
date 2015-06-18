#!/bin/env python
import sys
import numpy

"""
This small script parses the data present in GridSearch_*.txt to extract
information about how well the algorithm did after a given epoch. It finds the
information for every run and aggregates the information about the RMSE. It
finally reports the mean and standard deviation of it.
"""


def do_calc(buff, line):
    K, lam, gam, shr = line.split(',')[:4]

    epochs = {}
    for buf in buff:
        epoch = int(buf.split()[1][:-1], 10)
        rmse = float(buf.split()[4][:-1])
        if epoch not in epochs:
            epochs[epoch] = []
        epochs[epoch].append(rmse)

    for epoch in epochs:
        curr_data = epochs[epoch]
        epochs[epoch] = (numpy.mean(curr_data), numpy.std(curr_data))

    for epoch in sorted(epochs):
        mean = epochs[epoch][0]
        std = epochs[epoch][1]
        print("{},{},{},{},{},{:.6f},{:.6f}".format(
            K, lam, gam, shr, epoch, mean, std))


def main():
    print('K,Lambda,Gamma,Shrink,Epoch,Mean_RMSE,Std_RMSE')
    my_buffer = []
    for line in sys.stdin.readlines():
        if line[:2] == '64':
            do_calc(my_buffer, line)
            my_buffer = []
        elif line[:5] == 'Epoch':
            my_buffer.append(line)


if __name__ == '__main__':
    main()
