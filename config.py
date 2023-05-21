import os

path_net = 'networks/patient22.h5'
path_data = 'data/patient22_test.csv'
input_shape = None  # (d1, d2, d3)
mean = 0
std = 1
_, ext = os.path.splitext(path_data)
deltas = [0.0005, 0.001, 0.0015, 0.002]
timeout = 100000
