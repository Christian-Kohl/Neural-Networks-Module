import numpy as np
import sklearn as sk
exec(open('data_file.py').read())
exec(open('MLP.py').read())

network = MLP(2, [5, 5], 1, 0.01, 0.2, reg_term=0.001)
print(dataset)
