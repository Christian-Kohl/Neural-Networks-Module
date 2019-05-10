import numpy as np
from sklearn.model_selection import train_test_split
exec(open('data_file.py').read())
exec(open('MLP.py').read())

dataset, dataframe = load_data()
importance = corr_features_output(dataframe)

X = dataset[:, importance]
y = dataset[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

network = MLP(X_train.shape[1], [3, 2], 1, 0.005, 0.9, 0.01)
log = network.fit(X_train, y_train, X_test, y_test, 500)
print(log)
network.predict(X_test[:10, :])
