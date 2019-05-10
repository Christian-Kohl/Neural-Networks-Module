import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.preprocessing as ppr


def load_data():
    dataset = np.loadtxt('training_data.txt')
    dataset = dataset[dataset[:, -1] != 50, :]
    dataset[:, :3] = ppr.normalize(dataset[:, :3], axis=0)
    dataset[:, 4:-1] = ppr.normalize(dataset[:, 4:-1], axis=0)
    df = pd.DataFrame(dataset)
    return dataset, df


def discrete_or_cont(dataset):
    for idx, col in enumerate(dataset[:].T):
        unique = np.unique(col, axis=0)
        print(idx, ':', len(unique)/len(col)*100, '% unique')


def scatter_matrix(df):
    plt.figure(figsize=(28, 28))
    pd.plotting.scatter_matrix(df, diagonal="kde")
    plt.show()


# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# https://stackoverflow.com/questions/39409866/correlation-heatmap
def heatmap(df):
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                annot=True, cmap="RdYlGn")
    plt.show()


def corr_features_output(df):
    corr = df.corr()
    output_corr = corr.iloc[-1, :-1]
    output_corr = pd.Series.sort_values(pd.Series.abs(output_corr),
                                        ascending=False)
    return output_corr.index.values


dataset, df = load_data()
print(corr_features_output(df))
