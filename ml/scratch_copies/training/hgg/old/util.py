import pandas
import os
import pickle

from sklearn.preprocessing import StandardScaler


def load_df(prefix):
    df = pandas.read_parquet(f"{prefix}/data/data.parquet")
    #df["ytrue"] = pandas.read_csv(f"{prefix}/labels/data.labels.gz", header=None)[0].astype("i4")
    #df["weight"] = pandas.read_csv(f"{prefix}/weights/data.weights.gz", header=None)[0]
    return df


def scale_transform(data, params_file: str, train=False):
    if os.path.exists(params_file):
        with open(params_file, "rb") as fin:
            scaler = pickle.load(fin)
            return scaler.transform(data)
    elif train:
        scaler = StandardScaler().fit(X=data)
        with open(params_file, "wb") as fout:
            pickle.dump(scaler, fout)
        return scaler.transform(data)
    raise ValueError("No params and not in training mode!")


def batcher(length, size):
    start = range(0, length+size, size)
    stop = map(
        lambda x: min(x, length),
        range(size, length+size, size)
    )
    return zip(start, stop)
