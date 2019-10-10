import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def find_uniques(se, dtype="category"):
    name = se.name
    flatten = lambda l: [item for sublist in l for item in sublist]

    unique_values = se.values

    unique_values = np.unique(flatten(unique_values), return_counts=True)

    d = {name: unique_values[0], "Count": unique_values[1]}
    types = {name: dtype, "Count": "int32"}

    df_unique = pd.DataFrame(d).astype(types)

    return df_unique.sort_values("Count", ascending=False)


def one_hot(df_main, column):
    df = df_main.copy()
    uniques = find_uniques(df[column])

    for cat in uniques[column].values:
        df[cat] = df[column].apply(lambda x: cat in x).astype("int32")

    return df


def column_score(true: pd.DataFrame, predicted):
    pred_dict = []
    if type(predicted) == np.ndarray:
        for i, cat in enumerate(true.columns):
            pred_dict.append({"Category": cat,
                              "Accuracy Score": accuracy_score(true[cat], predicted[:, i]),
                              "Precision Score": precision_score(true[cat], predicted[:, i])})
    else:
        for i, cat in enumerate(true.columns):
            pred_dict.append({"Category": cat,
                              "Accuracy Score": accuracy_score(true[cat], predicted[:, i].A),
                              "Precision Score": precision_score(true[cat], predicted[:, i].A)})

    return pd.DataFrame(pred_dict)
