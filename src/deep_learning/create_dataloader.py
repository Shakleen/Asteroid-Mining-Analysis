import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader


def create_dataloader(
    df,
    numerical_columns,
    categorical_columns,
    target_columns,
    exclude_columns,
    batch_size,
    discard_outliers=True,
):
    df_processed = df.drop(columns=exclude_columns).copy(deep=True)

    _min_max_normalize(numerical_columns, target_columns, df_processed)
    df_processed = _one_hot_encode(df_processed, categorical_columns)

    set_1, set_2 = _split_sets(target_columns, df_processed)

    if discard_outliers:
        set_1 = _remove_outliers(target_columns, set_1)

    valid = set_1.sample(10240, random_state=29)
    train = set_1[~set_1.isin(valid)].dropna()

    train_loader = _build_data_loader(target_columns, batch_size, train, "Training")
    valid_loader = _build_data_loader(target_columns, batch_size, valid, "Validation")
    inf_loader = _build_data_loader(
        target_columns, batch_size, set_2, "Inference", False
    )

    return train_loader, valid_loader, inf_loader


def _remove_outliers(target_column, df):
    temp = df[target_column].dropna()
    q1 = np.quantile(temp, 0.25)
    q3 = np.quantile(temp, 0.75)
    iqr = q3 - q1
    span = 1.5 * iqr
    lower_fence = q1 - span
    upper_fense = q3 + span
    
    df = df[
        (df[target_column] >= lower_fence)
        & (df[target_column] <= upper_fense)
    ]
    print("After removing outliers:", df.shape)
    return df


def _split_sets(target_columns, df_processed):
    set_1 = df_processed
    for target in target_columns:
        set_1 = set_1[set_1[target].notnull()]

    set_2 = df_processed[~df_processed.isin(set_1)].dropna(how="all")

    print(f"Number of examples for training purposes: {set_1.shape[0]}")
    print(f"Number of examples for inference purposes: {set_2.shape[0]}")

    return set_1, set_2


def _one_hot_encode(df_processed, categorical_columns):
    df_processed = pd.get_dummies(
        df_processed,
        columns=categorical_columns,
        dummy_na=False,
    )

    return df_processed


def _min_max_normalize(numerical_columns, target_columns, df_processed):
    numerical_columns = numerical_columns.drop(target_columns)

    for column in numerical_columns:
        values = df_processed[column].values.reshape(-1, 1)
        df_processed[column] = MinMaxScaler().fit_transform(values)


def _build_data_loader(target_columns, batch_size, data, set_name, shuffle=True):
    X, y = _split_features_and_targets(target_columns, data, set_name)
    date_loader = DataLoader(
        list(zip(X, y)),
        shuffle=shuffle,
        batch_size=batch_size,
    )

    return date_loader


def _split_features_and_targets(target_columns, data, set_name):
    X = torch.tensor(
        data.drop(columns=target_columns).values.astype(np.float32),
        dtype=torch.float32,
    )
    y = torch.tensor(
        data[target_columns].values.astype(np.float32), dtype=torch.float32
    )

    print(f"{set_name} X shape: {X.shape}")
    print(f"{set_name} y shape: {y.shape}")

    return X, y
