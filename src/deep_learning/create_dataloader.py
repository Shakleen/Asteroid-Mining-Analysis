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
    batch_size,
):
    df_processed = df.copy(deep=True)

    _min_max_normalize(numerical_columns, df_processed)
    df_processed = _one_hot_encode(categorical_columns)

    set_1, set_2 = _split_sets(target_columns, df_processed)

    train = set_1.sample(frac=0.95, random_state=29)
    valid = set_1[~set_1.isin(train)].dropna()

    train_loader = _build_data_loader(target_columns, batch_size, train, "Training")
    valid_loader = _build_data_loader(target_columns, batch_size, valid, "Validation")
    inf_loader = _build_data_loader(
        target_columns, batch_size, set_2, "Inference", False
    )

    return train_loader, valid_loader, inf_loader


def _split_sets(target_columns, df_processed):
    set_1 = df_processed
    for target in target_columns:
        set_1 = set_1[set_1[target].notnull()]

    set_2 = df_processed[~df_processed.isin(set_1)].dropna(how="all")

    print(f"Number of examples for training purposes: {set_1.shape[0]}")
    print(f"Number of examples for inference purposes: {set_2.shape[0]}")

    return set_1, set_2


def _one_hot_encode(categorical_columns):
    df_processed = pd.get_dummies(
        df_processed,
        columns=categorical_columns,
        dummy_na=True,
    )

    return df_processed


def _min_max_normalize(numerical_columns, df_processed):
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
