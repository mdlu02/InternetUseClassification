import os
import numpy as np
import pandas as pd
import pingouin as pg
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Set, Dict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42

PCA_CUTOFF = 0.95

AGE_BUCKETS = {
    0: {5, 6, 7, 8, 9},
    1: {10, 11, 12, 13},
    2: {14, 15, 16, 17},
    3: {18, 19, 20, 21, 22}
}

def get_data_types() -> Dict[str, List[str]]:
    # Read in data descriptions
    data_descriptions = pd.read_csv(f"data/data_dictionary.csv")

    cont_columns = []
    cat_columns = ["sii"]

    # Plot pie charts/histograms (large output)
    for _, row in data_descriptions.iterrows():
        if row.Type == "float" or row.Type == "int":
            cont_columns.append(row.Field)

        if row.Type == "categorical int":
            cat_columns.append(row.Field)

        if row.Type == "str" and row.Values is not np.nan:
            cat_columns.append(row.Field)

    return {
        "cont_columns": cont_columns,
        "cat_columns": cat_columns
    }


def cat_cols_to_exclude(
    data: pd.DataFrame,
    columns: List[str],
    hypothesis_threshold: float = 0.1
):
    to_exclude: List[str] = []

    for col in columns:
        # Skip PCIAT columns since those are not available in the test dataset
        if "PCIAT" in col or "sii" in col: continue
        
        # Perform ANOVA
        aov = pg.welch_anova(dv="PCIAT-PCIAT_Total", between=col, data=data)
        
        if hypothesis_threshold < aov["p-unc"].values[0]:
            to_exclude.append(col)
        
    return to_exclude


def cont_cols_to_exclude(data: pd.DataFrame, corr_threshold: float = 0.8) -> Set:
    correlation_matrix: pd.DataFrame = data.corr()

    column_pair_cnts: Dict[str, int] = {}
    correlated_pair_counts: int = 0
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            col1: str = correlation_matrix.columns[i]
            col2: str = correlation_matrix.columns[j]
            correlation: float = correlation_matrix.loc[col1, col2]
            if abs(correlation) > corr_threshold:
                correlated_pair_counts += 1
                column_pair_cnts[col1] = column_pair_cnts.get(col1, 0) + 1
                column_pair_cnts[col2] = column_pair_cnts.get(col2, 0) + 1

    num_columns_with_correlated_pairs: int = len(column_pair_cnts)

    print(f"Number of correlated pairs: {correlated_pair_counts}")
    print(f"Number of columns with correlated pairs: {len(column_pair_cnts)}")

    to_exclude: Set = set()
    to_return: Set = set()
    to_exclude.add("PCIAT-PCIAT_Total")
    prev: int = correlated_pair_counts

    # Greedy search for minimum columns to remove to remove all highly correlated
    # column pairs
    while len(column_pair_cnts) > 0:
        max_column: str = max(column_pair_cnts, key=column_pair_cnts.get)
        if column_pair_cnts[max_column] == 0:
            break
        del column_pair_cnts[max_column]

        to_exclude.add(max_column)
        corr_pair_count: int = 0
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                col1: str = correlation_matrix.columns[i]
                col2: str = correlation_matrix.columns[j]

                # Skip already excluded columns
                if col1 in to_exclude or col2 in to_exclude:
                    continue

                if abs(correlation_matrix.loc[col1, col2]) > corr_threshold:
                    corr_pair_count += 1

                # If not change in the number of correlated pairs, keep the column
        if corr_pair_count == prev:
            print(f"Keeping {max_column}")
            to_return.add(max_column)
        else:
            print(
                f"Removing {max_column}, {corr_pair_count} correlated pairs remaining"
            )
        if corr_pair_count == 0: break
        prev = corr_pair_count

    final_exclusions = to_exclude - to_return
    print(
        f"Removing {len(final_exclusions)} columns out of "
        f"{num_columns_with_correlated_pairs} removes all correlated pairs"
    )
    print(
        "Selected columns: ",
        [x for x in data.columns if x not in final_exclusions]
    )
    return to_exclude


@dataclass
class SplitData:
    train_X: List[pd.DataFrame]
    val_X: List[pd.DataFrame]
    test_X:  List[pd.Series]
    train_y: pd.Series
    val_y: pd.Series
    test_y: pd.Series


def split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    num_folds: int = 5,
    seed: int = SEED
) -> SplitData:
  # Group by existence of timeseries data and stratify each group by `sii`
    original_size = len(data)
    data.dropna(subset=["sii"], inplace=True)
    print(f"Dropped {original_size - len(data)} rows with missing `sii` data")

    train_X, val_X, test_X = [], [], []
    train_y, val_y, test_y = [], [], []

    has_timeseries = set()
    for file_name in tqdm(
        os.listdir(f"data/series_train.parquet"),
        desc="Finding timeseries data"
    ):
        if file_name.startswith("id="):
            has_timeseries.add(file_name[3:])

    data["has_timeseries"] = data["id"].apply(lambda x: x in has_timeseries)
    data["age_group"] = data["Basic_Demos-Age"].apply(
        lambda x: next((k for k, v in AGE_BUCKETS.items() if x in v), -1)
    )
    data["stratify_target"] = (
        data["sii"].astype(str) + "|"
        + data["has_timeseries"].astype(str) + "|"
        + data["age_group"].astype(str)
    )
    stratify_counts = data["stratify_target"].value_counts()

    # union all groups with less samples than the numbef of folds
    data["stratify_target"] = data["stratify_target"].apply(
        lambda x: x if stratify_counts[x] > 5 else "union"
    )

    X = data.drop(columns=["stratify_target"])
    y = data["stratify_target"]
    # Split the test set off
    other_X, test_X, other_y, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    test_y = other_y["PCIAT-PCIAT_Total"]
    # Stratify by the target
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for train_index, val_index in skf.split(other_X, other_y):
        train_X.append(other_X.iloc[train_index])
        val_X.append(other_X.iloc[val_index])
        train_y.append(other_X.iloc[train_index]["PCIAT-PCIAT_Total"])
        val_y.append(other_X.iloc[val_index]["PCIAT-PCIAT_Total"])

    for fold in range(len(train_X)):
        assert len(train_X[fold]) == len(train_y[fold])
        assert len(val_X[fold]) == len(val_y[fold])
        assert len(test_X) == len(test_y)

        assert len(train_X[fold]) + len(val_X[fold]) + len(test_X) == len(X)
        assert abs(len(train_X[fold]) - (1 - 1 / num_folds)**2 * len(X)) < 2

    return SplitData(
        train_X=train_X,
        val_X=val_X,
        test_X=test_X,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
    )

def read_timeseries(file_name, dir_name):
    df = pd.read_parquet(
        os.path.join(dir_name, file_name, "part-0.parquet"),
        "fastparquet",
    )
    df["id"] = file_name.split("=")[1]
    return df


def preprocess(data: SplitData) -> SplitData:
    """
    for each training fold, preprocess as follows:
    - Remove highly correlated continuous features and save the list
    - Remove categorical features that don't have signficant relationship (ANOVA)
        and save the list
    - Load timeseries data and remove highly correlated features and save the list
    - Join the timeseries and non-timeseries data
    - Initialize preprocessing pipeline MinMaxScalar, StandardScalar, OneHotEncoder
    - Fit the pipeline to the training data and use it to transform the validation
        and test data. Store as a single fold for training
        - Store the list of columns to exclude for each fold as well
    """
    out: SplitData = SplitData(
        [], [], [], 
        data.train_y, data.val_y, data.test_y
    )

    results = []
    for file_name in tqdm(
        os.listdir("data/series_train.parquet"),
        desc="Loading timeseries data"
    ):
        if file_name.startswith("id="):
            results.append(read_timeseries(
                file_name, f"data/series_train.parquet"
            ))
    timeseries_df = pd.concat(results)
    del results
    timeseries_df = timeseries_df[
        ["id", "PCIAT-PCIAT_Total", "daytime", "weekday",
         "light", "anglez", "enmo", "non-wear_flag"]
    ]

    data_types: Dict[str, List[str]] = get_data_types()
    for i in range(len(data.train_X)):
        print(f"Preprocessing fold {i + 1}")
        train_X = data.train_X[i]
        val_X = data.val_X[i]
        test_X = data.test_X[i]

        to_exclude: List[str] = []
        # Get tabular continuous features to remove (based on correlation)
        to_exclude.append(
            cont_cols_to_exclude(train_X[data_types["cont_columns"]])
        )
        # Get tabular categorical features to remove (based on ANOVA)
        to_exclude.append(
            cont_cols_to_exclude(train_X[data_types["cat_columns"]])
        )
        train_X.drop(columns=to_exclude, inplace=True)
        val_X.drop(columns=to_exclude, inplace=True)
        test_X.drop(columns=to_exclude, inplace=True)
        train_X = pd.merge(train_X, timeseries_df, on="id")
        val_X = pd.merge(val_X, timeseries_df, on="id")
        test_X = pd.merge(test_X, timeseries_df, on="id")
        
        


    