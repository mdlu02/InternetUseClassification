import os
import warnings
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

warnings.filterwarnings("ignore", category=UserWarning)

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
    columns: List[str] = None,
    hypothesis_threshold: float = 0.1
):
    to_exclude: List[str] = []
    if columns is None:
        columns = data.columns
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

    # print(f"Number of correlated pairs: {correlated_pair_counts}")
    # print(f"Number of columns with correlated pairs: {len(column_pair_cnts)}")

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
            # print(f"Keeping {max_column}")
            to_return.add(max_column)
        # else:
        #     print(
        #         f"Removing {max_column}, {corr_pair_count} correlated pairs remaining"
        #     )
        if corr_pair_count == 0: break
        prev = corr_pair_count

    final_exclusions = to_exclude - to_return
    # print(
    #     f"Removing {len(final_exclusions)} columns out of "
    #     f"{num_columns_with_correlated_pairs} removes all correlated pairs"
    # )
    # print(
    #     "Selected columns: ",
    #     [x for x in data.columns if x not in final_exclusions]
    # )
    return list(to_exclude)


@dataclass
class SplitData:
    train_X: List[pd.DataFrame]
    val_X: List[pd.DataFrame]
    test_X:  List[pd.Series]
    train_y: pd.Series
    val_y: pd.Series
    test_y: pd.Series


@dataclass
class FoldData:
    train_X: pd.DataFrame
    val_X: pd.DataFrame
    test_X: pd.DataFrame
    train_y: pd.Series
    val_y: pd.Series
    test_y: pd.Series
    to_exclude: List[str]
    pipeline: Pipeline


def split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    num_folds: int = 5,
    seed: int = SEED
) -> SplitData:
  # Group by existence of timeseries data and stratify each group by `sii`
    print("Splitting data")
    original_size = len(data)
    data.dropna(subset=["sii"], inplace=True)
    print(f"Dropped {original_size - len(data)} rows with missing `sii` data")

    train_X, val_X = [], []
    train_y, val_y = [], []

    has_timeseries = set()
    for file_name in os.listdir(f"data/series_train.parquet"):
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

    X = data
    y = data["stratify_target"]
    # Split the test set off
    other_X, test_X, _, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    other_y = other_X["PCIAT-PCIAT_Total"]
    test_y = test_X["PCIAT-PCIAT_Total"]
    to_drop = ["age_group", "has_timeseries", "stratify_target"]
    for col in other_X.columns:
        if (col.startswith("PCIAT") and not col.endswith("Total")) or col.startswith("sii"):
            to_drop.append(col)
    other_X.drop(columns=to_drop, inplace=True)
    test_X.drop(columns=to_drop, inplace=True)
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


def gen_timeseries_summary_data(df):
    summary_data = []

    for id, group in df.groupby("id"):
        summary_data.append({
            "id": id,
            "light_median": group["light"].median(),
            "light_std": group["light"].std(),
            "light_min": group["light"].min(),
            "light_max": group["light"].max(),
            "light_25": group["light"].quantile(0.25),
            "light_50": group["light"].quantile(0.50),
            "light_75": group["light"].quantile(0.75),
            "light_90": group["light"].quantile(0.90),
            "anglez_median": group["anglez"].median(),
            "anglez_std": group["anglez"].std(),
            "anglez_min": group["anglez"].min(),
            "anglez_max": group["anglez"].max(),
            "anglez_25": group["anglez"].quantile(0.25),
            "anglez_50": group["anglez"].quantile(0.50),
            "anglez_75": group["anglez"].quantile(0.75),
            "anglez_90": group["anglez"].quantile(0.90),
            "enmo_median": group["enmo"].median(),
            "enmo_std": group["enmo"].std(),
            "enmo_min": group["enmo"].min(),
            "enmo_max": group["enmo"].max(),
            "enmo_25": group["enmo"].quantile(0.25),
            "enmo_50": group["enmo"].quantile(0.50),
            "enmo_75": group["enmo"].quantile(0.75),
            "enmo_90": group["enmo"].quantile(0.90),
        })

    return pd.DataFrame(summary_data).reset_index(drop=True)


def read_timeseries(file_name, dir_name):
    df = pd.read_parquet(
        os.path.join(dir_name, file_name, "part-0.parquet"),
        "fastparquet",
    )
    df["daytime"] = df["time_of_day"].apply(lambda x: x / 3_600_000_000_000)
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
    out: List[FoldData] = []

    results = []
    for file_name in tqdm(
        os.listdir("data/series_train.parquet"),
        desc="Loading timeseries data"
    ):
        if file_name.startswith("id="):
            results.append(read_timeseries(
                file_name, f"data/series_train.parquet"
            ))
        if len(results) > 100:
            break
    timeseries_df = pd.concat(results)
    del results
    # Getting summary stats for timeseries split by weekday and weekend
    print("Generating timeseries summary data")
    timeseries_df = timeseries_df[
        ["id", "daytime", "weekday",
         "light", "anglez", "enmo", "non-wear_flag"]
    ]
    timeseries_df = timeseries_df[timeseries_df["non-wear_flag"] == 0]
    weekday_df = timeseries_df[timeseries_df["weekday"].isin([0, 1, 2, 3, 4])]
    weekend_df = timeseries_df[timeseries_df["weekday"].isin([5, 6])]

    weekday_df = gen_timeseries_summary_data(weekday_df)
    weekend_df = gen_timeseries_summary_data(weekend_df)

    for col in weekday_df.columns:
        if col != "id":
            weekday_df.rename(columns={col: f"weekday_{col}"}, inplace=True)
            weekend_df.rename(columns={col: f"weekend_{col}"}, inplace=True)
    timeseries_df = pd.merge(timeseries_df, weekday_df, on="id")

    data_types: Dict[str, List[str]] = get_data_types()
    for i in tqdm(range(len(data.train_X)), desc="Preprocessing folds"):
        train_X = data.train_X[i]
        val_X = data.val_X[i]

        to_exclude: List[str] = []
        # Get tabular continuous features to remove (based on correlation)
        to_exclude.extend(
            cont_cols_to_exclude(train_X[data_types["cont_columns"]])
        )
        # Get tabular categorical features to remove (based on ANOVA)
        to_exclude.extend(
            cat_cols_to_exclude(train_X[
                [col for col in data_types["cat_columns"] if col in train_X.columns]
                + ["PCIAT-PCIAT_Total"]
            ])
        )
        train_X.drop(columns=to_exclude, inplace=True)
        val_X.drop(columns=to_exclude, inplace=True)

        # Timeseries columns to exclude
        timeseries_subset = timeseries_df[
            timeseries_df["id"].isin(train_X["id"])
        ]
        timeseries_to_exclude = cont_cols_to_exclude(timeseries_subset.drop(columns=["id"]))
        train_X = pd.merge(train_X, timeseries_subset, on="id")
        val_X = pd.merge(
            val_X,
            timeseries_df[timeseries_df["id"].isin(val_X["id"])],
            on="id"
        )
        test_X = pd.merge(
            data.test_X,
            timeseries_df[timeseries_df["id"].isin(test_X["id"])],
            on="id"
        )
        train_X.drop(columns=timeseries_to_exclude, inplace=True)
        val_X.drop(columns=timeseries_to_exclude, inplace=True)
        test_X.drop(columns=timeseries_to_exclude, inplace=True)

        # Initialize pipeline
        filtered_cont_columns = [
            col for col in train_X.columns if col not in data_types["cat_columns"]
        ]
        one_hot = [
            col for col in train_X.columns if col in data_types["cat_columns"]
        ]
        min_max, std_scaler = [], []
        for col in filtered_cont_columns:
            if col.startswith("anglez_"):
                min_max.append(col)
            elif col == "Basic_Demos-Age":
                one_hot.append(col)
            else:
                std_scaler.append(col)
        
        pipeline = Pipeline([
            ("min_max", MinMaxScaler(), min_max),
            ("std_scaler", StandardScaler(), std_scaler),
            ("one_hot", OneHotEncoder(), one_hot)
        ])

        # Fit pipeline
        train_X = pipeline.fit_transform(train_X)
        val_X = pipeline.transform(val_X)
        test_X = pipeline.transform(test_X)

        out.append(
            FoldData(
                train_X=train_X,
                val_X=val_X,
                test_X=test_X,
                train_y=data.train_y,
                val_y=data.val_y,
                test_y=data.test_y,
                to_exclude=to_exclude + timeseries_to_exclude,
                pipeline=pipeline
            )
        )
    return out


    