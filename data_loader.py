# %%
"""
data_process is a script that process train and test data
by filling out missing value. The transformed data would be
save in the `/data_fill` dir

Here is the heuristic for handling missing data

- For the missing team performance stat, it would try to fill
  average team performance stat from the same season
- For other stats, take a global average (across all team and season)

"""

import pandas as pd
from datetime import datetime
import tqdm
import numpy as np

# column that we can safely dropped without lossing information
drop_col = [
    "home_team_season",  # can be infer base team and season
    "away_team_season",
]

# non-float coloumns. you can still treat some of the
# column as continous if you prefer (i.e. date)
COL_CATEGORICAL = [
    "id",
    "home_team_abbr",
    "away_team_abbr",
    "date",
    "season",
    "is_night_game",
    "home_team_win",
    "home_pitcher",
    "away_pitcher",
    "home_team_season",
    "away_team_season",
]

def load_data(csv_path, dummy_col=None, is_train=True):
    """
    load csv data. it will drop categorical col unless it is specified in the 'dummy_col'
    column specified in the dummy_col would be converted into indicator variable
    (see https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
    """
    drop_col = [
        "id",
        "season",
        "home_team_abbr",
        "away_team_abbr",
        "date",
        "is_night_game",
        "home_pitcher",
        "away_pitcher",
        "home_team_season",
        "away_team_season",
    ]

    if not is_train:
        drop_col.remove("date") # test set does not have this

    df = pd.read_csv(csv_path)

    # convert categorical into one-hot
    if dummy_col:
        df = pd.get_dummies(df, columns=dummy_col)

    df.drop(columns=drop_col, inplace=True, errors="ignore")

    feature_col = [_ for _ in df.columns.tolist() if _ != "home_team_win"]
    data_x = df[feature_col]

    if is_train:
        data_y = df[['home_team_win']].values.ravel()
        return data_x, data_y
    else:
        return data_x


def process_data(src, dest):
    print("processing: ", src)

    df = pd.read_csv(src)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")

    fill_season(df)
    fill_all_team_stat(df)
    fill_night_game(df)

    # fill rest of numerical column by mean
    df.fillna(df.select_dtypes(include="float").mean(), inplace=True)
    df.drop(columns=drop_col, inplace=True)
    df.to_csv(dest, index=False)


def fill_night_game(df):
    true_ratio = df["is_night_game"].mean()
    false_ratio = 1 - true_ratio
    df["boolean_column"] = df["boolean_column"].apply(
        lambda x: np.random.choice([True, False], p=[true_ratio, false_ratio])
        if pd.isna(x)
        else x
    )


def fill_season(df):
    def fill(row):
        if not pd.isna(row["season"]):
            return row["season"]

        if (
            "date" in row
            and isinstance(row["date"], datetime)
            and not pd.isna(row["date"])
        ):
            return row["date"].year

        if isinstance(row["home_team_season"], str):
            return int(row["home_team_season"][-4:])

        if isinstance(row["away_team_season"], str):
            return int(row["away_team_season"][-4:])

        return row["season"]

    df["season"] = df.apply(fill, axis=1)
    missing_count = df["season"].isna().sum()
    if missing_count > 0:
        print(f"fill_season: unable to infer {missing_count} records")


def fill_stat(df, stat_key):
    """
    Fill in missing numeric statistic base on team's season performance
        - You should apply this AFTER fill in the missing season value
        - You should NOT apply this for pitcher stat (i.e. 'home_pitching_H_batters_faced_mean')
          as it should be infer differently
        - Here we assume a team's home & away stat should be simialr
    """

    def fill(row):
        if not pd.isna(row[stat_key]):
            return row[stat_key]

        team_col = "away_team_abbr" if "away_" in stat_key else "home_team_abbr"
        team_abbr = row[team_col]
        season = row["season"]

        if "away_" in stat_key:
            team_stat = df[(df["away_team_abbr"] == team_abbr) & (df[stat_key].notna())]
        else:
            team_stat = df[(df["home_team_abbr"] == team_abbr) & (df[stat_key].notna())]

        season_stat = team_stat[team_stat["season"] == season]

        if len(season_stat) > 0:
            return season_stat[stat_key].mean()

        return row[stat_key]

    # missing_before = len(df[df[stat_key].isna()])
    df[stat_key] = df.apply(fill, axis=1)
    # missing_after = len(df[df[stat_key].isna()])
    # print(f"{stat_key} |  before: {missing_before}  | after: {missing_after}")


def fill_all_team_stat(df):
    float_col = list(df.select_dtypes(include="float").columns)
    exclude = ["season", "home_team_rest", "away_team_rest"]
    team_stat_col = [
        _
        for _ in float_col
        if "pitcher" not in _ and "pitching" not in _ and _ not in exclude
    ]

    for stat in tqdm.tqdm(team_stat_col, desc="fill_all_team_stat"):
        fill_stat(df, stat)


if __name__ == "__main__":
    process_data("data/task1/train_data.csv", "data_fill/train_data.csv")
    process_data("data/task1/same_season_test_data.csv", "data_fill/task1_test_data.csv")
    process_data("data/task2/2024_test_data.csv", "data_fill/task2_test_data.csv")

# %%
