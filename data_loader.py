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


def load_and_fill(path):
    print("loading", path)

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")

    fill_season(df)
    fill_all_team_stat(df)
    fill_night_game(df)

    # fill rest of numerical column by mean
    df.fillna(df.select_dtypes(include="float").mean(), inplace=True)
    df.drop(columns=drop_col, inplace=True)

    return df


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


if __name__ == "main":
    train_df = load_and_fill("data/task1/train_data.csv")
    test1_df = load_and_fill("data/task1/same_season_test_data.csv")
    test2_df = load_and_fill("data/task2/2024_test_data.csv")

    train_df.to_csv("data_fill/train_data.csv", index=False)
    test1_df.to_csv("data_fill/task1_test_data.csv", index=False)
    test2_df.to_csv("data_fill/task2_test_data.csv", index=False)

# pd.set_option("display.max_rows", None)
# df = pd.read_csv("data_fill/train_data.csv")
# missing_df = pd.DataFrame(
#     {
#         "type": df.dtypes,
#         "missing": df.isnull().any(),
#         "missing count": df.isnull().sum(),
#         "missing percentage": (df.isnull().sum() / len(df)) * 100,
#     }
# )
# missing_df.sort_values(by="missing percentage", ascending=False, inplace=True)
# missing_df
# df['is_night_game'].describe()

# non_float_columns = df.select_dtypes(exclude="float").columns.tolist()
# non_float_columns
