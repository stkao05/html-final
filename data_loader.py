# %%
import pandas as pd
from datetime import datetime

def load_csv(path):
    df = pd.read_csv(path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors="raise")

    fill_season(df)
    return df


def fill_season(df):
    def fill(row):
        if not pd.isna(row['season']):
            return row['season']

        if 'date' in row and isinstance(row['date'], datetime) and not pd.isna(row['date']):
            return row['date'].year

        if isinstance(row['home_team_season'], str):
            return int(row['home_team_season'][-4:])

        if isinstance(row['away_team_season'], str):
            return int(row['away_team_season'][-4:])

        return row['season']

    df['season'] = df.apply(fill, axis=1)
    missing_count = df['season'].isna().sum()
    print(f"fill_season: unable to infer {missing_count} records")


# pd.set_option('display.max_columns', None)

# train_df = load_csv("data/task1/train_data.csv")
# test1_df = load_csv("data/task1/same_season_test_data.csv")
# test2_df = load_csv("data/task2/2024_test_data.csv")

# test2_df[['season', 'home_team_season', 'away_team_season']].head(10)