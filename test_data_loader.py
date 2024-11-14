import pytest
import pandas as pd
from datetime import datetime
from data_loader import fill_season

def test_fill_season():
    data = {
        'season': [2021, None, None, None, None],
        'date': [None, datetime(2022, 5, 17), None, None, None],
        'home_team_season': [None, None, 'Team_2019', None, None],
        'away_team_season': [None, None, None, "Team_2020", None]
    }
    df = pd.DataFrame(data)
    fill_season(df)
    assert df.loc[0, 'season'] == 2021
    assert df.loc[1, 'season'] == 2022
    assert df.loc[2, 'season'] == 2019
    assert df.loc[3, 'season'] == 2020
    assert pd.isna(df.loc[4, 'season'])