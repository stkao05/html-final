import pytest
import pandas as pd
from datetime import datetime
from data_loader import fill_season, fill_stat
import numpy as np

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



def test_fill_stat():
    data = [
        {'season': 2023, 'home_team_abbr': 'A', 'away_team_abbr': 'B', 'home_stat_mean': 10, 'away_stat_mean': 8},
        {'season': 2023, 'home_team_abbr': 'A', 'away_team_abbr': 'C', 'home_stat_mean': None, 'away_stat_mean': 12},
        {'season': 2023, 'home_team_abbr': 'A', 'away_team_abbr': 'D', 'home_stat_mean': 20, 'away_stat_mean': None},
        {'season': 2023, 'home_team_abbr': 'B', 'away_team_abbr': 'A', 'home_stat_mean': 15, 'away_stat_mean': None},
        {'season': 2023, 'home_team_abbr': 'B', 'away_team_abbr': 'C', 'home_stat_mean': None, 'away_stat_mean': 18},
        {'season': 2023, 'home_team_abbr': 'C', 'away_team_abbr': 'A', 'home_stat_mean': None, 'away_stat_mean': 14},
        {'season': 2022, 'home_team_abbr': 'A', 'away_team_abbr': 'B', 'home_stat_mean': 9,  'away_stat_mean': 7},
        {'season': 2022, 'home_team_abbr': 'C', 'away_team_abbr': 'A', 'home_stat_mean': None, 'away_stat_mean': 11}
    ]
    
    df = pd.DataFrame(data)
    
    fill_stat(df, 'home_stat_mean')
    
    assert df.loc[0, 'home_stat_mean'] == 10
    assert df.loc[1, 'home_stat_mean'] == pytest.approx(np.mean([10, 20]))
    assert df.loc[2, 'home_stat_mean'] == 20
    assert df.loc[3, 'home_stat_mean'] == 15
    assert df.loc[4, 'home_stat_mean'] == pytest.approx(15)
    assert pd.isna(df.loc[5, 'home_stat_mean'])
    assert df.loc[6, 'home_stat_mean'] == 9
    assert pd.isna(df.loc[7, 'home_stat_mean'])

    fill_stat(df, 'away_stat_mean')
    
    assert df.loc[0, 'away_stat_mean'] == 8
    assert df.loc[1, 'away_stat_mean'] == 12
    assert pd.isna(df.loc[2, 'away_stat_mean'])
    assert df.loc[3, 'away_stat_mean'] == pytest.approx(np.mean([14]))
    assert df.loc[4, 'away_stat_mean'] == 18
    assert df.loc[5, 'away_stat_mean'] == 14