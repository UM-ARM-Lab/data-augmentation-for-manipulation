from typing import List

import pandas as pd


def df_where(df: pd.DataFrame, column_name, value) -> pd.DataFrame:
    return df.loc[df[column_name] == value]


def df_append(df: pd.DataFrame, row: List) -> pd.DataFrame:
    s = pd.Series(row, index=df.columns)
    df = df.append(s, ignore_index=True)
    return df

def rlast(rolling):
    return rolling.values[-1]
