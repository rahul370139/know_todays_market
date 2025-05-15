import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

def merge_stock_and_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Already merged in data_collection; here you could re-index or rename if needed.
    """
    return df.copy()

def adjust_friday_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    daily = df.groupby("Date")[[
        "Average_Negative","Average_Neutral","Average_Positive","Average_Sentiment"
    ]].mean().reset_index()

    def adj(group):
        if group.name.weekday() != 4:
            return group
        # include weekend
        nxt = daily[daily["Date"].isin([
            group.name + pd.Timedelta(days=1),
            group.name + pd.Timedelta(days=2)
        ])]
        vals = pd.concat([group, nxt], axis=0).mean()
        return vals

    friday_adj = daily.groupby("Date").apply(adj).reset_index(drop=True)
    df = (
      df.drop(columns=[
        "Average_Negative","Average_Neutral","Average_Positive","Average_Sentiment"
      ])
      .merge(friday_adj, on="Date", how="left")
    )
    return df

def add_holidays(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Tag NYSE holidays.
    """
    nyse = mcal.get_calendar("NYSE")
    hols = nyse.holidays().holidays
    df = df.copy()
    df["is_holiday"] = pd.to_datetime(df["Date"]).isin(hols).astype(int)
    return df

def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to daily frequency and forward-fill.
    """
    df = df.copy().set_index("Date")
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx).ffill().reset_index()
    df = df.rename(columns={"index":"Date"})
    return df

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix and drop columns with corr > 0.97.
    """
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.97)]
    return df.drop(columns=to_drop)