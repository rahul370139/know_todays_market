import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, progress=False)
    df = df.reset_index().rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    })
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

def fetch_index_data(symbol: str, start_date: str, prefix: str) -> pd.DataFrame:
    """
    Download index data and prefix column names.
    """
    df = yf.download(symbol, start=start_date, progress=False)
    df = df.reset_index()
    df.columns = ["Date"] + [f"{prefix}_{col}" for col in ["Open","High","Low","Close","Volume"]]
    return df

def collect_raw_data(tickers, start_date, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Fetch each ticker
    2) Fetch S&P500 & NASDAQ
    3) Merge all + sentiment DataFrame on Date
    """
    # stock
    stock = pd.concat([
        fetch_stock_data(t, start_date).assign(Ticker=t)
        for t in tickers
    ], ignore_index=True)

    # indices
    sp = fetch_index_data("^GSPC", start_date, "SP500")
    ix = fetch_index_data("^IXIC", start_date, "NASDAQ")

    # merge
    df = stock.merge(sp, on="Date", how="left")\
              .merge(ix, on="Date", how="left")\
              .merge(sentiment_df, on="Date", how="left")

    # fill any missing sentiment
    for col in ["Average_Negative","Average_Neutral","Average_Positive","Average_Sentiment"]:
        if col in df:
            df[col] = df[col].fillna(0)
    return df
