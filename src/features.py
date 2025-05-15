import numpy as np
import pandas as pd
from arch import arch_model

def cap_outliers(col: pd.Series) -> pd.Series:
    Q1, Q3 = np.percentile(col, [25,75])
    IQR = Q3 - Q1
    return col.clip(Q1 - 2*IQR, Q3 + 2*IQR)

def compute_rsi(series: pd.Series, window=14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta>0,0).rolling(window).mean()
    loss = -delta.where(delta<0,0).rolling(window).mean()
    rs = gain/(loss+1e-9)
    return 100 - (100/(1+rs))

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    dir = np.sign(close.diff())
    return (volume * dir).fillna(0).cumsum()

def compute_macd(close: pd.Series, short=12, long=26, signal=9):
    e1 = close.ewm(span=short).mean()
    e2 = close.ewm(span=long).mean()
    macd = e1 - e2
    sig = macd.ewm(span=signal).mean()
    return macd, sig

def compute_bollinger_bands(close: pd.Series, window=20):
    m = close.rolling(window).mean()
    s = close.rolling(window).std()
    return m+2*s, m, m-2*s

def compute_stochastic_oscillator(h, l, c, window=14):
    hh = h.rolling(window).max()
    ll = l.rolling(window).min()
    k = 100*(c-ll)/(hh-ll+1e-9)
    d = k.rolling(3).mean()
    return k, d

def compute_atr(h, l, c, window=14):
    tr = pd.concat([
        h-l,
        (h-c.shift()).abs(),
        (l-c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_chaikin(c, h, l, v):
    mf = ((c-l)-(h-c))/(h-l+1e-9)*v
    return mf.rolling(3).sum() - mf.rolling(10).sum()

def compute_williams_r(h, l, c, window=14):
    hh = h.rolling(window).max()
    ll = l.rolling(window).min()
    return -100*(hh-c)/(hh-ll+1e-9)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # rolling stats
    df["Returns"] = df["Close"].pct_change()
    df["MA7_Close"] = df["Close"].rolling(7).mean()
    df["Volatility_7d"] = df["Close"].rolling(7).std()
    df["MA20_Close"] = df["Close"].rolling(20).mean()

    # indicators
    df["RSI"] = compute_rsi(df["Close"])
    df["OBV"] = compute_obv(df["Close"], df["Volume"])
    df["MACD"], df["MACD_Signal"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = compute_bollinger_bands(df["Close"])
    df["ATR"] = compute_atr(df["High"], df["Low"], df["Close"])
    df["Chaikin"] = compute_chaikin(df["Close"], df["High"], df["Low"], df["Volume"])
    df["Williams_%R"] = compute_williams_r(df["High"], df["Low"], df["Close"])
    df["Stoch_K"], df["Stoch_D"] = compute_stochastic_oscillator(
        df["High"], df["Low"], df["Close"]
    )

    # price lags & diffs
    for lag in [1,2]:
        for col in ["Open","High","Low","Close"]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for col in ["Open","High","Low","Close"]:
        df[f"{col}_diff"] = df[col].diff()

    # week slopes
    for col in ["Open","High","Low","Close"]:
        df[f"{col}_week_slope"] = (
            df[col].rolling(7)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        )

    # GARCH volatility
    ret = df["Returns"].dropna()
    if len(ret)>10:
        gm = arch_model(ret, p=1, q=1).fit(disp="off")
        df["GARCH_Volatility"] = gm.conditional_volatility

    # next-day targets
    df["High_next1"] = df["High"].shift(-1)
    df["Low_next1"]  = df["Low"].shift(-1)
    df["Close_next1"] = df["Close"].shift(-1)
    df["Open_next1"] = df["Open"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df
