#!/usr/bin/env python3
import argparse
import pandas as pd

from src.data_collection import collect_raw_data
from news_sentiment import compute_daily_sentiment

from src.preprocessing import (
    merge_stock_and_sentiment,
    adjust_friday_sentiments,
    add_holidays,
    fill_missing_dates
)
from src.features import generate_features

def main():
    parser = argparse.ArgumentParser(prog="main.py",
        description="End-to-end pipeline: collect → preprocess → feature-engineer")
    sub = parser.add_subparsers(dest="cmd")

    # ─── Collect ───
    p = sub.add_parser("collect", help="Download stock & index data")
    p.add_argument("--tickers", nargs="+", default=["TSLA"],
                   help="Tickers to fetch from Yahoo Finance")
    p.add_argument("--start-date", type=str, default="2017-09-30",
                   help="YYYY-MM-DD")
    p.add_argument("--news-csv", type=str, required=True,
                   help="Raw news CSV for sentiment")
    p.add_argument("--out-raw", type=str, default="data/raw.pkl")

    # ─── Preprocess ───
    p = sub.add_parser("preprocess", help="Merge & clean raw data")
    p.add_argument("--raw-pkl",    type=str, default="data/raw.pkl")
    p.add_argument("--out-pre",    type=str, default="data/preprocessed.pkl")
    p.add_argument("--start-date", type=str, default="2017-09-30",
                   help="(same start-date used in collect)")


    # ─── Features ───
    p = sub.add_parser("features", help="Compute all engineered features")
    p.add_argument("--pre-pkl", type=str, default="data/preprocessed.pkl")
    p.add_argument("--out-feat", type=str, default="data/features.pkl")

    args = parser.parse_args()

    if args.cmd == "collect":
        # 1) compute sentiment
        daily_sent = compute_daily_sentiment(args.news_csv)
        # 2) fetch raw stock & indices
        raw = collect_raw_data(
            tickers=args.tickers,
            start_date=args.start_date,
            sentiment_df=daily_sent
        )
        raw.to_pickle(args.out_raw)
        print(f"Saved raw data → {args.out_raw}")

    elif args.cmd == "preprocess":
        raw = pd.read_pickle(args.raw_pkl)
        merged = merge_stock_and_sentiment(raw)
        merged = adjust_friday_sentiments(merged)
        merged = add_holidays(merged, start=args.start_date, end=pd.Timestamp.today().strftime("%Y-%m-%d"))        
        merged = fill_missing_dates(merged)
        merged.to_pickle(args.out_pre)
        print(f"Saved preprocessed → {args.out_pre}")

    elif args.cmd == "features":
        pre = pd.read_pickle(args.pre_pkl)
        feats = generate_features(pre)
        feats.to_pickle(args.out_feat)
        print(f"Saved features → {args.out_feat}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
