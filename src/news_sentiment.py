import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    pipeline,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from scipy.special import softmax
from newsapi import NewsApiClient
from news_sentiment import compute_daily_sentiment


# -- News Fetching -------------------------------------------------------------

def fetch_and_save_articles(api_key: str, from_date: str, to_date: str, out_csv: str) -> pd.DataFrame:
    """
    Fetch Tesla-related articles via NewsAPI month-by-month and save raw CSV.
    API key must be provided.
    """
    newsapi = NewsApiClient(api_key=api_key)
    current = pd.to_datetime(from_date)
    end = pd.to_datetime(to_date)
    delta = pd.Timedelta(days=15)
    all_articles = []

    while current < end:
        next_date = min(current + delta, end)
        from_str = current.strftime('%Y-%m-%d')
        to_str = next_date.strftime('%Y-%m-%d')
        try:
            resp = newsapi.get_everything(
                q='(tesla OR TSLA)',
                language='en',
                from_param=from_str,
                to=to_str,
                page_size=100,
                sort_by='relevancy'
            )
            for art in resp.get('articles', []):
                all_articles.append({
                    'date': art.get('publishedAt'),
                    'source': art.get('source', {}).get('name'),
                    'title': art.get('title'),
                    'content': art.get('content') or '',
                    'description': art.get('description') or '',
                    'url': art.get('url')
                })
        except Exception as e:
            print(f"Error fetching {from_str} to {to_str}: {e}")
        current = next_date
        time.sleep(1)

    df = pd.DataFrame(all_articles)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df.to_csv(out_csv, index=False)
    return df

# -- Summarization ------------------------------------------------------------

def chunk_text(text: str, max_chunk_len: int = 400):
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(current) >= max_chunk_len:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

class NewsSummarizer:
    def __init__(self, model_name="human-centered-summarization/financial-summarization-pegasus"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def summarize_two_ways(self, text: str) -> str:
        parts = []
        for chunk in chunk_text(text):
            prompt = f"Focus on Tesla stock news only: {chunk}"
            out = self.summarizer(prompt, max_length=100, min_length=30, do_sample=False)
            parts.append(out[0]['summary_text'])
        return " ".join(parts)

    def batch_summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        summaries = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing"):
            text = row['content'] or row['description'] or ''
            summaries.append({
                'date': row['date'],
                'tesla_summary': self.summarize_two_ways(text)
            })
        return pd.DataFrame(summaries)

# -- Sentiment Analysis ------------------------------------------------------

def add_prompt(text: str, focus="Tesla stock and market") -> str:
    return f"Analyze sentiment for {focus}: {text}" if text else ""

class SentimentAnalyzer:
    def __init__(self, model_name='ProsusAI/finbert'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def get_finbert_scores_batch(self, texts):
        inputs = self.tokenizer(
            [add_prompt(t) for t in texts],
            return_tensors='pt', max_length=512, truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()
        return softmax(logits, axis=1)

    def compute_daily_sentiment(self, df_sum: pd.DataFrame) -> pd.DataFrame:
        negs, neus, poss = [], [], []
        batch_size = 32
        texts = df_sum['tesla_summary'].tolist()
        for i in range(0, len(texts), batch_size):
            probs = self.get_finbert_scores_batch(texts[i:i+batch_size])
            negs.extend(probs[:,0]); neus.extend(probs[:,1]); poss.extend(probs[:,2])
        df_sum['neg'], df_sum['neu'], df_sum['pos'] = negs, neus, poss
        df_sum['net_sentiment'] = df_sum['pos'] - df_sum['neg']
        daily = df_sum.groupby('date')[['neg','neu','pos','net_sentiment']].mean().reset_index()
        daily.columns = ['Date','Average_Negative','Average_Neutral','Average_Positive','Average_Sentiment']
        daily['Date'] = pd.to_datetime(daily['Date'])
        return daily

def compute_daily_sentiment(raw_news_csv: str) -> pd.DataFrame:
    """
    1) Load raw news CSV (must have 'date','content','description').
    2) Summarize each article via NewsSummarizer.
    3) Compute daily sentiment via SentimentAnalyzer.
    """
    # 1) Read raw articles
    df_raw = pd.read_csv(raw_news_csv, parse_dates=['date'])
    # 2) Summarize
    summarizer = NewsSummarizer()
    df_sum = summarizer.batch_summarize(df_raw)
    # 3) Sentiment
    analyzer = SentimentAnalyzer()
    daily = analyzer.compute_daily_sentiment(df_sum)
    return daily
