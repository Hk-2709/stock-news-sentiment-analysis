import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import pearsonr
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from .env
API_KEY = os.getenv("POLYGON_API_KEY")

if not API_KEY:
    raise ValueError(
        "❌ POLYGON_API_KEY not found!\n"
        "1. Create a file called .env in this folder\n"
        "2. Add this line inside it:\n"
        "   POLYGON_API_KEY=your_actual_key_here\n"
        "3. Never commit .env to GitHub!"
    )

# ====================== CONFIG ======================
tickers = ['AAPL', 'TSLA', 'NVDA']
start_date = '2025-09-01'      # Adjust if needed
end_date   = '2026-02-25'      # Adjust if needed
# ===================================================

client = RESTClient(api_key=API_KEY)           # ← fixed: use API_KEY here
analyzer = SentimentIntensityAnalyzer()

print("🚀 Starting News Sentiment vs Stock Returns Analysis...\n")

# ------------------- 1. Fetch Daily Prices -------------------
price_dfs = {}
for ticker in tickers:
    aggs = list(client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start_date,
        to=end_date,
        limit=50000
    ))
    df = pd.DataFrame([{
        'date': pd.to_datetime(a.timestamp, unit='ms').date(),
        'close': a.close
    } for a in aggs])
    df = df.drop_duplicates('date').set_index('date').sort_index()
    df['daily_return'] = df['close'].pct_change() * 100   # in percent
    price_dfs[ticker] = df
    print(f"✅ {ticker}: {len(df)} trading days fetched")

# ------------------- 2. Fetch Headlines + VADER Sentiment -------------------
news_list = []
for ticker in tickers:
    articles = list(client.list_ticker_news(
        ticker=ticker,
        published_utc_gte=start_date,
        published_utc_lte=end_date,
        limit=1000
    ))
    for article in articles:
        pub_date = article.published_utc.date() if hasattr(article.published_utc, 'date') else pd.to_datetime(article.published_utc).date()
        text = (article.title or "") + " " + (getattr(article, 'description', "") or "")
        sentiment = analyzer.polarity_scores(text)['compound']   # -1 to +1
        
        news_list.append({
            'date': pub_date,
            'ticker': ticker,
            'title': article.title,
            'sentiment': sentiment
        })
    print(f"✅ {ticker}: {len(articles)} headlines analyzed")

news_df = pd.DataFrame(news_list)
daily_sentiment = news_df.groupby(['date', 'ticker'])['sentiment'].mean().reset_index()

# ------------------- 3. Merge & Calculate Correlation -------------------
results = []
for ticker in tickers:
    price = price_dfs[ticker]
    sent = daily_sentiment[daily_sentiment['ticker'] == ticker].set_index('date')['sentiment']
    
    combined = price.join(sent, how='left').fillna(0)
    combined['ticker'] = ticker
    combined = combined.dropna(subset=['daily_return'])
    
    if len(combined) > 10:
        corr, pval = pearsonr(combined['sentiment'], combined['daily_return'])
        results.append({
            'ticker': ticker,
            'correlation': corr,
            'p_value': pval,
            'num_days': len(combined),
            'avg_sentiment': combined['sentiment'].mean()
        })
    
    # Save per-ticker data
    combined.to_csv(f'{ticker}_daily_data.csv')

results_df = pd.DataFrame(results)
print("\n📊 RESULTS")
print(results_df.round(4))

# ------------------- 4. Visualizations -------------------
sns.set_style("whitegrid")

for ticker in tickers:
    data = pd.read_csv(f'{ticker}_daily_data.csv', parse_dates=['date'])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Price + Sentiment overlay
    ax1 = axes[0]
    ax1.plot(data['date'], data['close'], color='#1f77b4', linewidth=2.5, label='Close Price')
    ax1.set_ylabel('Price ($)', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    ax2 = ax1.twinx()
    ax2.plot(data['date'], data['sentiment'], color='#ff7f0e', alpha=0.85, linewidth=2, label='News Sentiment')
    ax2.set_ylabel('Sentiment Score (-1 to +1)', color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    ax1.set_title(f'{ticker} — Stock Price vs News Sentiment', fontsize=16, pad=20)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Scatter: Sentiment vs Return
    axes[1].scatter(data['sentiment'], data['daily_return'], alpha=0.7, s=50, color='#2ca02c')
    axes[1].set_xlabel('Daily Average Sentiment', fontsize=12)
    axes[1].set_ylabel('Daily Return (%)', fontsize=12)
    axes[1].set_title(f'Sentiment vs Daily Returns (Correlation = {results_df[results_df["ticker"]==ticker]["correlation"].values[0]:.3f})')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_sentiment_vs_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n🎉 Analysis complete! Check the generated PNG files and CSV files.")