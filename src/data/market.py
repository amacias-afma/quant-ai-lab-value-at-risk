import yfinance as yf
import pandas as pd
import pandas_gbq
import numpy as np

# from statsmodels.tsa.stattools import adfuller
# # from google.cloud import bigquery
import os
from scipy.stats import norm

from google.cloud import bigquery

def read_bigquery_data(ticker):
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
    client = bigquery.Client(project=PROJECT_ID)

    def load_data(ticker):
        """Fetches historical price data from BigQuery."""
        query = f"""
            SELECT date, price 
            FROM `{PROJECT_ID}.market_data.historical_prices`
            WHERE ticker = '{ticker}'
            ORDER BY date ASC
        """
        df = client.query(query).to_dataframe()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    
    df = load_data(ticker)
    print(f"Loaded {len(df)} rows for {ticker}")
    df.dropna(inplace=True)
    return df

def read_data(ticker="BTC-USD", market_data_source='yfinance', end_date=None):
    """
    1. Downloads data.
    2. Calculates Log Returns.
    3. Generates Features (Variance).
    4. Generates the 'Anchor' (Parametric GARCH/Normal VaR) for the hybrid loss.
    
    Args:
        ticker: Stock/crypto ticker symbol
        market_data_source: 'yfinance' or 'bigquery'
        end_date: End date for data fetch (format: 'YYYY-MM-DD'). If None, uses today.
                  Start date is automatically calculated as 10 years before end_date.
    """
    if market_data_source == 'yfinance':
        # 1. Ingestion
        # Calculate start_date as 10 years before end_date
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - pd.DateOffset(years=10)
        start_date = start_dt.strftime('%Y-%m-%d')
        
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
        df = pd.DataFrame(df[('Close', ticker)].values, columns=['price'], index=df.index)
    elif market_data_source == 'bigquery':
        df = read_bigquery_data(ticker)
    else:
        raise ValueError("Invalid market_data_source")
    return df.dropna()

if __name__ == "__main__":
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
    # For local test: PROJECT_ID = "your-project-id"
    
    ingestor = BatchMarketIngestor(PROJECT_ID)
    ingestor.run_pipeline()