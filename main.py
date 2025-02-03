import time
import pandas as pd
from config import Config
import zipfile
import io
from trading_system import TradingSystem
import asyncio
import decimal as Decimal
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_historical_data(filename='historical_data.csv.zip'):
    with zipfile.ZipFile(filename) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Add validation
    print(f"Data shape: {df.shape}")
    print(f"Column lengths: {df.count()}")
    
    # Ensure no missing data
    df = df.dropna()
    return df

def validate_and_prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f"Initial data shape: {data.shape}")
    logger.debug(f"Columns present: {data.columns.tolist()}")
    
    # Ensure all required columns exist with consistent lengths
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    data = data.reindex(columns=required_columns)
    
    # Remove any rows with missing values
    data = data.dropna()
    
    # Ensure all columns have numeric data
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    logger.debug(f"Processed data shape: {data.shape}")
    return data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    try:
        config = Config()
        print(f"Main config BASE_PARAMS: {hasattr(config, 'BASE_PARAMS')}")
        historical_data = load_historical_data()
        processed_data = validate_and_prepare_data(historical_data)
        trading_system = TradingSystem(config, processed_data)
        await trading_system.start()
        await trading_system.main_loop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())