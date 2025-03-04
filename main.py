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
    print("Starting to load historical data")
    try:
        # Step 1: Open and verify zip contents
        with zipfile.ZipFile(filename) as z:
            print(f"Zip file opened, contents: {z.namelist()}")
            csv_filename = z.namelist()[0]
            file_size = z.getinfo(csv_filename).file_size
            print(f"CSV file size in zip: {file_size} bytes")
            
            # Step 2: Read in smaller blocks
            with z.open(csv_filename) as f:
                print("Opening CSV file")
                buffer_size = 1024 * 1024  # 1MB chunks
                data = bytearray()
                while True:
                    print(f"Reading block, current size: {len(data)} bytes")
                    block = f.read(buffer_size)
                    if not block:
                        break
                    data.extend(block)
                
                print(f"Total data read: {len(data)} bytes")
                
                # Step 3: Process the data
                df = pd.read_csv(io.BytesIO(data), chunksize=5000)
                chunks = []
                for i, chunk in enumerate(df):
                    chunks.append(chunk)
                    print(f"Processed chunk {i+1}")
                
                final_df = pd.concat(chunks)
                print("DataFrame creation complete")
                
        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
        final_df.set_index('timestamp', inplace=True)
        final_df = final_df.dropna()
        
        print(f"Final data shape: {final_df.shape}")
        return final_df
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        print(f"Error type: {type(e)}")
        raise

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
        print("Loading historical data...")
        historical_data = load_historical_data()
        print("Processing data...")
        processed_data = validate_and_prepare_data(historical_data)
        print("Initializing trading system...")
        trading_system = TradingSystem(config, processed_data)
        print("Starting trading system...")
        await trading_system.start()
        print("Entering main loop...")
        await trading_system.main_loop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)  # Added exc_info=True for stack trace


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())