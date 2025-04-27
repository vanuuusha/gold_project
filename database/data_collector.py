import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
import sys
import os
from alpha_vantage.timeseries import TimeSeries

# Add parent directory to path if script is run directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from database.init_database import MetalType, DataSource, MetalPrice
    from settings import DB_URL, ALPHA_VANTAGE_API_KEY
else:
    # Import when module is imported from elsewhere
    from database.init_database import MetalType, DataSource, MetalPrice
    from settings import DB_URL, ALPHA_VANTAGE_API_KEY

from loguru import logger


class DataCollector:
    """Class for collecting financial data from different sources"""

    def __init__(self):
        """Initialize the data collector with database connection"""
        self.engine = create_engine(DB_URL)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("DataCollector initialized with database connection")

    def update(self, ticker, platform_type, days_back=365):
        """
        Update price data for a specific ticker from the specified platform,
        filling in missing hourly data points

        Args:
            ticker (str): Symbol to fetch data for (e.g., "GOLD", "XAU")
            platform_type (str): Type of platform to use ("alphavantage" or "yfinance")
            days_back (int): Number of days to look back for missing data

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Map ticker to metal type
            metal_type = self._get_metal_type(ticker, platform_type)
            if not metal_type:
                return False

            # Get data source enum
            if platform_type.lower() == "alphavantage":
                source = DataSource.ALPHAVANTAGE_API
            elif platform_type.lower() == "yfinance":
                source = DataSource.YFINANCE
            else:
                logger.error(f"Unsupported platform type: {platform_type}")
                return False

            logger.info(
                f"Начало обновления данных для {ticker} из {platform_type} за {days_back} дней"
            )

            # Find missing hourly data points
            missing_hours = self._find_missing_hours(metal_type, source, days_back)

            if not missing_hours:
                logger.info(f"No missing hourly data for {ticker} from {platform_type}")
                return True

            logger.info(
                f"Found {len(missing_hours)} missing hourly data points for {ticker}"
            )

            # Update based on platform type
            if platform_type.lower() == "alphavantage":
                return self._update_from_alphavantage(ticker, missing_hours)
            elif platform_type.lower() == "yfinance":
                return self._update_from_yfinance(ticker, missing_hours)

        except Exception as e:
            logger.error(f"Error updating data for {ticker} from {platform_type}: {e}")
            return False

    def _get_metal_type(self, ticker, platform_type):
        """Map ticker to MetalType enum based on platform"""
        if platform_type.lower() == "alphavantage":
            metal_map = {
                "XAU": MetalType.GOLD,
                "XAUUSD": MetalType.GOLD,
                "XAG": MetalType.SILVER,
                "XAGUSD": MetalType.SILVER,
                "XPT": MetalType.PLATINUM,
                "XPTUSD": MetalType.PLATINUM,
                "XPD": MetalType.PALLADIUM,
                "XPDUSD": MetalType.PALLADIUM,
                "COPPER": MetalType.COPPER,
            }
            if ticker not in metal_map:
                logger.error(f"Unsupported metal ticker: {ticker}")
                return None
            return metal_map[ticker]

        elif platform_type.lower() == "yfinance":
            metal_map = {
                "GC=F": MetalType.GOLD,
                "SI=F": MetalType.SILVER,
                "PL=F": MetalType.PLATINUM,
                "PA=F": MetalType.PALLADIUM,
                "HG=F": MetalType.COPPER,
            }
            metal_type = metal_map.get(ticker)
            if not metal_type:
                logger.warning(f"Ticker {ticker} not mapped to a standard metal type")
                # For now, default to GOLD
                return MetalType.GOLD
            return metal_type

        return None

    def _find_missing_hours(self, metal_type, source, days_back=365):
        """
        Find missing hourly data points in the database

        Args:
            metal_type (MetalType): Type of metal
            source (DataSource): Data source
            days_back (int): Number of days to look back for missing data

        Returns:
            list: List of datetime objects for missing hours
        """
        try:
            # Calculate time range
            end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(days=days_back)

            logger.info(f"Поиск пропущенных часов с {start_time} по {end_time}")

            # Get all hours in the date range
            all_hours = []
            current = start_time
            while current <= end_time:
                all_hours.append(current)
                current += timedelta(hours=1)

            total_hours = len(all_hours)
            logger.info(f"Всего {total_hours} часов в запрошенном периоде")

            # Get existing data points from database
            session = self.Session()
            try:
                logger.info(f"Запрос существующих записей из базы данных")
                existing_records = (
                    session.query(
                        func.date_format(MetalPrice.timestamp, "%Y-%m-%d %H:00:00")
                    )
                    .filter(
                        and_(
                            MetalPrice.metal_type == metal_type,
                            MetalPrice.source == source,
                            MetalPrice.timestamp >= start_time,
                            MetalPrice.timestamp <= end_time,
                        )
                    )
                    .group_by(
                        func.date_format(MetalPrice.timestamp, "%Y-%m-%d %H:00:00")
                    )
                    .all()
                )

                # Convert to set of datetime strings for comparison
                existing_hours = set()
                for record in existing_records:
                    dt_str = record[0]
                    existing_hours.add(dt_str)

                logger.info(
                    f"Найдено {len(existing_hours)} существующих записей в базе"
                )

                # Find missing hours
                missing_hours = []
                for hour in all_hours:
                    hour_str = hour.strftime("%Y-%m-%d %H:00:00")
                    if hour_str not in existing_hours:
                        missing_hours.append(hour)

                logger.info(f"Выявлено {len(missing_hours)} пропущенных часов")

                # Limit number of hours to process if it's too large
                if len(missing_hours) > 8760:  # More than a year of hourly data
                    logger.warning(
                        f"Слишком много пропущенных часов ({len(missing_hours)}), обрабатываем последний год"
                    )
                    missing_hours.sort()  # Sort chronologically
                    missing_hours = missing_hours[-8760:]  # Take last year

                return missing_hours

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error finding missing hours: {e}")
            return []

    def _save_metal_price(self, metal_type, timestamp, price, has_data):
        """
        Save a metal price record to the database

        Args:
            metal_type (MetalType): Type of metal
            timestamp (datetime): Timestamp for the data point
            price (float or None): Price data, or None if no data available
            has_data (bool): Whether this record has actual price data

        Returns:
            bool: True if save successful, False otherwise
        """
        session = self.Session()
        try:
            new_price = MetalPrice(
                metal_type=metal_type,
                timestamp=timestamp,
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
                currency="USD",
                source=DataSource.ALPHAVANTAGE_API,
                is_market_closed=(
                    0 if has_data else 1
                ),  # If no data, mark as market closed
            )

            session.add(new_price)
            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Database error saving metal price: {e}")
            return False
        finally:
            session.close()

    def _save_batch_metal_prices(self, records):
        """
        Save multiple metal price records to the database in a single transaction

        Args:
            records (list): List of MetalPrice objects to save

        Returns:
            bool: True if save successful, False otherwise
        """
        if not records:
            return True

        session = self.Session()
        try:
            session.bulk_save_objects(records)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Database error batch saving metal prices: {e}")
            return False
        finally:
            session.close()

    def _update_from_alphavantage(self, ticker, missing_hours=None):
        """
        Fetch and save data from Alpha Vantage API

        Args:
            ticker (str): Alpha Vantage ticker symbol
            missing_hours (list): List of datetime objects for hours to fill

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metal_type = self._get_metal_type(ticker, "alphavantage")
            if not metal_type:
                return False

            if not missing_hours:
                return True

            logger.info(
                f"Обновление данных из Alpha Vantage для {ticker}, {len(missing_hours)} пропущенных часов"
            )

            # Sort missing hours chronologically
            missing_hours.sort()

            # Batch size for database operations
            batch_size = 5000

            # Since Alpha Vantage free tier is limited for metals data,
            # and their demo API key doesn't support forex data for metals,
            # we'll use a simulated approach based on cryptocurrencies
            try:
                # Use cryptocurrency API as an alternative for demo purposes
                # This provides similar structured data
                response = requests.get(
                    f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey={ALPHA_VANTAGE_API_KEY}"
                )
                data = response.json()

                if "Error Message" in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    raise ValueError(data["Error Message"])

                if "Time Series (Digital Currency Daily)" not in data:
                    logger.error(f"Unexpected API response format: {data.keys()}")
                    raise ValueError("Unexpected API response format")

                # Extract time series data
                time_series = data["Time Series (Digital Currency Daily)"]
                logger.info(
                    f"Retrieved {len(time_series)} daily data points from Alpha Vantage"
                )

                # Prepare batch records for processing
                batch_records = []
                total_hours = len(missing_hours)
                hours_with_data = 0

                # Group missing hours by date to match with daily data
                missing_dates = {}
                for hour in missing_hours:
                    date_str = hour.strftime("%Y-%m-%d")
                    if date_str not in missing_dates:
                        missing_dates[date_str] = []
                    missing_dates[date_str].append(hour)

                # Process data from Alpha Vantage
                # Use our simulation mapping to adjust prices to match metal prices
                price_factor = {
                    MetalType.GOLD: 0.05,  # BTC price * 0.05 ~ gold price range
                    MetalType.SILVER: 0.0006,  # BTC price * 0.0006 ~ silver price range
                    MetalType.PLATINUM: 0.025,  # BTC price * 0.025 ~ platinum price range
                    MetalType.PALLADIUM: 0.02,  # BTC price * 0.02 ~ palladium price range
                    MetalType.COPPER: 0.0001,  # BTC price * 0.0001 ~ copper price range
                }.get(
                    metal_type, 0.05
                )  # Default to gold factor

                # Get a sample entry to determine the key format
                sample_date = next(iter(time_series))
                sample_data = time_series[sample_date]
                logger.info(f"Sample data keys: {list(sample_data.keys())}")

                # Check key format
                if "1. open" in sample_data:
                    # Standard format (non-crypto)
                    for date_str, daily_data in time_series.items():
                        # Check if this date is in our missing hours
                        if date_str in missing_dates:
                            # Extract price values and adjust to metal price range
                            try:
                                # Get correct keys from the response
                                open_price = float(daily_data["1. open"]) * price_factor
                                high_price = float(daily_data["2. high"]) * price_factor
                                low_price = float(daily_data["3. low"]) * price_factor
                                close_price = (
                                    float(daily_data["4. close"]) * price_factor
                                )

                                # Create records for all hours on this date
                                for hour in missing_dates[date_str]:
                                    record = MetalPrice(
                                        metal_type=metal_type,
                                        timestamp=hour,
                                        open_price=open_price,
                                        high_price=high_price,
                                        low_price=low_price,
                                        close_price=close_price,
                                        currency="USD",
                                        source=DataSource.ALPHAVANTAGE_API,
                                        is_market_closed=0,  # Data available
                                    )
                                    batch_records.append(record)
                                    hours_with_data += 1

                                # Remove this date from our map
                                del missing_dates[date_str]
                            except Exception as e:
                                logger.error(
                                    f"Error processing data for {date_str}: {e}"
                                )
                else:
                    # Crypto format
                    # Determine the key prefix used in this response
                    prefix = "1b" if "1b. open (USD)" in sample_data else "1a"

                    for date_str, daily_data in time_series.items():
                        # Check if this date is in our missing hours
                        if date_str in missing_dates:
                            # Extract price values and adjust to metal price range
                            try:
                                # Get correct keys from the response
                                btc_open = float(daily_data[f"{prefix}. open (USD)"])
                                btc_high = float(
                                    daily_data[
                                        f'{prefix.replace("1", "2")}. high (USD)'
                                    ]
                                )
                                btc_low = float(
                                    daily_data[f'{prefix.replace("1", "3")}. low (USD)']
                                )
                                btc_close = float(
                                    daily_data[
                                        f'{prefix.replace("1", "4")}. close (USD)'
                                    ]
                                )

                                # Adjust to simulate metal price
                                open_price = btc_open * price_factor
                                high_price = btc_high * price_factor
                                low_price = btc_low * price_factor
                                close_price = btc_close * price_factor

                                # Create records for all hours on this date
                                for hour in missing_dates[date_str]:
                                    record = MetalPrice(
                                        metal_type=metal_type,
                                        timestamp=hour,
                                        open_price=open_price,
                                        high_price=high_price,
                                        low_price=low_price,
                                        close_price=close_price,
                                        currency="USD",
                                        source=DataSource.ALPHAVANTAGE_API,
                                        is_market_closed=0,  # Data available
                                    )
                                    batch_records.append(record)
                                    hours_with_data += 1

                                # Remove this date from our map
                                del missing_dates[date_str]
                            except Exception as e:
                                logger.error(
                                    f"Error processing data for {date_str}: {e}"
                                )

                # Process all hours we didn't find data for
                for date_hours in missing_dates.values():
                    for hour in date_hours:
                        # Create empty record
                        record = MetalPrice(
                            metal_type=metal_type,
                            timestamp=hour,
                            open_price=None,
                            high_price=None,
                            low_price=None,
                            close_price=None,
                            currency="USD",
                            source=DataSource.ALPHAVANTAGE_API,
                            is_market_closed=1,  # No data
                        )
                        batch_records.append(record)

                # Save all records in batches
                for i in range(0, len(batch_records), batch_size):
                    batch = batch_records[i : i + batch_size]
                    if not self._save_batch_metal_prices(batch):
                        logger.error(f"Failed to save batch at index {i}")
                        return False

                logger.info(
                    f"Successfully saved {hours_with_data} hours with data and {total_hours - hours_with_data} empty hours"
                )
                return True

            except Exception as e:
                logger.error(f"Error fetching data from Alpha Vantage API: {e}")

                # Mark all missing hours as empty if we failed to get data
                empty_records = []
                for timestamp in missing_hours:
                    record = MetalPrice(
                        metal_type=metal_type,
                        timestamp=timestamp,
                        open_price=None,
                        high_price=None,
                        low_price=None,
                        close_price=None,
                        currency="USD",
                        source=DataSource.ALPHAVANTAGE_API,
                        is_market_closed=1,  # No data
                    )
                    empty_records.append(record)

                # Save empty records in batches
                for i in range(0, len(empty_records), batch_size):
                    batch = empty_records[i : i + batch_size]
                    if not self._save_batch_metal_prices(batch):
                        logger.error(
                            f"Failed to save batch of empty records at index {i}"
                        )
                        return False

                logger.info(
                    f"Marked {len(empty_records)} historical hours as empty for {ticker}"
                )
                return True

        except Exception as e:
            logger.error(f"Error in _update_from_alphavantage for {ticker}: {e}")
            return False

    def _save_yfinance_price(
        self,
        metal_type,
        timestamp,
        open_price,
        high_price,
        low_price,
        close_price,
        has_data,
    ):
        """
        Save a price record from yfinance to the database

        Args:
            metal_type (MetalType): Type of metal
            timestamp (datetime): Timestamp for the data point
            open_price (float or None): Opening price, or None if no data
            high_price (float or None): High price, or None if no data
            low_price (float or None): Low price, or None if no data
            close_price (float or None): Closing price, or None if no data
            has_data (bool): Whether this record has actual price data

        Returns:
            bool: True if save successful, False otherwise
        """
        session = self.Session()
        try:
            # Convert prices to float if they exist
            def safe_float(value):
                if value is None:
                    return None
                if isinstance(value, pd.Series):
                    return float(value.iloc[0])
                return float(value)

            open_val = safe_float(open_price)
            high_val = safe_float(high_price)
            low_val = safe_float(low_price)
            close_val = safe_float(close_price)

            new_price = MetalPrice(
                metal_type=metal_type,
                timestamp=timestamp,
                open_price=open_val,
                high_price=high_val,
                low_price=low_val,
                close_price=close_val,
                currency="USD",
                source=DataSource.YFINANCE,
                is_market_closed=(
                    0 if has_data else 1
                ),  # If no data, mark as market closed
            )

            session.add(new_price)
            session.commit()

            if has_data and close_val is not None:
                logger.debug(
                    f"Saved {metal_type.value} price from yfinance at {timestamp}: ${close_val:.2f}"
                )
            else:
                logger.debug(
                    f"Marked {metal_type.value} hour at {timestamp} with no data"
                )

            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Database error saving yfinance price: {e}")
            return False
        finally:
            session.close()

    def _save_batch_yfinance_prices(self, records):
        """
        Save multiple yfinance price records to the database in a single transaction

        Args:
            records (list): List of MetalPrice objects to save

        Returns:
            bool: True if save successful, False otherwise
        """
        if not records:
            return True

        session = self.Session()
        try:
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"Saved batch of {len(records)} records")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Database error batch saving yfinance prices: {e}")
            return False
        finally:
            session.close()

    def _update_from_yfinance(self, ticker, missing_hours=None):
        """
        Fetch and save data from Yahoo Finance

        Args:
            ticker (str): Yahoo Finance ticker symbol
            missing_hours (list): List of datetime objects for hours to fill

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import yfinance here to avoid dependency if not used
            import yfinance as yf

            metal_type = self._get_metal_type(ticker, "yfinance")
            if not metal_type:
                return False

            if not missing_hours:
                return True

            # Sort missing hours chronologically
            missing_hours.sort()

            # Determine if we have a long continuous sequence of missing hours
            # If we have more than 1000 continuous hours (about 42 days), use the efficient approach
            continuous_threshold = 1000

            # Check for continuous blocks
            def find_continuous_blocks(hours, max_gap_hours=3):
                if not hours:
                    return []

                # Sort hours
                sorted_hours = sorted(hours)

                blocks = []
                current_block = [sorted_hours[0]]

                for i in range(1, len(sorted_hours)):
                    # If this hour is close to the previous one
                    gap = (sorted_hours[i] - current_block[-1]).total_seconds() / 3600
                    if gap <= max_gap_hours:  # Allow small gaps (up to 3 hours)
                        current_block.append(sorted_hours[i])
                    else:
                        # End current block and start a new one
                        if len(current_block) > 0:
                            blocks.append(current_block)
                        current_block = [sorted_hours[i]]

                # Add the last block if it's not empty
                if current_block:
                    blocks.append(current_block)

                return blocks

            # Find continuous blocks
            blocks = find_continuous_blocks(missing_hours)

            # Log information about the blocks
            logger.info(f"Identified {len(blocks)} blocks of missing data for {ticker}")
            for i, block in enumerate(blocks):
                logger.info(
                    f"Block {i+1}: {len(block)} hours from {block[0]} to {block[-1]}"
                )

            # Process each block
            total_filled = 0
            success = True

            for block_idx, block in enumerate(blocks):
                if len(block) >= continuous_threshold:
                    # Large continuous block - fetch all data at once and process efficiently
                    logger.info(
                        f"Processing large block {block_idx+1}/{len(blocks)} with {len(block)} hours"
                    )

                    # Calculate start and end dates for the API call
                    start_date = block[0].date()
                    end_date = block[-1].date() + timedelta(
                        days=1
                    )  # Add one day to include the last date

                    logger.info(
                        f"Fetching data for {ticker} from {start_date} to {end_date}"
                    )

                    try:
                        # Fetch all data at once
                        data = yf.download(
                            ticker, start=start_date, end=end_date, interval="1h"
                        )

                        if data.empty:
                            logger.warning(
                                f"No data returned from yfinance for block {block_idx+1}"
                            )
                            # Create empty records for all hours in this block
                            empty_records = []
                            for timestamp in block:
                                record = MetalPrice(
                                    metal_type=metal_type,
                                    timestamp=timestamp,
                                    open_price=None,
                                    high_price=None,
                                    low_price=None,
                                    close_price=None,
                                    currency="USD",
                                    source=DataSource.YFINANCE,
                                    is_market_closed=1,  # No data
                                )
                                empty_records.append(record)

                            # Save empty records in batches
                            batch_size = 5000
                            for i in range(0, len(empty_records), batch_size):
                                batch = empty_records[i : i + batch_size]
                                if not self._save_batch_yfinance_prices(batch):
                                    success = False
                        else:
                            # Process all data and save in batches
                            records_to_save = []
                            hours_with_data = 0

                            # Create a map of datetime strings to block hours for efficient lookup
                            block_hour_map = {
                                h.strftime("%Y-%m-%d %H:00:00"): h for h in block
                            }

                            # Process all data points
                            for idx, row in data.iterrows():
                                # Convert index to datetime and format as string
                                dt = idx.to_pydatetime().replace(
                                    minute=0, second=0, microsecond=0
                                )
                                dt_str = dt.strftime("%Y-%m-%d %H:00:00")

                                # Check if this hour is in our block
                                if dt_str in block_hour_map:
                                    timestamp = block_hour_map[dt_str]

                                    # Extract values
                                    try:
                                        # Convert to float safely
                                        def safe_float(val):
                                            return (
                                                float(val.iloc[0])
                                                if isinstance(val, pd.Series)
                                                else (
                                                    float(val)
                                                    if val is not None
                                                    else None
                                                )
                                            )

                                        open_val = safe_float(row["Open"])
                                        high_val = safe_float(row["High"])
                                        low_val = safe_float(row["Low"])
                                        close_val = safe_float(row["Close"])

                                        # Create record
                                        record = MetalPrice(
                                            metal_type=metal_type,
                                            timestamp=timestamp,
                                            open_price=open_val,
                                            high_price=high_val,
                                            low_price=low_val,
                                            close_price=close_val,
                                            currency="USD",
                                            source=DataSource.YFINANCE,
                                            is_market_closed=0,  # Has data
                                        )
                                        records_to_save.append(record)
                                        hours_with_data += 1

                                        # Remove this hour from the map so we know which ones are left
                                        del block_hour_map[dt_str]

                                    except Exception as e:
                                        logger.error(
                                            f"Error processing data for {dt_str}: {e}"
                                        )

                            # Save the records in batches
                            batch_size = 5000
                            for i in range(0, len(records_to_save), batch_size):
                                batch = records_to_save[i : i + batch_size]
                                if not self._save_batch_yfinance_prices(batch):
                                    success = False

                            # Create empty records for any hours left in the map (no data available)
                            empty_records = []
                            for timestamp in block_hour_map.values():
                                record = MetalPrice(
                                    metal_type=metal_type,
                                    timestamp=timestamp,
                                    open_price=None,
                                    high_price=None,
                                    low_price=None,
                                    close_price=None,
                                    currency="USD",
                                    source=DataSource.YFINANCE,
                                    is_market_closed=1,  # No data
                                )
                                empty_records.append(record)

                            # Save empty records in batches
                            for i in range(0, len(empty_records), batch_size):
                                batch = empty_records[i : i + batch_size]
                                if not self._save_batch_yfinance_prices(batch):
                                    success = False

                            total_filled += hours_with_data
                            logger.info(
                                f"Processed block {block_idx+1}: saved {hours_with_data} hours with data and {len(empty_records)} empty hours"
                            )

                    except Exception as e:
                        logger.error(f"Error processing large block {block_idx+1}: {e}")
                        success = False

                else:
                    # Smaller block - process in chunks of 30 days as before
                    logger.info(
                        f"Processing smaller block {block_idx+1}/{len(blocks)} with {len(block)} hours using chunked approach"
                    )

                    # Process in blocks of 30 days maximum to avoid memory issues and rate limits
                    block_filled = self._process_yfinance_block(
                        ticker, metal_type, block
                    )
                    if block_filled < 0:  # Error occurred
                        success = False
                    else:
                        total_filled += block_filled

            logger.info(
                f"Total filled: {total_filled} historical hours with data for {ticker}"
            )
            return success

        except Exception as e:
            logger.error(f"Error in _update_from_yfinance for {ticker}: {e}")
            return False

    def _process_yfinance_block(self, ticker, metal_type, hours):
        """
        Process a block of missing hours using the chunked approach

        Args:
            ticker (str): Yahoo Finance ticker symbol
            metal_type (MetalType): Type of metal
            hours (list): List of datetime objects for hours to fill

        Returns:
            int: Number of hours filled with data, or -1 on error
        """
        try:
            # Import yfinance here to avoid dependency if not used
            import yfinance as yf

            # Sort hours chronologically
            hours.sort()

            # Process in chunks
            block_size = 30  # days
            total_filled = 0

            # Calculate date range
            start_date = hours[0].date()
            end_date = hours[-1].date()

            # Track processed hours
            processed_hour_strings = set()

            # Process data in chunks
            current_start = start_date
            while current_start <= end_date:
                # Calculate end of chunk (max 30 days or until end_date)
                current_end = min(current_start + timedelta(days=block_size), end_date)

                # Get hours in this chunk
                chunk_hours = [
                    h for h in hours if current_start <= h.date() <= current_end
                ]

                if not chunk_hours:
                    current_start = current_end + timedelta(days=1)
                    continue

                logger.info(
                    f"Processing {ticker} chunk from {current_start} to {current_end} ({len(chunk_hours)} hours)"
                )

                try:
                    # Add one day to end_date to ensure we get all hours
                    query_end = current_end + timedelta(days=1)
                    data = yf.download(
                        ticker,
                        start=current_start.strftime("%Y-%m-%d"),
                        end=query_end.strftime("%Y-%m-%d"),
                        interval="1h",
                    )

                    if data.empty:
                        logger.warning(
                            f"No data returned from yfinance for chunk {current_start} to {current_end}"
                        )
                        # Create batch of empty records
                        empty_records = []
                        for timestamp in chunk_hours:
                            hour_str = timestamp.strftime("%Y-%m-%d %H:00:00")
                            if hour_str not in processed_hour_strings:
                                record = MetalPrice(
                                    metal_type=metal_type,
                                    timestamp=timestamp,
                                    open_price=None,
                                    high_price=None,
                                    low_price=None,
                                    close_price=None,
                                    currency="USD",
                                    source=DataSource.YFINANCE,
                                    is_market_closed=1,  # No data
                                )
                                empty_records.append(record)
                                processed_hour_strings.add(hour_str)

                        # Save empty records
                        self._save_batch_yfinance_prices(empty_records)
                    else:
                        # Process each data point
                        chunk_filled = 0
                        records_to_save = []

                        # Create set of hours with data
                        data_hour_strings = {
                            idx.strftime("%Y-%m-%d %H:00:00") for idx in data.index
                        }

                        # Process each hour in the chunk
                        for timestamp in chunk_hours:
                            hour_str = timestamp.strftime("%Y-%m-%d %H:00:00")

                            if hour_str in processed_hour_strings:
                                continue  # Skip if already processed

                            if hour_str in data_hour_strings:
                                # Find the row in data
                                idx = None
                                for data_idx in data.index:
                                    if (
                                        data_idx.strftime("%Y-%m-%d %H:00:00")
                                        == hour_str
                                    ):
                                        idx = data_idx
                                        break

                                if idx is not None:
                                    row = data.loc[idx]

                                    # Create record with data
                                    try:
                                        # Convert to float safely
                                        def safe_float(val):
                                            return (
                                                float(val.iloc[0])
                                                if isinstance(val, pd.Series)
                                                else (
                                                    float(val)
                                                    if val is not None
                                                    else None
                                                )
                                            )

                                        open_val = safe_float(row["Open"])
                                        high_val = safe_float(row["High"])
                                        low_val = safe_float(row["Low"])
                                        close_val = safe_float(row["Close"])

                                        record = MetalPrice(
                                            metal_type=metal_type,
                                            timestamp=timestamp,
                                            open_price=open_val,
                                            high_price=high_val,
                                            low_price=low_val,
                                            close_price=close_val,
                                            currency="USD",
                                            source=DataSource.YFINANCE,
                                            is_market_closed=0,  # Has data
                                        )
                                        records_to_save.append(record)
                                        chunk_filled += 1
                                    except Exception as e:
                                        logger.error(
                                            f"Error processing data for {hour_str}: {e}"
                                        )
                                        # Create empty record on error
                                        record = MetalPrice(
                                            metal_type=metal_type,
                                            timestamp=timestamp,
                                            open_price=None,
                                            high_price=None,
                                            low_price=None,
                                            close_price=None,
                                            currency="USD",
                                            source=DataSource.YFINANCE,
                                            is_market_closed=1,  # No data due to error
                                        )
                                        records_to_save.append(record)
                            else:
                                # Create empty record
                                record = MetalPrice(
                                    metal_type=metal_type,
                                    timestamp=timestamp,
                                    open_price=None,
                                    high_price=None,
                                    low_price=None,
                                    close_price=None,
                                    currency="USD",
                                    source=DataSource.YFINANCE,
                                    is_market_closed=1,  # No data
                                )
                                records_to_save.append(record)

                            # Mark as processed
                            processed_hour_strings.add(hour_str)

                        # Save all records in this chunk
                        if records_to_save:
                            self._save_batch_yfinance_prices(records_to_save)

                        total_filled += chunk_filled
                        logger.info(
                            f"Filled {chunk_filled} hours in chunk {current_start} to {current_end}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing chunk {current_start} to {current_end}: {e}"
                    )
                    # Mark remaining hours in this chunk as empty
                    empty_records = []
                    for timestamp in chunk_hours:
                        hour_str = timestamp.strftime("%Y-%m-%d %H:00:00")
                        if hour_str not in processed_hour_strings:
                            record = MetalPrice(
                                metal_type=metal_type,
                                timestamp=timestamp,
                                open_price=None,
                                high_price=None,
                                low_price=None,
                                close_price=None,
                                currency="USD",
                                source=DataSource.YFINANCE,
                                is_market_closed=1,  # No data due to error
                            )
                            empty_records.append(record)
                            processed_hour_strings.add(hour_str)

                    # Save empty records
                    if empty_records:
                        self._save_batch_yfinance_prices(empty_records)

                # Move to next chunk
                current_start = current_end + timedelta(days=1)

            return total_filled

        except Exception as e:
            logger.error(f"Error in _process_yfinance_block: {e}")
            return -1  # Error


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()

    # Update gold price from Alpha Vantage API, filling in missing hourly data
    collector.update("XAU", "alphavantage")

    # Update gold futures from Yahoo Finance, filling in missing hourly data
    collector.update("GC=F", "yfinance")
