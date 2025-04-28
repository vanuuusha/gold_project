import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
import sys
import os
from alpha_vantage.timeseries import TimeSeries

# Constants
ALPHA_VANTAGE_API_KEY = (
    "YOUR_ALPHA_VANTAGE_API_KEY"  # Замените на ваш настоящий ключ API
)

# Add parent directory to path if script is run directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from database.init_database import (
        MetalType,
        DataSource,
        MetalPrice,
        CollectorSchedule,
    )
    from settings import DB_URL
else:
    # Import when module is imported from elsewhere
    from database.init_database import (
        MetalType,
        DataSource,
        MetalPrice,
        CollectorSchedule,
    )
    from settings import DB_URL

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
                "COPPER": MetalType.COPPER,
                "ALUMINUM": MetalType.ALUMINUM,
            }
            if ticker not in metal_map:
                logger.error(f"Unsupported metal ticker: {ticker}")
                return None
            return metal_map[ticker]

        elif platform_type.lower() == "yfinance":
            metal_map = {
                "GC=F": MetalType.GOLD,
                "HG=F": MetalType.COPPER,
                "ALI=F": MetalType.ALUMINUM,  # Aluminum futures
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

            try:
                # Используем прямые URL для данных металлов
                api_url = f"https://www.alphavantage.co/query?function={ticker}&interval=monthly&apikey={ALPHA_VANTAGE_API_KEY}"

                logger.info(f"Запрашиваем данные по URL: {api_url}")
                response = requests.get(api_url)
                data = response.json()

                if "Error Message" in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    raise ValueError(data["Error Message"])

                # Проверяем наличие данных
                if "name" not in data or "data" not in data:
                    logger.error(f"Unexpected API response format: {data.keys()}")
                    raise ValueError("Unexpected API response format")

                logger.info(f"Получены данные для {data['name']}")

                # Извлекаем временные ряды данных
                time_series_data = data["data"]
                logger.info(
                    f"Получено {len(time_series_data)} точек данных от Alpha Vantage"
                )

                # Готовим пакетные записи для обработки
                batch_records = []
                total_hours = len(missing_hours)
                hours_with_data = 0

                # Группируем пропущенные часы по датам для сопоставления с ежедневными данными
                missing_dates = {}
                for hour in missing_hours:
                    date_str = hour.strftime("%Y-%m-%d")
                    if date_str not in missing_dates:
                        missing_dates[date_str] = []
                    missing_dates[date_str].append(hour)

                # Обрабатываем данные Alpha Vantage
                # Создаем словарь дат и цен для быстрого доступа
                price_by_date = {}
                for entry in time_series_data:
                    # Проверяем, что у нас есть и дата, и значение цены
                    if "date" in entry and "value" in entry and entry["value"] != ".":
                        try:
                            # Преобразуем дату в формат YYYY-MM-DD
                            date_obj = datetime.strptime(entry["date"], "%Y-%m-%d")
                            date_str = date_obj.strftime("%Y-%m-%d")

                            # Преобразуем значение цены в float
                            price = float(entry["value"])

                            # Сохраняем в словаре
                            price_by_date[date_str] = price
                        except (ValueError, TypeError) as e:
                            logger.error(
                                f"Ошибка при обработке записи даты {entry}: {e}"
                            )

                # Обработка каждой даты из missing_dates
                for date_str, hours in list(missing_dates.items()):
                    # Находим ближайшую предыдущую дату с данными
                    price = None
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

                    # Если у нас есть цена для этого дня, используем её
                    if date_str in price_by_date:
                        price = price_by_date[date_str]
                    else:
                        # Ищем ближайшую дату не более 30 дней назад
                        for i in range(1, 31):
                            prev_date = (date_obj - timedelta(days=i)).strftime(
                                "%Y-%m-%d"
                            )
                            if prev_date in price_by_date:
                                price = price_by_date[prev_date]
                                break

                    if price is not None:
                        # Создаем записи для всех часов этой даты
                        for hour in hours:
                            record = MetalPrice(
                                metal_type=metal_type,
                                timestamp=hour,
                                open_price=price,
                                high_price=price,
                                low_price=price,
                                close_price=price,
                                currency="USD",
                                source=DataSource.ALPHAVANTAGE_API,
                                is_market_closed=0,  # Данные доступны
                            )
                            batch_records.append(record)
                            hours_with_data += 1

                        # Удаляем обработанную дату из словаря
                        del missing_dates[date_str]

                # Обрабатываем все часы, для которых не нашли данных
                for date_hours in missing_dates.values():
                    for hour in date_hours:
                        # Создаем пустую запись
                        record = MetalPrice(
                            metal_type=metal_type,
                            timestamp=hour,
                            open_price=None,
                            high_price=None,
                            low_price=None,
                            close_price=None,
                            currency="USD",
                            source=DataSource.ALPHAVANTAGE_API,
                            is_market_closed=1,  # Нет данных
                        )
                        batch_records.append(record)

                # Сохраняем все записи пакетами
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

    def run_scheduled_tasks(self):
        """
        Run scheduled collection tasks based on the collector_schedules table.
        This should be called from a scheduler at regular intervals (e.g., every hour)

        Returns:
            dict: A dictionary with metal types as keys and success status as values
        """
        try:
            session = self.Session()
            results = {}

            # Get all active schedules
            active_schedules = (
                session.query(CollectorSchedule)
                .filter(CollectorSchedule.is_active == 1)
                .all()
            )

            if not active_schedules:
                logger.info("No active schedules found")
                return {}

            current_time = datetime.now()

            for schedule in active_schedules:
                # Check if schedule should run now based on interval
                should_run = False

                # If never run before, run it now
                if not schedule.last_run:
                    should_run = True
                    logger.info(
                        f"Schedule {schedule.id} has never run before, running now"
                    )
                else:
                    # Calculate minutes since last run
                    minutes_since_last_run = (
                        current_time - schedule.last_run
                    ).total_seconds() / 60

                    # Check if the interval has elapsed
                    if minutes_since_last_run >= schedule.interval_minutes:
                        should_run = True
                        logger.info(
                            f"Schedule {schedule.id} for {schedule.metal_type.value} from {schedule.source.value}: "
                            f"{minutes_since_last_run:.1f} minutes elapsed since last run, interval is {schedule.interval_minutes} minutes"
                        )

                if should_run:
                    # Get the ticker for this metal and source
                    ticker = None

                    if schedule.source == DataSource.YFINANCE:
                        ticker_map = {
                            MetalType.GOLD: "GC=F",
                            MetalType.SILVER: "SI=F",
                            MetalType.PLATINUM: "PL=F",
                            MetalType.PALLADIUM: "PA=F",
                            MetalType.COPPER: "HG=F",
                        }
                        ticker = ticker_map.get(schedule.metal_type)
                    elif schedule.source == DataSource.ALPHAVANTAGE_API:
                        ticker_map = {
                            MetalType.GOLD: "XAUUSD",
                            MetalType.SILVER: "XAGUSD",
                            MetalType.PLATINUM: "XPTUSD",
                            MetalType.PALLADIUM: "XPDUSD",
                            MetalType.COPPER: "COPPER",
                        }
                        ticker = ticker_map.get(schedule.metal_type)

                    if not ticker:
                        logger.error(
                            f"No ticker found for {schedule.metal_type.value} from {schedule.source.value}"
                        )
                        results[
                            f"{schedule.metal_type.value}_{schedule.source.value}"
                        ] = False
                        continue

                    # Calculate days back based on interval type
                    days_back = 1  # Default for hourly
                    if schedule.interval_type == "daily":
                        days_back = 2  # Get 2 days of data for daily updates
                    elif schedule.interval_type == "weekly":
                        days_back = 8  # Get 8 days of data for weekly updates

                    # Run the update
                    platform_type = schedule.source.name.lower()
                    result = self.update(
                        ticker=ticker, platform_type=platform_type, days_back=days_back
                    )

                    # Record result
                    results[f"{schedule.metal_type.value}_{schedule.source.value}"] = (
                        result
                    )

                    # Update last_run timestamp if successful
                    if result:
                        schedule.last_run = current_time
                        logger.info(
                            f"Successfully updated {schedule.metal_type.value} from {schedule.source.value}"
                        )
                    else:
                        logger.error(
                            f"Failed to update {schedule.metal_type.value} from {schedule.source.value}"
                        )

            # Commit changes to last_run timestamps
            session.commit()
            return results

        except Exception as e:
            logger.error(f"Error running scheduled tasks: {e}")
            if "session" in locals():
                session.rollback()
            return {"error": str(e)}

        finally:
            if "session" in locals():
                session.close()


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()

    # Update gold price from Alpha Vantage API, filling in missing hourly data
    collector.update("XAU", "alphavantage")

    # Update gold futures from Yahoo Finance, filling in missing hourly data
    collector.update("GC=F", "yfinance")
