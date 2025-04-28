#!/usr/bin/env python
"""
Scheduler runner script for the Precious Metals Analytics application.
This script is intended to be run regularly (e.g. every hour via cron)
to check and execute scheduled data collection tasks.
"""

import sys
import os
from loguru import logger
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from database.data_collector import DataCollector
from settings import LOG_DIR

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"scheduler_{datetime.now().strftime('%Y%m%d')}.log")

# Configure logger
logger.remove()  # Remove default handlers
logger.add(
    log_file,
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(sys.stderr, level="INFO")


def main():
    """Run the scheduler to check and process scheduled tasks"""
    start_time = time.time()
    logger.info("=== Starting scheduled data collection task runner ===")

    # Initialize data collector
    collector = DataCollector()

    # Run scheduled tasks
    results = collector.run_scheduled_tasks()

    if not results:
        logger.info("No tasks were executed")
    else:
        logger.info(f"Tasks executed: {len(results)}")
        for task, success in results.items():
            logger.info(f"Task {task}: {'Success' if success else 'Failed'}")

    elapsed_time = time.time() - start_time
    logger.info(f"=== Scheduler finished in {elapsed_time:.2f} seconds ===")


if __name__ == "__main__":
    main()
