"""
Application settings module
"""

import os

ALPHA_VANTAGE_API_KEY = "E7EOLYSDOF974K1N"  # Replace with your Alpha Vantage API key

# Database configuration
DB_CONFIG = {
    "user": "gold",
    "password": "gold",
    "host": "localhost",
    "port": 3306,
    "database": "gold",
}

# For backward compatibility
DB_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Application configuration
APP_CONFIG = {
    "debug": True,
    "port": 5001,
    "host": "0.0.0.0",
    "enable_scheduler": True,  # Set to False to disable the background scheduler thread
}

# Collector configuration
COLLECTOR_CONFIG = {
    "default_interval_minutes": 60,
    "initial_load_enabled": True,
}

# Logging configuration
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
