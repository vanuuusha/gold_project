"""
Application settings module
"""

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
}

# Collector configuration
COLLECTOR_CONFIG = {
    "default_interval_minutes": 60,
    "initial_load_enabled": True,
}
