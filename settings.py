"""
Application settings module
"""

DB_CONFIG = {
    "user": "gold",
    "password": "gold",
    "host": "localhost",
    "port": 3306,
    "database": "gold",
}

DB_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

APP_CONFIG = {
    "debug": True,
    "port": 5001,
    "host": "0.0.0.0",
    "enable_scheduler": True,
}

COLLECTOR_CONFIG = {
    "default_interval_minutes": 60,
    "initial_load_enabled": True,
}

METAL_TYPES = {
    "GOLD": "gold",
    "COPPER": "copper",
}

SOURCE_METALS_CONFIG = {
    "YFINANCE": ["GOLD", "COPPER"],
}
