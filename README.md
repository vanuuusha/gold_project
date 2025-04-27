# Precious Metals Analytics

This is an information and analytics application for precious metals market data with a web interface. The application collects price data for various precious metals from open sources, stores it in a database, and provides interactive dashboards and reports for analysis.

## Features

- Data collection from multiple sources (YFinance and Alpha Vantage API)
- Scheduled and on-demand data updates
- Interactive dashboards with various chart types:
  - Time series analysis
  - Price distribution analysis
  - Box plots
  - Candlestick charts
  - Price comparison
  - Volatility analysis
- Database management interface
- Text-based reports with export to CSV and Excel
- Data visualization with dynamic filtering

## Technology Stack

- **Backend**: Flask
- **Frontend**: Dash, Plotly, Bootstrap
- **Database**: MySQL with SQLAlchemy ORM
- **Logging**: Loguru
- **Data Processing**: Pandas, NumPy

## Prerequisites

- Python 3.7+
- MySQL database server

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd precious-metals-analytics
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Configure the database settings in `settings.py`:

```python
# Database configuration
DB_CONFIG = {
    "user": "your_username",
    "password": "your_password",
    "host": "localhost",
    "port": 3306,
    "database": "your_database_name",
}
```

5. Initialize the database:

```bash
python -c "from database.init_database import init_database; init_database()"
```

## Running the Application

Start the application with:

```bash
python app.py
```

The application will be available at: `http://localhost:5001/`

You can also access the database initialization route directly: `http://localhost:5001/initialize-database`

## Data Collection

The application can collect data from:

- **YFinance**: Gold, Silver, Platinum, Palladium, and Copper futures
- **Alpha Vantage API**: Precious metals price data (requires an API key in settings.py)

Data collection can be scheduled or triggered manually through the web interface.

## Project Structure

- `app.py`: Main application entry point
- `settings.py`: Application configuration
- `database/`: Database models and data collection logic
  - `init_database.py`: Database models and initialization
  - `data_collector.py`: Data collection logic
- `views/`: Dashboard and UI components
  - `data_collection.py`: Data collection interface
  - `database_management.py`: Database management interface
  - `visual_dashboard.py`: Interactive visualizations
  - `reports.py`: Text-based reports

## License

This project is licensed under the MIT License - see the LICENSE file for details.