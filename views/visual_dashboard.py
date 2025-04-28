from dash import html, dcc, dash_table, dash, callback_context
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, date
from loguru import logger
import json

from database.init_database import MetalType, DataSource, MetalPrice
from settings import DB_URL

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)


# Function to calculate technical indicators
def calculate_indicators(df, metal_type=None, custom_params=None):
    """Calculate various technical indicators for the data"""
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    if metal_type:
        # Filter for specific metal type
        mask = result_df["metal_type"] == metal_type
        temp_df = result_df[mask].copy()
    else:
        temp_df = result_df.copy()

    if temp_df.empty:
        return result_df

    # Sort by timestamp
    temp_df = temp_df.sort_values("timestamp")

    # Get custom parameters
    if custom_params is None:
        custom_params = {}

    # Get custom moving averages
    custom_mas = custom_params.get("moving_avgs", [])
    bb_period = custom_params.get("bb_period", 20)
    bb_stddev = custom_params.get("bb_stddev", 2)
    rsi_period = custom_params.get("rsi_period", 14)

    # Simple Moving Averages (SMA)
    for ma_entry in custom_mas:
        if ma_entry["type"] == "sma":
            window = ma_entry["period"]
            col_name = f"sma_{window}"
            temp_df[col_name] = temp_df["close_price"].rolling(window=window).mean()

    # Exponential Moving Averages (EMA)
    for ma_entry in custom_mas:
        if ma_entry["type"] == "ema":
            window = ma_entry["period"]
            col_name = f"ema_{window}"
            temp_df[col_name] = (
                temp_df["close_price"].ewm(span=window, adjust=False).mean()
            )

    # Bollinger Bands
    bb_col = f"sma_{bb_period}"
    # Add the SMA if it doesn't exist yet
    if bb_col not in temp_df.columns:
        temp_df[bb_col] = temp_df["close_price"].rolling(window=bb_period).mean()

    # Use the SMA as middle band
    temp_df["bb_middle"] = temp_df[bb_col]
    # Standard deviation of the price
    temp_df["bb_stddev"] = temp_df["close_price"].rolling(window=bb_period).std()
    # Upper and lower bands (using specified standard deviations)
    temp_df["bb_upper"] = temp_df["bb_middle"] + bb_stddev * temp_df["bb_stddev"]
    temp_df["bb_lower"] = temp_df["bb_middle"] - bb_stddev * temp_df["bb_stddev"]

    # Relative Strength Index (RSI)
    # Calculate price changes
    temp_df["price_change"] = temp_df["close_price"].diff()

    # Calculate gains and losses
    temp_df["gain"] = np.where(temp_df["price_change"] > 0, temp_df["price_change"], 0)
    temp_df["loss"] = np.where(temp_df["price_change"] < 0, -temp_df["price_change"], 0)

    # Calculate average gains and losses over specified period
    temp_df["avg_gain"] = temp_df["gain"].rolling(window=rsi_period).mean()
    temp_df["avg_loss"] = temp_df["loss"].rolling(window=rsi_period).mean()

    # Calculate relative strength (RS)
    temp_df["rs"] = temp_df["avg_gain"] / temp_df["avg_loss"]

    # Calculate RSI
    temp_df["rsi"] = 100 - (100 / (1 + temp_df["rs"]))

    # Ensure RSI is within bounds (0-100)
    temp_df["rsi"] = np.clip(temp_df["rsi"], 0, 100)

    # If we're working with a subset, update the original dataframe
    if metal_type:
        # Update the original dataframe with the calculated indicators
        for col in temp_df.columns:
            if col not in result_df.columns:
                result_df.loc[mask, col] = temp_df[col].values
            elif col not in ["metal_type", "timestamp"]:  # Don't overwrite keys
                result_df.loc[mask, col] = temp_df[col].values
    else:
        result_df = temp_df

    return result_df


# Define layout
layout = html.Div(
    [
        # Store for user preferences
        dcc.Store(id="user-preferences", storage_type="local"),
        # Store to track if metals were fetched for a source
        dcc.Store(id="metals-fetched-store", storage_type="session", data={}),
        # Theme toggle - will be positioned in the navbar
        html.Div(
            [
                dbc.Switch(
                    id="theme-switch",
                    label="Dark Mode",
                    value=True,
                    className="ms-auto",
                ),
            ],
            className="ms-auto d-flex align-items-center no-print",
            style={"marginRight": "20px"},
        ),
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard/", active=True)),
                dbc.NavItem(
                    dbc.NavLink("Data Collection", href="/dashboard/data-collection")
                ),
                dbc.NavItem(
                    dbc.NavLink("Database Management", href="/dashboard/database")
                ),
                dbc.NavItem(dbc.NavLink("Reports", href="/dashboard/reports")),
                # Theme switch will be inserted here via a callback
                html.Div(id="theme-switch-container"),
            ],
            brand="Precious Metals Analytics",
            brand_href="/dashboard/",
            color="primary",
            dark=True,
            id="navbar",
            className="no-print",
        ),
        dbc.Container(
            [
                html.H1("Dashboard", className="my-4"),
                html.P("Interactive visualization dashboard for metal price analysis."),
                html.Hr(),
                # Dashboard Controls
                dbc.Row(
                    [
                        # Metal Selection
                        dbc.Col(
                            [
                                html.Label("Select Metals"),
                                dcc.Dropdown(
                                    id="metals-dropdown",
                                    options=[
                                        {
                                            "label": metal_type.value,
                                            "value": metal_type.name,
                                        }
                                        for metal_type in MetalType
                                    ],
                                    value=[MetalType.GOLD.name],
                                    multi=True,
                                ),
                            ],
                            width=4,
                        ),
                        # Source Selection
                        dbc.Col(
                            [
                                html.Label("Select Data Source"),
                                dcc.Dropdown(
                                    id="source-dropdown",
                                    options=[
                                        {"label": source.value, "value": source.name}
                                        for source in DataSource
                                    ],
                                    value=DataSource.YFINANCE.name,
                                ),
                            ],
                            width=4,
                        ),
                        # Timeframe Selection
                        dbc.Col(
                            [
                                html.Label("Timeframe"),
                                dcc.Dropdown(
                                    id="timeframe-dropdown",
                                    options=[
                                        {"label": "Hourly", "value": "1H"},
                                        {"label": "4 Hours", "value": "4H"},
                                        {"label": "Daily", "value": "1D"},
                                    ],
                                    value="1D",
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-3",
                ),
                # Time Period Selection
                dbc.Row(
                    [
                        # Time Period Selection
                        dbc.Col(
                            [
                                html.Label("Time Period"),
                                dcc.Dropdown(
                                    id="time-period-dropdown",
                                    options=[
                                        {"label": "Last 7 Days", "value": "7D"},
                                        {"label": "Last 30 Days", "value": "30D"},
                                        {"label": "Last 90 Days", "value": "90D"},
                                        {"label": "Last 180 Days", "value": "180D"},
                                        {"label": "Last Year", "value": "365D"},
                                        {"label": "Custom Range", "value": "custom"},
                                    ],
                                    value="30D",
                                ),
                            ],
                            width=6,
                        ),
                        # Chart Type Selection
                        dbc.Col(
                            [
                                html.Label("Chart Type"),
                                dcc.Dropdown(
                                    id="chart-type-dropdown",
                                    options=[
                                        {
                                            "label": "Advanced Chart",
                                            "value": "advanced",
                                        },
                                        {
                                            "label": "Price Distribution",
                                            "value": "histogram",
                                        },
                                        {
                                            "label": "Box Plot Analysis",
                                            "value": "box_plot",
                                        },
                                        {
                                            "label": "Candlestick Chart",
                                            "value": "candlestick",
                                        },
                                        {
                                            "label": "Price Comparison",
                                            "value": "comparison",
                                        },
                                        {"label": "Scatter Plot", "value": "scatter"},
                                    ],
                                    value="advanced",
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-3",
                ),
                # Custom Date Range (initially hidden)
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="dashboard-custom-date-range-container",
                                    children=[
                                        html.Label("Custom Date Range"),
                                        dcc.DatePickerRange(
                                            id="dashboard-custom-date-range",
                                            min_date_allowed=datetime.now()
                                            - timedelta(days=365 * 2),
                                            max_date_allowed=datetime.now(),
                                            start_date=(
                                                datetime.now() - timedelta(days=30)
                                            ).date(),
                                            end_date=datetime.now().date(),
                                        ),
                                    ],
                                    style={"display": "none"},
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-3",
                ),
                # Technical Indicators Controls (shown by default now)
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="technical-indicators-container",
                                    children=[
                                        html.H5("Technical Indicators"),
                                        # Price Series Controls
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Price Series"),
                                                dbc.CardBody(
                                                    [
                                                        dbc.Checklist(
                                                            options=[
                                                                {
                                                                    "label": "Open",
                                                                    "value": "open",
                                                                },
                                                                {
                                                                    "label": "High",
                                                                    "value": "high",
                                                                },
                                                                {
                                                                    "label": "Low",
                                                                    "value": "low",
                                                                },
                                                                {
                                                                    "label": "Close",
                                                                    "value": "close",
                                                                },
                                                            ],
                                                            value=["close"],
                                                            id="price-series-checklist",
                                                            inline=True,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        # Moving Averages
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Moving Averages"),
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Simple Moving Averages (SMA)"
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Col(
                                                                                    [
                                                                                        dbc.InputGroup(
                                                                                            [
                                                                                                dbc.InputGroupText(
                                                                                                    "Period:"
                                                                                                ),
                                                                                                dbc.Input(
                                                                                                    id="sma-period-input",
                                                                                                    type="number",
                                                                                                    value=0,
                                                                                                    min=2,
                                                                                                    max=200,
                                                                                                    step=1,
                                                                                                ),
                                                                                            ],
                                                                                            size="sm",
                                                                                        ),
                                                                                    ],
                                                                                    width=12,
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                    className="mb-2",
                                                                ),
                                                            ]
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Exponential Moving Averages (EMA)"
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Col(
                                                                                    [
                                                                                        dbc.InputGroup(
                                                                                            [
                                                                                                dbc.InputGroupText(
                                                                                                    "Period:"
                                                                                                ),
                                                                                                dbc.Input(
                                                                                                    id="ema-period-input",
                                                                                                    type="number",
                                                                                                    value=0,
                                                                                                    min=2,
                                                                                                    max=200,
                                                                                                    step=1,
                                                                                                ),
                                                                                            ],
                                                                                            size="sm",
                                                                                        ),
                                                                                    ],
                                                                                    width=12,
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                    className="mb-2",
                                                                ),
                                                            ]
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Active Moving Averages"
                                                                ),
                                                                dbc.Checklist(
                                                                    id="moving-avg-checklist",
                                                                    options=[],
                                                                    value=[],
                                                                    inline=True,
                                                                ),
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        # Bollinger Bands
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Bollinger Bands"),
                                                dbc.CardBody(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.InputGroup(
                                                                            [
                                                                                dbc.InputGroupText(
                                                                                    "Period:"
                                                                                ),
                                                                                dbc.Input(
                                                                                    id="bb-period-input",
                                                                                    type="number",
                                                                                    value=20,
                                                                                    min=2,
                                                                                    max=50,
                                                                                    step=1,
                                                                                ),
                                                                            ],
                                                                            size="sm",
                                                                            className="mb-2",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        dbc.InputGroup(
                                                                            [
                                                                                dbc.InputGroupText(
                                                                                    "Std Dev:"
                                                                                ),
                                                                                dbc.Input(
                                                                                    id="bb-stddev-input",
                                                                                    type="number",
                                                                                    value=2,
                                                                                    min=1,
                                                                                    max=4,
                                                                                    step=0.5,
                                                                                ),
                                                                            ],
                                                                            size="sm",
                                                                            className="mb-2",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Checklist(
                                                            options=[
                                                                {
                                                                    "label": "Show Bollinger Bands",
                                                                    "value": "bb",
                                                                },
                                                            ],
                                                            value=[],
                                                            id="bb-checklist",
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        # Other indicators
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Other Indicators"),
                                                dbc.CardBody(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.InputGroup(
                                                                            [
                                                                                dbc.InputGroupText(
                                                                                    "RSI Period:"
                                                                                ),
                                                                                dbc.Input(
                                                                                    id="rsi-period-input",
                                                                                    type="number",
                                                                                    value=14,
                                                                                    min=2,
                                                                                    max=50,
                                                                                    step=1,
                                                                                ),
                                                                            ],
                                                                            size="sm",
                                                                            className="mb-2",
                                                                        ),
                                                                    ],
                                                                    width=12,
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Checklist(
                                                            options=[
                                                                {
                                                                    "label": "RSI",
                                                                    "value": "rsi",
                                                                },
                                                            ],
                                                            value=[],
                                                            id="other-indicators-checklist",
                                                            inline=True,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-3",
                ),
                # Main Chart
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Price Analysis"),
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    id="loading-main-chart",
                                                    type="circle",
                                                    children=[
                                                        dcc.Graph(
                                                            id="main-chart",
                                                            style={"height": "54vh"},
                                                            config={
                                                                "displayModeBar": True,
                                                                "scrollZoom": True,
                                                                "modeBarButtonsToAdd": [
                                                                    "drawline",
                                                                    "drawopenpath",
                                                                    "eraseshape",
                                                                ],
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-4",
                ),
                # Spacer for RSI - will be dynamically sized
                html.Div(id="rsi-spacer", style={"height": "0px"}),
                # Additional Charts Row
                dbc.Row(
                    [
                        # Price Statistics
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Price Statistics"),
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    id="loading-stats-chart",
                                                    type="circle",
                                                    children=[
                                                        html.Div(id="price-statistics"),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                        # Volatility Analysis
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Volatility Analysis"),
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    id="loading-volatility-chart",
                                                    type="circle",
                                                    children=[
                                                        dcc.Graph(
                                                            id="volatility-chart",
                                                            style={"height": "30vh"},
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Data Table Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Historical Price Data"),
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    id="loading-data-table",
                                                    type="circle",
                                                    children=[
                                                        html.Div(id="price-data-table"),
                                                    ],
                                                ),
                                                # Add download component here instead
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-4",
                ),
                # Hidden components
                dcc.Store(id="chart-data-store"),
                dcc.Store(id="custom-indicators-store"),
            ],
            className="mt-4",
            id="main-container",
        ),
    ],
    id="dashboard-container",
)


# Helper function to get price data
def get_price_data(metals, source, start_date, end_date, timeframe="1D"):
    try:
        session = Session()

        # Convert metals to MetalType enums if they're strings
        if isinstance(metals, list) and all(isinstance(m, str) for m in metals):
            metals = [MetalType[m] for m in metals]
        elif isinstance(metals, str):
            metals = [MetalType[metals]]

        # Convert source to DataSource enum if it's a string
        if isinstance(source, str):
            source = DataSource[source]

        # Build query
        query = (
            session.query(
                MetalPrice.metal_type,
                MetalPrice.timestamp,
                MetalPrice.open_price,
                MetalPrice.high_price,
                MetalPrice.low_price,
                MetalPrice.close_price,
            )
            .filter(
                and_(
                    MetalPrice.source == source,
                    MetalPrice.timestamp >= start_date,
                    MetalPrice.timestamp <= end_date,
                    MetalPrice.metal_type.in_(metals),
                )
            )
            .order_by(MetalPrice.metal_type, MetalPrice.timestamp)
        )

        # Execute query
        results = query.all()

        # Convert to DataFrame
        data = []
        for result in results:
            metal_type, timestamp, open_price, high_price, low_price, close_price = (
                result
            )

            # Skip records with missing prices
            if close_price is None:
                continue

            data.append(
                {
                    "metal_type": metal_type.value,
                    "timestamp": timestamp,
                    "open_price": open_price,
                    "high_price": high_price,
                    "low_price": low_price,
                    "close_price": close_price,
                }
            )

        df = pd.DataFrame(data)

        # Handle empty results
        if df.empty:
            return None, "No data found for the selected criteria"

        # Apply timeframe resampling if needed
        if timeframe != "1D" and not df.empty:
            # Define resample rule based on timeframe
            resample_map = {
                "1H": "1H",
                "4H": "4H",
                "1D": "1D",
            }
            rule = resample_map.get(timeframe, "1D")

            # Process each metal type separately
            all_resampled = []

            for metal in df["metal_type"].unique():
                # Get data for this metal
                metal_df = df[df["metal_type"] == metal].copy()

                # Set timestamp as index for resampling
                metal_df.set_index("timestamp", inplace=True)

                try:
                    # Resample the data
                    resampled = metal_df.resample(rule).agg(
                        {
                            "open_price": "first",
                            "high_price": "max",
                            "low_price": "min",
                            "close_price": "last",
                        }
                    )

                    # Fill any missing values that might occur during resampling
                    resampled.fillna(method="ffill", inplace=True)

                    # Add back the metal_type
                    resampled["metal_type"] = metal

                    # Reset the index to get timestamp as a column
                    resampled.reset_index(inplace=True)

                    all_resampled.append(resampled)
                except Exception as e:
                    logger.error(f"Error resampling data for {metal}: {e}")

            # Combine all the resampled dataframes
            if all_resampled:
                df = pd.concat(all_resampled, ignore_index=True)
        elif timeframe == "1D":  # Ensure daily data is handled properly
            # Process each metal type separately
            all_daily = []

            for metal in df["metal_type"].unique():
                # Get data for this metal
                metal_df = df[df["metal_type"] == metal].copy()

                # Convert timestamp to date for grouping
                metal_df["date"] = metal_df["timestamp"].dt.date

                # Group by date to get daily OHLC values
                daily = metal_df.groupby("date").agg(
                    {
                        "open_price": "first",
                        "high_price": "max",
                        "low_price": "min",
                        "close_price": "last",
                        "metal_type": "first",
                    }
                )

                # Reset index and convert date back to datetime
                daily.reset_index(inplace=True)
                daily["timestamp"] = pd.to_datetime(daily["date"])
                daily.drop("date", axis=1, inplace=True)

                all_daily.append(daily)

            # Combine all daily dataframes
            if all_daily:
                df = pd.concat(all_daily, ignore_index=True)

        logger.info(f"Returning dataframe with {len(df)} rows")
        return df, None

    except Exception as e:
        logger.error(f"Error getting price data: {e}")
        return None, f"Error: {str(e)}"

    finally:
        session.close()


# Function to get available metal types for a specific data source
def get_available_metals_for_source(source):
    """
    Query the database to get all metal types available for a specific data source

    Args:
        source (str or DataSource): The data source to check

    Returns:
        list: List of dictionaries with metal type options
        str or None: Error message if any
    """
    try:
        session = Session()

        # Convert source to DataSource enum if it's a string
        if isinstance(source, str):
            source = DataSource[source]

        # Query distinct metal types for this source
        query = (
            session.query(MetalPrice.metal_type)
            .filter(MetalPrice.source == source)
            .distinct()
        )

        # Execute query
        results = query.all()

        # Convert to options format for dropdown
        options = []
        for result in results:
            metal_type = result[0]  # Get the metal type from the result tuple
            options.append({"label": metal_type.value, "value": metal_type.name})

        # If no results, return all metal types as fallback
        if not options:
            logger.warning(
                f"No metals found for source {source}, using all metals as fallback"
            )
            options = [
                {"label": metal_type.value, "value": metal_type.name}
                for metal_type in MetalType
            ]

        return options, None

    except Exception as e:
        logger.error(f"Error getting available metals: {e}")
        return [
            {"label": metal_type.value, "value": metal_type.name}
            for metal_type in MetalType
        ], f"Error: {str(e)}"

    finally:
        session.close()


# Register callbacks
def register_callbacks(app):
    # Show/hide custom date range based on time period selection
    @app.callback(
        Output("dashboard-custom-date-range-container", "style"),
        [Input("time-period-dropdown", "value")],
    )
    def toggle_custom_date_range(time_period):
        if time_period == "custom":
            return {"display": "block"}
        return {"display": "none"}

    # Update metals dropdown based on selected data source
    @app.callback(
        [Output("metals-dropdown", "options"), Output("metals-fetched-store", "data")],
        [Input("source-dropdown", "value")],
        [State("metals-fetched-store", "data")],
    )
    def update_metals_dropdown(source, fetched_sources):
        # Handle case when fetched_sources is None
        if fetched_sources is None:
            fetched_sources = {}

        # Determine which input triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            # Initial load
            triggered = "initial"
        else:
            triggered = ctx.triggered[0]["prop_id"].split(".")[0]

        if not source:
            # Default to all metals if no source selected
            return [
                {"label": metal_type.value, "value": metal_type.name}
                for metal_type in MetalType
            ], fetched_sources

        # Check if we've already fetched metals for this source
        if source in fetched_sources:
            # Return the cached options
            return fetched_sources[source]["options"], fetched_sources

        # Get available metals for this source
        options, error = get_available_metals_for_source(source)
        if error:
            logger.error(f"Error updating metals dropdown: {error}")

        # Update the fetched_sources dictionary
        fetched_sources[source] = {"options": options}

        return options, fetched_sources

    # Update metals dropdown value when source changes to ensure compatibility
    @app.callback(
        Output("metals-dropdown", "value", allow_duplicate=True),
        [Input("source-dropdown", "value")],
        [State("metals-dropdown", "value"), State("metals-fetched-store", "data")],
        # Don't run this callback when loading preferences
        prevent_initial_call=True,
    )
    def update_metals_selection(source, current_selection, fetched_sources):
        # Handle case when fetched_sources is None
        if fetched_sources is None:
            fetched_sources = {}

        if not source:
            return current_selection

        # Check if we have info for this source
        if source not in fetched_sources:
            return current_selection

        # Get available options for this source
        available_options = fetched_sources[source]["options"]
        available_values = [option["value"] for option in available_options]

        # Filter current selection to only include available metals
        if current_selection:
            filtered_selection = [
                metal for metal in current_selection if metal in available_values
            ]
            # Allow empty selection - don't automatically add a metal
            return filtered_selection

        # Keep empty selection if that's what user has chosen
        return current_selection

    # Show/hide technical indicators based on chart type
    @app.callback(
        Output("technical-indicators-container", "style"),
        [Input("chart-type-dropdown", "value")],
    )
    def toggle_technical_indicators(chart_type):
        if chart_type == "advanced":
            return {"display": "block"}
        return {"display": "none"}

    # Update end date of custom range when time period changes
    @app.callback(
        [
            Output("dashboard-custom-date-range", "start_date"),
            Output("dashboard-custom-date-range", "end_date"),
        ],
        [Input("time-period-dropdown", "value")],
    )
    def update_custom_date_range(time_period):
        end_date = datetime.now().date()

        if time_period == "1D":
            start_date = (datetime.now() - timedelta(days=1)).date()
        elif time_period == "7D":
            start_date = (datetime.now() - timedelta(days=7)).date()
        elif time_period == "30D":
            start_date = (datetime.now() - timedelta(days=30)).date()
        elif time_period == "90D":
            start_date = (datetime.now() - timedelta(days=90)).date()
        elif time_period == "180D":
            start_date = (datetime.now() - timedelta(days=180)).date()
        elif time_period == "365D":
            start_date = (datetime.now() - timedelta(days=365)).date()
        else:
            # For custom, keep the current values (will be overridden by user)
            return dash.no_update, dash.no_update

        return start_date, end_date

    # Theme switch callback
    @app.callback(
        [
            Output("dashboard-container", "className"),
            Output("navbar", "dark"),
            Output("navbar", "color"),
        ],
        [Input("theme-switch", "value")],
    )
    def update_theme(dark_mode):
        # Default to dark mode if value is None
        if dark_mode is None:
            dark_mode = True

        if dark_mode:
            return "dbc dbc-dark", True, "primary"
        else:
            return "dbc", False, "light"

    # Load user preferences from local storage
    @app.callback(
        [
            Output("metals-dropdown", "value"),
            Output("source-dropdown", "value"),
            Output("timeframe-dropdown", "value"),
            Output("time-period-dropdown", "value"),
            Output("chart-type-dropdown", "value"),
            Output("theme-switch", "value"),
            Output("price-series-checklist", "value"),
            Output("moving-avg-checklist", "value", allow_duplicate=True),
            Output("bb-checklist", "value"),
            Output("other-indicators-checklist", "value", allow_duplicate=True),
        ],
        [Input("user-preferences", "data")],
        [
            State("metals-dropdown", "value"),
            State("source-dropdown", "value"),
            State("timeframe-dropdown", "value"),
            State("time-period-dropdown", "value"),
            State("chart-type-dropdown", "value"),
            State("theme-switch", "value"),
            State("price-series-checklist", "value"),
            State("moving-avg-checklist", "value"),
            State("bb-checklist", "value"),
            State("other-indicators-checklist", "value"),
            State("metals-fetched-store", "data"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def load_preferences(
        stored_prefs,
        metals,
        source,
        timeframe,
        time_period,
        chart_type,
        theme_switch,
        price_series,
        moving_avgs,
        bb,
        other_indicators,
        fetched_sources,
    ):
        # Handle case when fetched_sources is None
        if fetched_sources is None:
            fetched_sources = {}

        # Default values if no stored preferences
        if not stored_prefs:
            return (
                metals or [],  # Return empty list instead of [MetalType.GOLD.name]
                source or DataSource.YFINANCE.name,
                timeframe or "1D",
                time_period or "30D",
                chart_type or "advanced",
                theme_switch,
                price_series or ["close"],
                moving_avgs or [],
                bb or [],
                other_indicators or [],
            )

        # Return stored values or defaults if keys don't exist
        try:
            # Get the stored or default source
            preferred_source = stored_prefs.get(
                "source", source or DataSource.YFINANCE.name
            )

            # Get available metals for this source
            # Use fetched_sources if available for this source, otherwise query
            available_metal_values = []
            if preferred_source in fetched_sources:
                available_options = fetched_sources[preferred_source]["options"]
                available_metal_values = [
                    option["value"] for option in available_options
                ]
            else:
                # Fallback to querying the database if not cached yet
                available_metals, _ = get_available_metals_for_source(preferred_source)
                available_metal_values = [
                    option["value"] for option in available_metals
                ]
                # We don't update fetched_sources here because another callback will handle that

            # Get stored metals and filter to only include available ones
            stored_metals = stored_prefs.get("metals", metals or [])

            # If user has already selected metals and we're not changing the source,
            # keep the user's selection instead of filtering
            current_source = source or DataSource.YFINANCE.name
            if current_source == preferred_source and metals is not None:
                filtered_metals = metals
            else:
                # Filter to only include available metals for the selected source
                if isinstance(stored_metals, list):
                    filtered_metals = [
                        m for m in stored_metals if m in available_metal_values
                    ]
                    # Don't add a default metal when filtering results in empty list
                    # This allows empty selection
                else:
                    # Handle case when stored_metals is not a list
                    filtered_metals = []

            return (
                filtered_metals,
                preferred_source,
                stored_prefs.get("timeframe", timeframe or "1D"),
                stored_prefs.get("time_period", time_period or "30D"),
                stored_prefs.get("chart_type", chart_type or "advanced"),
                stored_prefs.get("theme_switch", theme_switch),
                stored_prefs.get("price_series", price_series or ["close"]),
                stored_prefs.get("moving_avgs", moving_avgs or []),
                stored_prefs.get("bb", bb or []),
                stored_prefs.get("other_indicators", other_indicators or []),
            )
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    # Save user preferences to local storage
    @app.callback(
        Output("user-preferences", "data", allow_duplicate=True),
        [
            Input("metals-dropdown", "value"),
            Input("source-dropdown", "value"),
            Input("timeframe-dropdown", "value"),
            Input("time-period-dropdown", "value"),
            Input("chart-type-dropdown", "value"),
            Input("theme-switch", "value"),
            Input("price-series-checklist", "value"),
            Input("moving-avg-checklist", "value"),
            Input("bb-checklist", "value"),
            Input("other-indicators-checklist", "value"),
        ],
        [
            State("user-preferences", "data"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def save_preferences(
        metals,
        source,
        timeframe,
        time_period,
        chart_type,
        theme_switch,
        price_series,
        moving_avgs,
        bb,
        other_indicators,
        current_prefs,
    ):
        # Determine which input triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            # Initial load, don't update
            return current_prefs or {}

        # Create or update preferences
        prefs = current_prefs or {}
        prefs.update(
            {
                "metals": metals,
                "source": source,
                "timeframe": timeframe,
                "time_period": time_period,
                "chart_type": chart_type,
                "theme_switch": theme_switch,
                "price_series": price_series,
                "moving_avgs": moving_avgs,
                "bb": bb,
                "other_indicators": other_indicators,
            }
        )

        return prefs

    # Manage custom moving averages - Add SMA
    @app.callback(
        [
            Output("moving-avg-checklist", "options", allow_duplicate=True),
            Output("moving-avg-checklist", "value", allow_duplicate=True),
            Output("custom-indicators-store", "data", allow_duplicate=True),
        ],
        [
            Input("add-sma-button", "n_clicks"),
            Input("add-ema-button", "n_clicks"),
        ],
        [
            State("sma-period-input", "value"),
            State("ema-period-input", "value"),
            State("moving-avg-checklist", "value"),
            State("moving-avg-checklist", "options"),
            State("custom-indicators-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def add_custom_moving_average(
        sma_clicks,
        ema_clicks,
        sma_period,
        ema_period,
        current_ma_values,
        current_options,
        custom_indicators,
    ):
        ctx = callback_context
        if not ctx.triggered:
            # Initial load - don't update
            return dash.no_update, dash.no_update, dash.no_update

        # Initialize data if needed
        if custom_indicators is None:
            custom_indicators = {"moving_avgs": []}

        # Get the current moving averages list
        if "moving_avgs" not in custom_indicators:
            custom_indicators["moving_avgs"] = []

        # Determine which button was clicked
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Start with current options
        if not current_options:
            current_options = []

        # Add the new moving average based on the button clicked
        if button_id == "add-sma-button" and sma_period:
            # Check if SMA already exists
            sma_value = f"sma_{sma_period}"
            sma_exists = any(opt["value"] == sma_value for opt in current_options)

            if not sma_exists and sma_period >= 2:
                # Add new SMA
                custom_indicators["moving_avgs"].append(
                    {"type": "sma", "period": sma_period}
                )
                # Add to options
                current_options.append(
                    {"label": f"SMA-{sma_period}", "value": sma_value}
                )
                # Add to selected values
                if current_ma_values is None:
                    current_ma_values = []
                current_ma_values.append(sma_value)

        elif button_id == "add-ema-button" and ema_period:
            # Check if EMA already exists
            ema_value = f"ema_{ema_period}"
            ema_exists = any(opt["value"] == ema_value for opt in current_options)

            if not ema_exists and ema_period >= 2:
                # Add new EMA
                custom_indicators["moving_avgs"].append(
                    {"type": "ema", "period": ema_period}
                )
                # Add to options
                current_options.append(
                    {"label": f"EMA-{ema_period}", "value": ema_value}
                )
                # Add to selected values
                if current_ma_values is None:
                    current_ma_values = []
                current_ma_values.append(ema_value)

        # Sort options by type and period
        current_options.sort(
            key=lambda x: (x["value"].split("_")[0], int(x["value"].split("_")[1]))
        )

        return current_options, current_ma_values, custom_indicators

    # Store custom indicator parameters
    @app.callback(
        Output("custom-indicators-store", "data", allow_duplicate=True),
        [
            Input("bb-period-input", "value"),
            Input("bb-stddev-input", "value"),
            Input("rsi-period-input", "value"),
        ],
        [
            State("custom-indicators-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_custom_indicators(
        bb_period,
        bb_stddev,
        rsi_period,
        custom_indicators,
    ):
        # Initialize data if needed - preserve existing moving averages
        if custom_indicators is None:
            custom_indicators = {
                "moving_avgs": [],
                "bb_period": 20,
                "bb_stddev": 2,
                "rsi_period": 14,
            }
        elif "moving_avgs" not in custom_indicators:
            custom_indicators["moving_avgs"] = []

        # Context to determine which input triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update

        # Get the changed parameter
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Update only the changed parameter
        if input_id == "bb-period-input" and bb_period is not None:
            custom_indicators["bb_period"] = bb_period
        elif input_id == "bb-stddev-input" and bb_stddev is not None:
            custom_indicators["bb_stddev"] = bb_stddev
        elif input_id == "rsi-period-input" and rsi_period is not None:
            custom_indicators["rsi_period"] = rsi_period

        return custom_indicators

    # Main callback to update all charts
    @app.callback(
        [
            Output("chart-data-store", "data"),
            Output("main-chart", "figure"),
            Output("volatility-chart", "figure"),
            Output("price-statistics", "children"),
            Output("price-data-table", "children"),
        ],
        [
            Input("metals-dropdown", "value"),
            Input("source-dropdown", "value"),
            Input("timeframe-dropdown", "value"),
            Input("time-period-dropdown", "value"),
            Input("chart-type-dropdown", "value"),
            Input("dashboard-custom-date-range", "start_date"),
            Input("dashboard-custom-date-range", "end_date"),
            Input("price-series-checklist", "value"),
            Input("moving-avg-checklist", "value"),
            Input("bb-checklist", "value"),
            Input("other-indicators-checklist", "value"),
            Input("custom-indicators-store", "data"),
        ],
        prevent_initial_call="initial_duplicate",  # Use initial_duplicate for initial load with duplicates
    )
    def update_dashboard(
        metals,
        source,
        timeframe,
        time_period,
        chart_type,
        custom_start_date,
        custom_end_date,
        price_series,
        moving_avgs,
        bollinger_bands,
        other_indicators,
        custom_indicators,
    ):
        try:
            # Determine which input triggered the callback
            ctx = callback_context
            if not ctx.triggered:
                # No trigger, initial load
                trigger_id = None
            else:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            logger.info(f"Dashboard update triggered by: {trigger_id}")

            # Handle empty metals list - show empty charts instead of error
            if not metals:
                message = "Не выбраны металлы"
                empty_fig = {
                    "data": [],
                    "layout": {
                        "title": message,
                        "xaxis": {"title": ""},
                        "yaxis": {"title": ""},
                        "annotations": [
                            {
                                "text": message,
                                "xref": "paper",
                                "yref": "paper",
                                "showarrow": False,
                                "font": {"size": 20},
                                "xanchor": "center",
                                "yanchor": "middle",
                                "x": 0.5,
                                "y": 0.5,
                            }
                        ],
                    },
                }
                empty_stats = html.Div(
                    html.Div(
                        message,
                        style={
                            "textAlign": "center",
                            "fontSize": "20px",
                            "padding": "50px",
                        },
                    )
                )
                empty_table = html.Div(
                    html.Div(
                        message,
                        style={
                            "textAlign": "center",
                            "fontSize": "20px",
                            "padding": "50px",
                        },
                    )
                )
                return None, empty_fig, empty_fig, empty_stats, empty_table

            # Ensure timeframe has a valid value
            if not timeframe or timeframe not in ["1H", "4H", "1D"]:
                timeframe = "1D"  # Default to daily timeframe

            # Calculate date range based on time period
            end_date = datetime.now()

            if time_period == "custom":
                if not custom_start_date or not custom_end_date:
                    error_msg = "Custom date range requires both start and end dates"
                    empty_fig = {
                        "data": [],
                        "layout": {
                            "title": error_msg,
                            "xaxis": {"title": ""},
                            "yaxis": {"title": ""},
                        },
                    }
                    return (
                        None,
                        empty_fig,
                        empty_fig,
                        html.Div(error_msg),
                        html.Div(error_msg),
                    )

                start_date = datetime.strptime(custom_start_date, "%Y-%m-%d")
                end_date = datetime.strptime(custom_end_date, "%Y-%m-%d")
            else:
                days_map = {
                    "1D": 1,
                    "7D": 7,
                    "30D": 30,
                    "90D": 90,
                    "180D": 180,
                    "365D": 365,
                }
                days = days_map.get(time_period, 30)
                start_date = end_date - timedelta(days=days)

            # Log all parameters for debugging
            logger.info(
                f"Getting data with params: metals={metals}, source={source}, start_date={start_date}, end_date={end_date}, timeframe={timeframe}"
            )

            # Get data
            df, error = get_price_data(
                metals, source, start_date, end_date, timeframe=timeframe
            )

            if error:
                # Return empty figures with error message
                empty_fig = {
                    "data": [],
                    "layout": {
                        "title": error,
                        "xaxis": {"title": ""},
                        "yaxis": {"title": ""},
                    },
                }
                empty_table = html.Div(f"Error: {error}")
                return None, empty_fig, empty_fig, html.Div(error), empty_table

            if df is None or df.empty:
                error_msg = "No data available for the selected criteria"
                empty_fig = {
                    "data": [],
                    "layout": {
                        "title": error_msg,
                        "xaxis": {"title": ""},
                        "yaxis": {"title": ""},
                    },
                }
                return (
                    None,
                    empty_fig,
                    empty_fig,
                    html.Div(error_msg),
                    html.Div(error_msg),
                )

            # Log the head of the dataframe to check the data
            logger.info(f"DataFrame head before plotting: {df.head().to_dict()}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")

            # Make a copy to avoid modifying the original data
            df = df.copy()

            # Calculate technical indicators for each metal with custom parameters
            for metal in df["metal_type"].unique():
                metal_df = calculate_indicators(
                    df[df["metal_type"] == metal].copy(), metal, custom_indicators
                )
                # Update the main dataframe with calculated indicators
                for col in metal_df.columns:
                    if col not in df.columns:
                        df[col] = None
                    # Update values for this metal only
                    df.loc[df["metal_type"] == metal, col] = metal_df[col].values

            # Store data for potential export
            stored_data = df.to_json(date_format="iso", orient="split")

            # Create main chart based on chart type
            if chart_type == "histogram":
                main_fig = create_histogram_chart(df)
            elif chart_type == "box_plot":
                main_fig = create_box_plot_chart(df)
            elif chart_type == "candlestick":
                main_fig = create_candlestick_chart(df, price_series)
            elif chart_type == "comparison":
                main_fig = create_comparison_chart(df)
            elif chart_type == "scatter":
                main_fig = create_scatter_chart(df)
            elif chart_type == "advanced":
                result = create_advanced_chart(
                    df, price_series, moving_avgs, bollinger_bands, other_indicators
                )
                # Check if result is a dictionary or a figure
                if isinstance(result, dict):
                    main_fig = result["main_fig"]
                else:
                    main_fig = result
            # Create volatility chart
            volatility_fig = create_volatility_chart(df)

            # Create statistics summary
            stats_component = create_statistics_summary(df)

            # Create data table
            table_component = create_data_table(df, timeframe=timeframe)

            return (
                stored_data,
                main_fig,
                volatility_fig,
                stats_component,
                table_component,
            )

        except Exception as e:
            # Log the full error with traceback
            logger.exception(f"Error in update_dashboard: {str(e)}")

            # Return empty figures with error message
            error_msg = f"An error occurred: {str(e)}"
            empty_fig = {
                "data": [],
                "layout": {
                    "title": error_msg,
                    "xaxis": {"title": ""},
                    "yaxis": {"title": ""},
                },
            }
            return None, empty_fig, empty_fig, html.Div(error_msg), html.Div(error_msg)

    # Initialize custom indicators on page load
    @app.callback(
        Output("custom-indicators-store", "data"),
        Input("dashboard-container", "children"),
        prevent_initial_call=False,
    )
    def initialize_custom_indicators(_):
        # Set up default indicators
        return {
            "moving_avgs": [
                {"type": "sma", "period": 10},
                {"type": "sma", "period": 20},
                {"type": "sma", "period": 50},
                {"type": "ema", "period": 10},
                {"type": "ema", "period": 20},
                {"type": "ema", "period": 50},
            ],
            "bb_period": 20,
            "bb_stddev": 2,
            "rsi_period": 14,
        }

    # Initialize moving averages checklist options based on custom indicators
    @app.callback(
        [
            Output("moving-avg-checklist", "options"),
            Output("moving-avg-checklist", "value"),
        ],
        Input("custom-indicators-store", "data"),
        prevent_initial_call=False,
    )
    def initialize_moving_avg_options(custom_indicators):
        # Standard periods for Moving Averages
        standard_sma_periods = [10, 20, 50]
        standard_ema_periods = [10, 20, 50]

        # Get existing moving averages
        moving_avgs = []
        custom_sma = None
        custom_ema = None

        if custom_indicators and "moving_avgs" in custom_indicators:
            moving_avgs = custom_indicators["moving_avgs"]

            # Find custom SMA and EMA if they exist
            for ma in moving_avgs:
                if ma["type"] == "sma" and ma["period"] not in standard_sma_periods:
                    custom_sma = ma["period"]
                elif ma["type"] == "ema" and ma["period"] not in standard_ema_periods:
                    custom_ema = ma["period"]

        # Create options for dropdown
        options = []

        # Add standard SMA options
        for period in standard_sma_periods:
            options.append({"label": f"SMA {period}", "value": f"sma_{period}"})

        # Add standard EMA options
        for period in standard_ema_periods:
            options.append({"label": f"EMA {period}", "value": f"ema_{period}"})

        # Add custom SMA if it exists
        if custom_sma:
            options.append(
                {
                    "label": f"SMA {custom_sma}",
                    "value": f"sma_{custom_sma}",
                }
            )

        # Add custom EMA if it exists
        if custom_ema:
            options.append(
                {
                    "label": f"EMA {custom_ema}",
                    "value": f"ema_{custom_ema}",
                }
            )

        # Sort options by type and period
        options.sort(
            key=lambda x: (x["value"].split("_")[0], int(x["value"].split("_")[1]))
        )

        # By default, only include custom indicators in values
        values = []
        if custom_sma:
            values.append(f"sma_{custom_sma}")
        if custom_ema:
            values.append(f"ema_{custom_ema}")

        return options, values

    # Initialize other indicators based on custom parameters
    @app.callback(
        Output("other-indicators-checklist", "value", allow_duplicate=True),
        Input("custom-indicators-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def initialize_other_indicators(custom_indicators):
        # Default to RSI active
        return ["rsi"]

    # Update RSI label based on period
    @app.callback(
        Output("other-indicators-checklist", "options"),
        Input("rsi-period-input", "value"),
        prevent_initial_call=False,
    )
    def update_rsi_label(rsi_period):
        if rsi_period is None:
            rsi_period = 14

        return [
            {"label": f"RSI ({rsi_period})", "value": "rsi"},
        ]

    @app.callback(
        Output("rsi-spacer", "style"),
        [Input("other-indicators-checklist", "value")],
    )
    def update_rsi_spacer(other_indicators):
        # Add extra space when RSI is displayed to prevent overlapping
        if "rsi" in other_indicators:
            return {"height": "210px"}
        return {"height": "10px"}  # Small height even when no RSI

    @app.callback(
        Output("main-chart", "style"),
        [Input("other-indicators-checklist", "value")],
    )
    def update_chart_height(other_indicators):
        # Increase chart height when RSI is shown
        if "rsi" in other_indicators:
            return {"height": "68vh"}
        return {"height": "54vh"}

    @app.callback(
        Output("custom-indicators-store", "data", allow_duplicate=True),
        [
            Input("sma-period-input", "value"),
            Input("ema-period-input", "value"),
        ],
        [
            State("custom-indicators-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_moving_averages(sma_period, ema_period, custom_indicators):
        # Determine which element triggered the callback
        context = callback_context
        trigger_id = context.triggered[0]["prop_id"].split(".")[0]

        # Initialize data if needed - preserve existing moving averages
        if custom_indicators is None:
            custom_indicators = {"moving_avgs": []}

        if "moving_avgs" not in custom_indicators:
            custom_indicators["moving_avgs"] = []

        moving_avgs = custom_indicators["moving_avgs"]

        # Standard periods for reference
        standard_sma_periods = [5, 10, 20, 50, 100, 200]
        standard_ema_periods = [5, 10, 20, 50, 100, 200]

        # Handle SMA period change
        if trigger_id == "sma-period-input" and sma_period is not None:
            # Check if SMA with this period already exists
            sma_exists = any(
                ma["type"] == "sma" and ma["period"] == sma_period for ma in moving_avgs
            )

            # Add new SMA if it doesn't exist and isn't a standard one
            if not sma_exists and sma_period not in standard_sma_periods:
                moving_avgs.append({"type": "sma", "period": sma_period})

        # Handle EMA period change
        elif trigger_id == "ema-period-input" and ema_period is not None:
            # Check if EMA with this period already exists
            ema_exists = any(
                ma["type"] == "ema" and ma["period"] == ema_period for ma in moving_avgs
            )

            # Add new EMA if it doesn't exist and isn't a standard one
            if not ema_exists and ema_period not in standard_ema_periods:
                moving_avgs.append({"type": "ema", "period": ema_period})

        # Update the list in custom_indicators
        custom_indicators["moving_avgs"] = moving_avgs

        return custom_indicators

    @app.callback(
        Output({"type": "download-metal-excel", "metal": ALL}, "data"),
        Input({"type": "export-metal-button", "metal": ALL}, "n_clicks"),
        [
            State("chart-data-store", "data"),
            State(
                {"type": "filtered-data-table", "metal": ALL}, "derived_virtual_data"
            ),
            State("timeframe-dropdown", "value"),  # Add timeframe state
        ],
    )
    def export_metal_data_excel(
        n_clicks_list, stored_data, filtered_data_list, timeframe
    ):
        # Get the context to determine which button was clicked
        ctx = callback_context

        if not ctx.triggered:
            return [None] * len(n_clicks_list)

        # Get the id of the button that was clicked
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        triggered_metal = json.loads(button_id)["metal"]

        # Create a list of None values for all outputs
        outputs = [None] * len(n_clicks_list)

        # Ensure timeframe has a valid value
        if not timeframe or timeframe not in ["1H", "4H", "1D"]:
            timeframe = "1D"  # Default to daily timeframe

        # Find the index of the triggered button
        for i, button_prop in enumerate(ctx.inputs_list[0]):
            # Get the metal from the button ID
            metal = button_prop["id"]["metal"]
            if metal == triggered_metal:
                # This is the button that was clicked
                if n_clicks_list[i] > 0:
                    try:
                        from loguru import logger

                        logger.info(
                            f"Excel export triggered for {metal} with timeframe {timeframe}: n_clicks={n_clicks_list[i]}"
                        )

                        # Try to use filtered data first
                        if (
                            filtered_data_list
                            and i < len(filtered_data_list)
                            and filtered_data_list[i]
                        ):
                            # We have filtered data available
                            logger.info(
                                f"Using filtered data for export: {len(filtered_data_list[i])} records"
                            )

                            # Convert to DataFrame
                            filtered_df = pd.DataFrame(filtered_data_list[i])

                            # Sort by date (newest first)
                            if "date" in filtered_df.columns:
                                filtered_df = filtered_df.sort_values(
                                    "date", ascending=False
                                )

                            # Determine filename based on timeframe
                            if timeframe == "1D":
                                filename = f"{metal.lower()}_daily_filtered_data.xlsx"
                                sheet_name = f"{metal} Daily Data"
                            else:
                                filename = (
                                    f"{metal.lower()}_{timeframe}_filtered_data.xlsx"
                                )
                                sheet_name = f"{metal} {timeframe} Data"

                            outputs[i] = dcc.send_data_frame(
                                filtered_df.to_excel,
                                filename,
                                sheet_name=sheet_name,
                                index=False,
                            )
                        elif stored_data is not None:
                            # Fall back to original data if filtered not available
                            logger.info(
                                "Filtered data not available, using original data"
                            )

                            # Parse stored data
                            df = pd.read_json(stored_data, orient="split")

                            # Filter for this metal's data
                            metal_df = df[df["metal_type"] == metal].copy()

                            # If no data is found
                            if metal_df.empty:
                                logger.warning(
                                    f"No {metal} data found for Excel export"
                                )
                                continue

                            # Process the data based on timeframe
                            if timeframe == "1D":
                                # For daily timeframe
                                metal_df["date"] = metal_df["timestamp"].dt.date

                                # Group by date for daily OHLC values
                                processed_df = (
                                    metal_df.groupby("date")
                                    .agg(
                                        {
                                            "open_price": "first",
                                            "high_price": "max",
                                            "low_price": "min",
                                            "close_price": "last",
                                        }
                                    )
                                    .reset_index()
                                )

                                # Calculate change
                                processed_df["price_change"] = (
                                    processed_df["close_price"]
                                    - processed_df["open_price"]
                                )
                                processed_df["price_change_pct"] = (
                                    processed_df["price_change"]
                                    / processed_df["open_price"]
                                ) * 100

                                # Sort by date (newest first)
                                processed_df = processed_df.sort_values(
                                    "date", ascending=False
                                )

                                filename = f"{metal.lower()}_daily_price_data.xlsx"
                                sheet_name = f"{metal} Daily Prices"
                            else:
                                # For hourly timeframes, keep the original timestamp
                                processed_df = metal_df.copy()

                                # Rename timestamp column for clarity
                                processed_df = processed_df.rename(
                                    columns={"timestamp": "date"}
                                )

                                # Calculate change
                                processed_df["price_change"] = (
                                    processed_df["close_price"]
                                    - processed_df["open_price"]
                                )
                                processed_df["price_change_pct"] = (
                                    processed_df["price_change"]
                                    / processed_df["open_price"]
                                ) * 100

                                # Sort by timestamp (newest first)
                                processed_df = processed_df.sort_values(
                                    "date", ascending=False
                                )

                                filename = (
                                    f"{metal.lower()}_{timeframe}_price_data.xlsx"
                                )
                                sheet_name = f"{metal} {timeframe} Prices"

                            logger.info(
                                f"Exporting {metal} data to Excel with timeframe {timeframe}: {len(processed_df)} records"
                            )

                            outputs[i] = dcc.send_data_frame(
                                processed_df.to_excel,
                                filename,
                                sheet_name=sheet_name,
                                index=False,
                            )
                        else:
                            logger.warning("No data available for export")

                    except Exception as e:
                        logger.error(f"Error exporting {metal} data to Excel: {e}")
                break

        return outputs


def create_volatility_chart(df):
    # Create a dataframe for volatility
    volatility_df = df.copy()

    # Ensure we're using a clean DataFrame
    if not isinstance(volatility_df.index, pd.RangeIndex):
        volatility_df = volatility_df.reset_index()

    # Calculate volatility if it doesn't exist
    if "volatility" not in volatility_df.columns:
        # Group by metal type to calculate volatility for each metal separately
        metals = volatility_df["metal_type"].unique()
        for metal in metals:
            # Get data for this metal
            metal_mask = volatility_df["metal_type"] == metal
            metal_data = volatility_df[metal_mask].copy()

            # Sort by timestamp
            metal_data = metal_data.sort_values("timestamp")

            # Calculate daily returns (percentage change)
            metal_data["daily_return"] = metal_data["close_price"].pct_change() * 100

            # Calculate 7-day rolling volatility (standard deviation of returns)
            metal_data["volatility"] = (
                metal_data["daily_return"].rolling(window=7).std()
            )

            # Update the main dataframe
            volatility_df.loc[metal_mask, "volatility"] = metal_data[
                "volatility"
            ].values

    # Remove NaN values to prevent plotting issues
    volatility_df = volatility_df.dropna(subset=["volatility"])

    # Create a new dataframe with only the columns we need for this chart
    plot_df = volatility_df[["timestamp", "metal_type", "volatility"]].copy()

    # Create volatility chart
    fig = px.line(
        plot_df,
        x="timestamp",
        y="volatility",
        color="metal_type",
        title="7-Day Rolling Volatility",
        labels={
            "volatility": "Volatility (% Std Dev)",
            "timestamp": "Date",
            "metal_type": "Metal",
        },
    )

    # Find a reasonable y-axis range based on the data
    max_vol = plot_df["volatility"].max()
    y_max = max(max_vol * 1.2, 10)  # At least 10% or up to 20% above max value

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        yaxis=dict(
            title="Volatility (% Std Dev)",
            range=[0, y_max],  # Set Y-axis range from 0 to calculated max
        ),
    )

    return fig


def create_statistics_summary(df):
    # Create a statistics summary for each metal
    metals = df["metal_type"].unique()

    # Create a list of cards for each metal
    cards = []

    for metal in metals:
        # Filter data for this metal only
        metal_df = df[df["metal_type"] == metal].copy()

        # Calculate statistics only from price columns
        try:
            # Calculate statistics
            current_price = metal_df["close_price"].iloc[-1]
            avg_price = metal_df["close_price"].mean()
            min_price = metal_df["close_price"].min()
            max_price = metal_df["close_price"].max()

            # Calculate price change
            first_price = metal_df["close_price"].iloc[0]
            price_change = current_price - first_price
            price_change_pct = (price_change / first_price) * 100

            # Calculate volatility (standard deviation of daily returns)
            daily_returns = metal_df["close_price"].pct_change() * 100
            volatility = daily_returns.std()

            # Create card
            card = dbc.Card(
                [
                    dbc.CardHeader(metal, className="text-center fw-bold"),
                    dbc.CardBody(
                        [
                            html.Table(
                                [
                                    html.Tr(
                                        [
                                            html.Td("Current Price:"),
                                            html.Td(f"${current_price:.2f}"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Average Price:"),
                                            html.Td(f"${avg_price:.2f}"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Min/Max Price:"),
                                            html.Td(
                                                f"${min_price:.2f} / ${max_price:.2f}"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Price Change:"),
                                            html.Td(
                                                [
                                                    f"${price_change:.2f} ",
                                                    html.Span(
                                                        f"({price_change_pct:.2f}%)",
                                                        style={
                                                            "color": (
                                                                "green"
                                                                if price_change_pct >= 0
                                                                else "red"
                                                            )
                                                        },
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Volatility:"),
                                            html.Td(f"{volatility:.2f}%"),
                                        ]
                                    ),
                                ],
                                style={"width": "100%"},
                            )
                        ]
                    ),
                ],
                className="mb-3",
            )

            cards.append(card)
        except Exception as e:
            # If there's an error calculating statistics, log it and skip this metal
            logger.error(f"Error calculating statistics for {metal}: {e}")
            continue

    # Arrange cards in rows
    rows = []
    for i in range(0, len(cards), 2):
        row_cards = cards[i : i + 2]
        row = dbc.Row([dbc.Col(card, width=6) for card in row_cards])
        rows.append(row)

    return html.Div(rows)


def create_advanced_chart(
    df, price_series, moving_avgs, bollinger_bands, other_indicators
):
    """
    Create an advanced chart with multiple indicators and subplots

    Returns:
        dict or Figure: Either a dictionary containing main figure and RSI figure (if RSI is selected)
                        or a single Figure for simple charts
    """
    try:
        # Create a copy of the dataframe to avoid modifications
        plot_df = df.copy()

        # Make sure we have a date column available
        if "date" not in plot_df.columns and "timestamp" in plot_df.columns:
            plot_df["date"] = plot_df["timestamp"]

        # Determine unique metals in the data
        metals = plot_df["metal_type"].unique()

        # Determine if we should show RSI
        show_rsi = "rsi" in other_indicators

        # For multiple metals with RSI, create a chart with subplots
        if len(metals) > 1 and show_rsi:
            # Create a figure with subplots for RSI
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3],
                subplot_titles=["Metals Price Comparison", "RSI"],
            )

            # Add price series for each metal
            for metal in metals:
                metal_df = plot_df[plot_df["metal_type"] == metal].copy()

                # Drop any NaN values for close price
                metal_df = metal_df.dropna(subset=["close_price"])

                if not metal_df.empty:
                    # Add close price line for each metal
                    fig.add_trace(
                        go.Scatter(
                            x=metal_df["date"],
                            y=metal_df["close_price"],
                            mode="lines",
                            name=f"{metal}",
                        ),
                        row=1,
                        col=1,
                    )

                    # Add moving averages for each metal
                    for ma in moving_avgs:
                        # Handle both string format and dictionary format
                        if isinstance(ma, dict):
                            ma_type = ma["type"]
                            period = ma["period"]
                            column_name = f"{ma_type}_{period}"
                            ma_type_upper = ma_type.upper()
                        else:
                            # Legacy string format (e.g. "sma_20")
                            column_name = ma
                            try:
                                ma_type, period = ma.split("_")
                                ma_type_upper = ma_type.upper()
                            except (ValueError, AttributeError):
                                # Skip this MA if it can't be parsed
                                continue

                        # Check if the column exists in the dataframe
                        if column_name in metal_df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=metal_df["date"],
                                    y=metal_df[column_name],
                                    mode="lines",
                                    name=f"{metal} {ma_type_upper}-{period}",
                                    line={"width": 1.5},
                                ),
                                row=1,
                                col=1,
                            )

                    # Add Bollinger Bands
                    if "bb" in bollinger_bands:
                        if all(
                            col in metal_df.columns
                            for col in ["bb_upper", "bb_middle", "bb_lower"]
                        ):
                            # Upper band
                            fig.add_trace(
                                go.Scatter(
                                    x=metal_df["date"],
                                    y=metal_df["bb_upper"],
                                    mode="lines",
                                    name=f"{metal} BB Upper",
                                    line={
                                        "width": 1,
                                        "color": "rgba(31, 119, 180, 0.3)",
                                    },
                                ),
                                row=1,
                                col=1,
                            )

                            # Middle band
                            fig.add_trace(
                                go.Scatter(
                                    x=metal_df["date"],
                                    y=metal_df["bb_middle"],
                                    mode="lines",
                                    name=f"{metal} BB Middle",
                                    line={
                                        "width": 1,
                                        "color": "rgba(31, 119, 180, 0.7)",
                                    },
                                ),
                                row=1,
                                col=1,
                            )

                            # Lower band
                            fig.add_trace(
                                go.Scatter(
                                    x=metal_df["date"],
                                    y=metal_df["bb_lower"],
                                    mode="lines",
                                    name=f"{metal} BB Lower",
                                    line={
                                        "width": 1,
                                        "color": "rgba(31, 119, 180, 0.3)",
                                    },
                                ),
                                row=1,
                                col=1,
                            )

                    # Add RSI
                    rsi_df = metal_df.dropna(subset=["rsi"])
                    if not rsi_df.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=rsi_df["date"],
                                y=rsi_df["rsi"],
                                mode="lines",
                                name=f"{metal} RSI",
                                line=dict(dash="solid"),
                            ),
                            row=2,
                            col=1,
                        )

            # Add RSI reference lines
            rsi_data = plot_df.dropna(subset=["rsi"])
            if not rsi_data.empty:
                min_date = rsi_data["date"].min()
                max_date = rsi_data["date"].max()

                # Add overbought/oversold lines
                fig.add_shape(
                    type="line",
                    x0=min_date,
                    x1=max_date,
                    y0=70,
                    y1=70,
                    line=dict(color="red", width=1, dash="dash"),
                    row=2,
                    col=1,
                )

                fig.add_shape(
                    type="line",
                    x0=min_date,
                    x1=max_date,
                    y0=30,
                    y1=30,
                    line=dict(color="green", width=1, dash="dash"),
                    row=2,
                    col=1,
                )

                # Add 50 level line
                fig.add_shape(
                    type="line",
                    x0=min_date,
                    x1=max_date,
                    y0=50,
                    y1=50,
                    line=dict(color="gray", width=0.5, dash="dot"),
                    row=2,
                    col=1,
                )

            # Update Y-axis labels
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

            # Update layout
            fig.update_layout(
                title="Advanced Price Analysis with RSI",
                xaxis_title="Date",
                height=800,  # Fixed taller height for RSI
                margin=dict(l=40, r=40, t=50, b=40),
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            )

            return fig

        # For single metal charts or multiple metals without RSI, use standard approach
        # Create a base figure with subplots if RSI is selected
        if show_rsi:
            # Create figure with two subplots
            main_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3],
                subplot_titles=(
                    ["Price Chart", "RSI"]
                    if len(metals) > 1
                    else [f"Price Chart - {metals[0]}", "RSI"]
                ),
            )
        else:
            # Create a main figure with just one plot
            main_fig = make_subplots(rows=1, cols=1)

        # Track the y-axis range to ensure consistent scales
        y_min, y_max = float("inf"), float("-inf")

        # Add traces for each metal
        for metal in metals:
            metal_df = plot_df[plot_df["metal_type"] == metal]

            # Add price series based on selection
            if "close" in price_series:
                main_fig.add_trace(
                    go.Scatter(
                        x=metal_df["date"],
                        y=metal_df["close_price"],
                        mode="lines",
                        name=f"{metal} Close",
                        line={"width": 1.5},
                    ),
                    row=1,
                    col=1,
                )
                y_min = min(y_min, metal_df["close_price"].min())
                y_max = max(y_max, metal_df["close_price"].max())

            if "open" in price_series:
                main_fig.add_trace(
                    go.Scatter(
                        x=metal_df["date"],
                        y=metal_df["open_price"],
                        mode="lines",
                        name=f"{metal} Open",
                        line={"width": 1, "dash": "dot"},
                    ),
                    row=1,
                    col=1,
                )
                y_min = min(y_min, metal_df["open_price"].min())
                y_max = max(y_max, metal_df["open_price"].max())

            if "high" in price_series:
                main_fig.add_trace(
                    go.Scatter(
                        x=metal_df["date"],
                        y=metal_df["high_price"],
                        mode="lines",
                        name=f"{metal} High",
                        line={"width": 1, "dash": "dash"},
                    ),
                    row=1,
                    col=1,
                )
                y_min = min(y_min, metal_df["high_price"].min())
                y_max = max(y_max, metal_df["high_price"].max())

            if "low" in price_series:
                main_fig.add_trace(
                    go.Scatter(
                        x=metal_df["date"],
                        y=metal_df["low_price"],
                        mode="lines",
                        name=f"{metal} Low",
                        line={"width": 1, "dash": "dashdot"},
                    ),
                    row=1,
                    col=1,
                )
                y_min = min(y_min, metal_df["low_price"].min())
                y_max = max(y_max, metal_df["low_price"].max())

            # Add moving averages
            for ma in moving_avgs:
                # Handle both string format and dictionary format
                if isinstance(ma, dict):
                    ma_type = ma["type"]
                    period = ma["period"]
                    column_name = f"{ma_type}_{period}"
                    ma_type_upper = ma_type.upper()
                else:
                    # Legacy string format (e.g. "sma_20")
                    column_name = ma
                    try:
                        ma_type, period = ma.split("_")
                        ma_type_upper = ma_type.upper()
                    except (ValueError, AttributeError):
                        # Skip this MA if it can't be parsed
                        continue

                # Check if the column exists in the dataframe
                if column_name in metal_df.columns:
                    main_fig.add_trace(
                        go.Scatter(
                            x=metal_df["date"],
                            y=metal_df[column_name],
                            mode="lines",
                            name=f"{metal} {ma_type_upper}-{period}",
                            line={"width": 1.5},
                        ),
                        row=1,
                        col=1,
                    )

            # Add Bollinger Bands
            if "bb" in bollinger_bands:
                if all(
                    col in metal_df.columns
                    for col in ["bb_upper", "bb_middle", "bb_lower"]
                ):
                    # Upper band
                    main_fig.add_trace(
                        go.Scatter(
                            x=metal_df["date"],
                            y=metal_df["bb_upper"],
                            mode="lines",
                            name=f"{metal} BB Upper",
                            line={"width": 1, "color": "rgba(31, 119, 180, 0.3)"},
                        ),
                        row=1,
                        col=1,
                    )

                    # Middle band (typically the SMA)
                    main_fig.add_trace(
                        go.Scatter(
                            x=metal_df["date"],
                            y=metal_df["bb_middle"],
                            mode="lines",
                            name=f"{metal} BB Middle",
                            line={"width": 1, "color": "rgba(31, 119, 180, 0.7)"},
                        ),
                        row=1,
                        col=1,
                    )

                    # Lower band
                    main_fig.add_trace(
                        go.Scatter(
                            x=metal_df["date"],
                            y=metal_df["bb_lower"],
                            mode="lines",
                            name=f"{metal} BB Lower",
                            line={"width": 1, "color": "rgba(31, 119, 180, 0.3)"},
                            fill="tonexty",
                            fillcolor="rgba(31, 119, 180, 0.1)",
                        ),
                        row=1,
                        col=1,
                    )

            # Add RSI if selected
            if show_rsi and "rsi" in metal_df.columns:
                rsi_df = metal_df.dropna(subset=["rsi"])

                if not rsi_df.empty:
                    # Add RSI line
                    main_fig.add_trace(
                        go.Scatter(
                            x=rsi_df["date"],
                            y=rsi_df["rsi"],
                            mode="lines",
                            name=f"{metal} RSI",
                            line={"width": 1.5},
                        ),
                        row=2,
                        col=1,
                    )

        # If we're showing RSI, add reference lines
        if show_rsi:
            # Get the overall date range
            date_range = plot_df["date"]
            min_date = date_range.min()
            max_date = date_range.max()

            # Add overbought line (70)
            main_fig.add_shape(
                type="line",
                x0=min_date,
                x1=max_date,
                y0=70,
                y1=70,
                line=dict(dash="dash", color="red", width=1),
                row=2,
                col=1,
            )

            # Add oversold line (30)
            main_fig.add_shape(
                type="line",
                x0=min_date,
                x1=max_date,
                y0=30,
                y1=30,
                line=dict(dash="dash", color="green", width=1),
                row=2,
                col=1,
            )

            # Add middle line (50)
            main_fig.add_shape(
                type="line",
                x0=min_date,
                x1=max_date,
                y0=50,
                y1=50,
                line=dict(dash="dot", color="grey", width=1),
                row=2,
                col=1,
            )

            # Update RSI y-axis range
            main_fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        # Update layout for the main figure
        title = "Advanced Price Chart"
        if len(metals) == 1:
            title += f" - {metals[0]}"

        height = 700 if show_rsi else 500

        main_fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            height=height,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        if show_rsi:
            # Create a single combined figure
            return main_fig
        else:
            # For non-RSI cases, return the main figure
            return main_fig

    except Exception as e:
        from loguru import logger

        logger.exception(f"Error creating advanced chart: {str(e)}")
        # Return a simple error chart
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return error_fig


def create_data_table(df, timeframe="1D"):
    """
    Create an interactive data table with price information according to the selected timeframe

    Args:
        df (pd.DataFrame): DataFrame containing the price data
        timeframe (str): The selected timeframe ("1H", "4H", or "1D")
    """
    # Process the dataframe for display
    display_df = df.copy()

    # Convert timestamp to date for grouping
    if timeframe == "1D":
        # For daily timeframe, aggregate by date
        display_df["date"] = display_df["timestamp"].dt.date

        # Group by date and metal_type for daily values
        result_df = (
            display_df.groupby(["date", "metal_type"])
            .agg(
                {
                    "open_price": "first",
                    "high_price": "max",
                    "low_price": "min",
                    "close_price": "last",
                }
            )
            .reset_index()
        )

        # Sort by date (newest first) and metal type
        result_df = result_df.sort_values(
            ["date", "metal_type"], ascending=[False, True]
        )

        # Date label for table header
        date_label = "Date"
    else:
        # For hourly timeframes, use the full timestamp
        result_df = display_df.copy()

        # Rename timestamp column to date for consistency with the table structure
        result_df = result_df.rename(columns={"timestamp": "date"})

        # Sort by timestamp (newest first) and metal type
        result_df = result_df.sort_values(
            ["date", "metal_type"], ascending=[False, True]
        )

        # Date label with timeframe indicator
        date_label = f"Timestamp ({timeframe})"

    # Calculate change
    result_df["price_change"] = result_df["close_price"] - result_df["open_price"]
    result_df["price_change_pct"] = result_df["price_change"] / result_df["open_price"]

    # Create a list of DataTable for each metal
    tables = []

    # Create an exports section for all metals
    export_section = html.Div(
        [
            html.H5("Exports", className="mt-4 mb-3"),
            html.Div(
                id="export-buttons-container",
                className="d-flex flex-wrap gap-2 mb-3",
                children=[],  # Initialize with empty list to avoid NoneType error
            ),
        ]
    )

    # Create export buttons for each metal
    export_buttons = []

    # Add data stores for filtered table data
    stores = []

    for metal in result_df["metal_type"].unique():
        metal_data = result_df[result_df["metal_type"] == metal]

        # Format the data for the DataTable
        metal_records = metal_data.to_dict("records")

        # Create store for filtered data
        store_id = f"filtered-data-{metal.lower().replace(' ', '-')}"
        stores.append(dcc.Store(id=store_id, data=metal_records))

        # Title with timeframe info
        title = f"{metal} Price Data ({timeframe})"

        # Create the table
        table = html.Div(
            [
                html.H5(title, className="mt-3 mb-2"),
                dash_table.DataTable(
                    id={"type": "filtered-data-table", "metal": metal},
                    columns=[
                        {"name": date_label, "id": "date"},
                        {
                            "name": "Open",
                            "id": "open_price",
                            "type": "numeric",
                            "format": {"specifier": "$.2f"},
                        },
                        {
                            "name": "High",
                            "id": "high_price",
                            "type": "numeric",
                            "format": {"specifier": "$.2f"},
                        },
                        {
                            "name": "Low",
                            "id": "low_price",
                            "type": "numeric",
                            "format": {"specifier": "$.2f"},
                        },
                        {
                            "name": "Close",
                            "id": "close_price",
                            "type": "numeric",
                            "format": {"specifier": "$.2f"},
                        },
                        {
                            "name": "Change",
                            "id": "price_change",
                            "type": "numeric",
                            "format": {"specifier": "+$.2f"},
                        },
                        {
                            "name": "Change %",
                            "id": "price_change_pct",
                            "type": "numeric",
                            "format": {"specifier": "+.2%"},
                        },
                    ],
                    data=metal_records,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "right",
                        "padding": "5px",
                        "backgroundColor": "transparent",
                    },
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "lightgray",
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": "{price_change} < 0"},
                            "color": "red",
                        },
                        {
                            "if": {"filter_query": "{price_change} > 0"},
                            "color": "green",
                        },
                    ],
                    page_size=10,  # Show 10 rows per page
                    filter_action="native",  # Allow filtering
                    sort_action="native",  # Allow sorting
                    sort_mode="multi",  # Allow sorting by multiple columns
                    style_as_list_view=True,
                    # Add callback on filtered and sorted data
                    persistence=True,  # Allow persisting filter settings
                    persistence_type="session",  # Store in session
                ),
            ]
        )

        tables.append(table)

        # Create export button for this metal
        export_button = dbc.Button(
            f"Export {metal} Data as Excel",
            id={"type": "export-metal-button", "metal": metal},
            color="success",
            className="me-2 mb-2",
            n_clicks=0,
        )

        # Add to our list of export buttons
        export_buttons.append(export_button)

    # Set all the export buttons at once
    export_section.children[1].children = export_buttons

    # Add export section at the end
    tables.append(export_section)

    # Add stores for filtered data
    for store in stores:
        tables.append(store)

    # Add multiple download components for each metal
    for metal in result_df["metal_type"].unique():
        tables.append(dcc.Download(id={"type": "download-metal-excel", "metal": metal}))

    return html.Div(tables)


def create_histogram_chart(df):
    """Create a histogram chart showing price distribution"""
    # Create a copy of the dataframe
    plot_df = df.copy()

    # Create a histogram chart using plotly express
    fig = px.histogram(
        plot_df,
        x="close_price",
        color="metal_type",
        nbins=30,
        opacity=0.7,
        title="Price Distribution",
        labels={
            "close_price": "Price (USD)",
            "count": "Frequency",
            "metal_type": "Metal",
        },
        marginal="box",
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        bargap=0.1,
    )

    return fig


def create_box_plot_chart(df):
    """Create a box plot chart for price analysis"""
    # Create a copy of the dataframe
    plot_df = df.copy()

    # Create a box plot using plotly express
    fig = px.box(
        plot_df,
        y="close_price",
        x="metal_type",
        color="metal_type",
        title="Price Distribution Analysis",
        labels={
            "close_price": "Price (USD)",
            "metal_type": "Metal",
        },
        points="all",
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )

    return fig


def create_candlestick_chart(df, price_series=None):
    """Create a candlestick chart for price analysis"""
    # Price series parameter is not used for candlestick but included for API compatibility

    # Create a copy of the dataframe
    plot_df = df.copy()

    # Handle multiple metals by creating subplots
    metals = plot_df["metal_type"].unique()

    # Create subplots - one for each metal
    fig = make_subplots(
        rows=len(metals),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{metal} Price" for metal in metals],
    )

    for i, metal in enumerate(metals, 1):
        metal_df = plot_df[plot_df["metal_type"] == metal].copy()

        # Sort by timestamp to ensure data is in correct order
        metal_df = metal_df.sort_values("timestamp")

        # Filter out duplicate candles (where all OHLC values are identical to previous candle)
        filtered_df = metal_df.iloc[0:1].copy()  # Keep the first row

        for j in range(1, len(metal_df)):
            current = metal_df.iloc[j]
            previous = metal_df.iloc[j - 1]

            # Check if the current candle is different from the previous one
            if (
                current["open_price"] != previous["open_price"]
                or current["high_price"] != previous["high_price"]
                or current["low_price"] != previous["low_price"]
                or current["close_price"] != previous["close_price"]
            ):
                filtered_df = pd.concat([filtered_df, current.to_frame().T])

        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=filtered_df["timestamp"],
                open=filtered_df["open_price"],
                high=filtered_df["high_price"],
                low=filtered_df["low_price"],
                close=filtered_df["close_price"],
                name=metal,
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        # Update y-axis label
        fig.update_yaxes(title_text="Price (USD)", row=i, col=1)

    # Update layout
    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        height=max(600, 300 * len(metals)),  # Dynamic height based on number of metals
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,  # Disable rangeslider for cleaner look
    )

    return fig


def create_comparison_chart(df):
    """Create a comparison chart for multiple metals"""
    # Create a copy of the dataframe
    plot_df = df.copy()

    # Create a line chart comparing close prices
    fig = px.line(
        plot_df,
        x="timestamp",
        y="close_price",
        color="metal_type",
        title="Metal Price Comparison",
        labels={
            "close_price": "Price (USD)",
            "timestamp": "Date",
            "metal_type": "Metal",
        },
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
    )

    return fig


def create_scatter_chart(df):
    """Create a scatter plot for price analysis"""
    # Create a copy of the dataframe
    plot_df = df.copy()

    # Calculate daily returns for x-axis
    for metal in plot_df["metal_type"].unique():
        mask = plot_df["metal_type"] == metal
        plot_df.loc[mask, "daily_return"] = (
            plot_df.loc[mask, "close_price"].pct_change() * 100
        )

    # Create a scatter plot
    fig = px.scatter(
        plot_df.dropna(),  # Remove NaN values
        x="daily_return",
        y="close_price",
        color="metal_type",
        title="Price vs. Daily Return",
        labels={
            "daily_return": "Daily Return (%)",
            "close_price": "Price (USD)",
            "metal_type": "Metal",
        },
        size_max=10,
        opacity=0.7,
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
    )

    return fig
