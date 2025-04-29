from dash import html, dcc, dash_table, dash, callback_context
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, and_, func, inspect
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, date
from loguru import logger
import json
from io import StringIO

from database.init_database import (
    Base,
    MetalType,
    DataSource,
    MetalPrice,
    CollectorSchedule,
)
from settings import DB_URL
from views.components import create_theme_switch

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)


def get_available_data_sources():
    """
    Запрос в базу данных для получения всех уникальных источников данных
    """
    try:
        session = Session()

        data_sources = session.query(MetalPrice.source).distinct().all()

        options = []
        for ds in data_sources:
            data_source = ds[0]
            if isinstance(data_source, DataSource):
                options.append({"label": data_source.value, "value": data_source.name})

        if not options:
            logger.warning(
                f"No data sources found in database, using all sources as fallback"
            )
            options = [
                {"label": data_source.value, "value": data_source.name}
                for data_source in DataSource
            ]

        return options, None

    except Exception as e:
        logger.error(f"Error getting available data sources: {e}")
        return [
            {"label": data_source.value, "value": data_source.name}
            for data_source in DataSource
        ], f"Error: {str(e)}"

    finally:
        session.close()


def initialize_dropdown_options():
    """
    Инициализация опций выпадающих списков для металлов и источников данных
    """
    data_source_options, _ = get_available_data_sources()

    default_source = DataSource.YFINANCE.name
    if data_source_options:
        default_source = data_source_options[0]["value"]

    metal_options, _ = get_available_metals_for_source(default_source)

    return metal_options, data_source_options, default_source


try:
    initial_metal_options, initial_source_options, default_source = (
        [
            {"label": metal_type.value, "value": metal_type.name}
            for metal_type in MetalType
        ],
        [
            {"label": data_source.value, "value": data_source.name}
            for data_source in DataSource
        ],
        DataSource.YFINANCE.name,
    )
    logger.info("Using default dropdown options")
except Exception as e:
    logger.error(f"Error initializing dropdown options: {e}")
    initial_metal_options = [
        {"label": metal_type.value, "value": metal_type.name}
        for metal_type in MetalType
    ]
    initial_source_options = [
        {"label": data_source.value, "value": data_source.name}
        for data_source in DataSource
    ]
    default_source = DataSource.YFINANCE.name


def calculate_indicators(df, metal_type=None, custom_params=None):
    """Расчет различных технических индикаторов для данных"""
    result_df = df.copy()

    if metal_type:
        mask = result_df["metal_type"] == metal_type
        temp_df = result_df[mask].copy()
    else:
        temp_df = result_df.copy()

    if temp_df.empty:
        return result_df

    temp_df = temp_df.sort_values("timestamp")

    if custom_params is None:
        custom_params = {}

    custom_mas = custom_params.get("moving_avgs", [])
    bb_period = custom_params.get("bb_period", 20)
    bb_stddev = custom_params.get("bb_stddev", 2)
    rsi_period = custom_params.get("rsi_period", 14)

    for ma_entry in custom_mas:
        if ma_entry["type"] == "sma":
            window = ma_entry["period"]
            col_name = f"sma_{window}"
            temp_df[col_name] = temp_df["close_price"].rolling(window=window).mean()

    for ma_entry in custom_mas:
        if ma_entry["type"] == "ema":
            window = ma_entry["period"]
            col_name = f"ema_{window}"
            temp_df[col_name] = (
                temp_df["close_price"].ewm(span=window, adjust=False).mean()
            )

    bb_col = f"sma_{bb_period}"
    # Добавление  SMA if it doesn't exist yet
    if bb_col not in temp_df.columns:
        temp_df[bb_col] = temp_df["close_price"].rolling(window=bb_period).mean()

    # Use  SMA as средняя полоса
    temp_df["bb_middle"] = temp_df[bb_col]
    temp_df["bb_stddev"] = temp_df["close_price"].rolling(window=bb_period).std()
    temp_df["bb_upper"] = temp_df["bb_middle"] + bb_stddev * temp_df["bb_stddev"]
    temp_df["bb_lower"] = temp_df["bb_middle"] - bb_stddev * temp_df["bb_stddev"]

    temp_df["price_change"] = temp_df["close_price"].diff()

    temp_df["gain"] = np.where(temp_df["price_change"] > 0, temp_df["price_change"], 0)
    temp_df["loss"] = np.where(temp_df["price_change"] < 0, -temp_df["price_change"], 0)

    temp_df["avg_gain"] = temp_df["gain"].rolling(window=rsi_period).mean()
    temp_df["avg_loss"] = temp_df["loss"].rolling(window=rsi_period).mean()

    temp_df["rs"] = temp_df["avg_gain"] / temp_df["avg_loss"]

    temp_df["rsi"] = 100 - (100 / (1 + temp_df["rs"]))

    temp_df["rsi"] = np.clip(temp_df["rsi"], 0, 100)

    if metal_type:
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
        dcc.Store(id="theme-store", storage_type="local"),
        dcc.Store(id="user-preferences", storage_type="local"),
        dcc.Store(id="metals-fetched-store", storage_type="session"),
        dcc.Store(id="custom-indicators-store", storage_type="session"),
        dcc.Store(id="chart-data-store"),
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard/", active=True)),
                dbc.NavItem(
                    dbc.NavLink("Data Collection", href="/dashboard/data-collection")
                ),
                dbc.NavItem(
                    dbc.NavLink("Database Management", href="/dashboard/database")
                ),
                dbc.NavItem(
                    dbc.Switch(
                        id="theme-switch",
                        label="Dark Mode",
                        value=True,
                        className="ms-auto mt-2 mb-2",
                    ),
                    className="ms-auto d-flex align-items-center",
                ),
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
                # панели Controls
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Metals"),
                                dcc.Dropdown(
                                    id="metals-dropdown",
                                    options=initial_metal_options,
                                    value=(
                                        [initial_metal_options[0]["value"]]
                                        if initial_metal_options
                                        else []
                                    ),
                                    multi=True,
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Select Data Source"),
                                dcc.Dropdown(
                                    id="source-dropdown",
                                    options=initial_source_options,
                                    value=default_source,
                                ),
                            ],
                            width=4,
                        ),
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
                dbc.Row(
                    [
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
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="technical-indicators-container",
                                    children=[
                                        html.H5("Technical Indicators"),
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
                                        # скользящих средних
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
                                        # Полосы Боллинджера
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
                # Main графика
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
                html.Div(id="rsi-spacer", style={"height": "0px"}),
                dbc.Row(
                    [
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
            ],
            id="dashboard-container",
            className="ag-theme-quartz-dark",
            fluid=True,
        ),
    ],
    id="main-layout",
    className="main-container",
)


def get_price_data(metals, source, start_date, end_date, timeframe="1D"):
    try:
        session = Session()

        if isinstance(metals, list) and all(isinstance(m, str) for m in metals):
            metals = [MetalType[m] for m in metals]
        elif isinstance(metals, str):
            metals = [MetalType[metals]]

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

        results = query.all()

        data = []
        for result in results:
            metal_type, timestamp, open_price, high_price, low_price, close_price = (
                result
            )

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

        if df.empty:
            return None, "No data found for the selected criteria"

        if timeframe != "1D" and not df.empty:
            resample_map = {
                "1H": "1H",
                "4H": "4H",
                "1D": "1D",
            }
            rule = resample_map.get(timeframe, "1D")

            all_resampled = []

            for metal in df["metal_type"].unique():
                metal_df = df[df["metal_type"] == metal].copy()

                metal_df.set_index("timestamp", inplace=True)

                try:
                    # Resample  данных
                    resampled = metal_df.resample(rule).agg(
                        {
                            "open_price": "first",
                            "high_price": "max",
                            "low_price": "min",
                            "close_price": "last",
                        }
                    )

                    resampled.fillna(method="ffill", inplace=True)

                    resampled["metal_type"] = metal

                    resampled.reset_index(inplace=True)

                    all_resampled.append(resampled)
                except Exception as e:
                    logger.error(f"Error resampling data for {metal}: {e}")

            if all_resampled:
                df = pd.concat(all_resampled, ignore_index=True)
        elif timeframe == "1D":  # Ensure daily data is handled properly
            all_daily = []

            for metal in df["metal_type"].unique():
                metal_df = df[df["metal_type"] == metal].copy()

                metal_df["date"] = metal_df["timestamp"].dt.date

                daily = metal_df.groupby("date").agg(
                    {
                        "open_price": "first",
                        "high_price": "max",
                        "low_price": "min",
                        "close_price": "last",
                        "metal_type": "first",
                    }
                )

                daily.reset_index(inplace=True)
                daily["timestamp"] = pd.to_datetime(daily["date"])
                daily.drop("date", axis=1, inplace=True)

                all_daily.append(daily)

            if all_daily:
                df = pd.concat(all_daily, ignore_index=True)

        logger.info(f"Returning dataframe with {len(df)} rows")
        return df, None

    except Exception as e:
        logger.error(f"Error getting price data: {e}")
        return None, f"Error: {str(e)}"

    finally:
        session.close()


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

        # Query distinct metal types из  базы данных without filtering by источника
        metal_types = session.query(MetalPrice.metal_type).distinct().all()

        options = []
        for mt in metal_types:
            metal_type = mt[0]
            if isinstance(metal_type, MetalType):
                options.append({"label": metal_type.value, "value": metal_type.name})

        if not options:
            logger.warning(f"No metals found in database, using all metals as fallback")
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


def register_callbacks(app):
    @app.callback(
        Output("source-dropdown", "options"),
        Input("dashboard-container", "children"),
        prevent_initial_call=False,
    )
    def initialize_data_source_options(_):
        data_source_options, _ = get_available_data_sources()
        return data_source_options

    @app.callback(
        Output("dashboard-custom-date-range-container", "style"),
        [Input("time-period-dropdown", "value")],
    )
    def toggle_custom_date_range(time_period):
        if time_period == "custom":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        [Output("metals-dropdown", "options"), Output("metals-fetched-store", "data")],
        [Input("source-dropdown", "value")],
        [State("metals-fetched-store", "data")],
    )
    def update_metals_dropdown(source, fetched_sources):
        if fetched_sources is None:
            fetched_sources = {}

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "source-dropdown" or not fetched_sources:
            metal_options, error = get_available_metals_for_source(source)

            if source:
                fetched_sources[source] = [option["value"] for option in metal_options]

            return metal_options, fetched_sources

        return dash.no_update, fetched_sources

    @app.callback(
        Output("metals-dropdown", "value", allow_duplicate=True),
        [Input("source-dropdown", "value")],
        [State("metals-dropdown", "value"), State("metals-fetched-store", "data")],
        prevent_initial_call=True,
    )
    def update_metals_selection(source, current_selection, fetched_sources):
        return current_selection

    @app.callback(
        Output("technical-indicators-container", "style"),
        [Input("chart-type-dropdown", "value")],
    )
    def toggle_technical_indicators(chart_type):
        if chart_type == "advanced":
            return {"display": "block"}
        return {"display": "none"}

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
            # для custom, keep  current значений (will be overridden by user)
            return dash.no_update, dash.no_update

        return start_date, end_date

    @app.callback(
        [
            Output("dashboard-container", "className"),
            Output("navbar", "dark"),
            Output("navbar", "color"),
        ],
        [Input("theme-store", "data")],
        prevent_initial_call=False,
    )
    def update_dashboard_theme(theme_data):
        # Получение dark mode из Тема Хранилище
        dark_mode = theme_data.get("dark_mode", True) if theme_data else True

        if dark_mode:
            return (
                "ag-theme-quartz-dark",
                True,
                "dark",
            )
        else:
            return (
                "ag-theme-quartz",
                False,
                "light",
            )

    # Загрузка user preferences из local storage
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
            State("theme-store", "data"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def load_dashboard_preferences(
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
        theme_data,
    ):
        if fetched_sources is None:
            fetched_sources = {}

        theme_from_store = theme_data.get("dark_mode", True) if theme_data else True

        if not stored_prefs:
            return (
                metals
                or [MetalType.GOLD.name],  # Default to GOLD if no metals selected
                source or default_source,
                timeframe or "1D",
                time_period or "30D",
                chart_type or "advanced",
                theme_from_store,  # Use theme from store
                price_series or ["close"],
                moving_avgs or [],
                bb or [],
                other_indicators or [],
            )

        try:
            preferred_source = stored_prefs.get("source", source or default_source)

            stored_metals = stored_prefs.get("metals", metals or [])

            return (
                stored_metals,
                preferred_source,
                stored_prefs.get("timeframe", timeframe or "1D"),
                stored_prefs.get("time_period", time_period or "30D"),
                stored_prefs.get("chart_type", chart_type or "advanced"),
                theme_from_store,  # Use theme from store instead of preferences
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

    @app.callback(
        Output("user-preferences", "data", allow_duplicate=True),
        [
            Input("metals-dropdown", "value"),
            Input("source-dropdown", "value"),
            Input("timeframe-dropdown", "value"),
            Input("time-period-dropdown", "value"),
            Input("chart-type-dropdown", "value"),
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
    def save_dashboard_preferences(
        metals,
        source,
        timeframe,
        time_period,
        chart_type,
        price_series,
        moving_avgs,
        bb,
        other_indicators,
        current_prefs,
    ):
        ctx = callback_context

        if not ctx.triggered:
            return dash.no_update

        triggered_input = ctx.triggered[0]["prop_id"].split(".")[0]

        # Инициализация preferences if they don't exist
        if current_prefs is None:
            current_prefs = {}

        # Обновление individual field only if changed
        if triggered_input == "metals-dropdown":
            current_prefs["metals"] = metals
        elif triggered_input == "source-dropdown":
            current_prefs["source"] = source
        elif triggered_input == "timeframe-dropdown":
            current_prefs["timeframe"] = timeframe
        elif triggered_input == "time-period-dropdown":
            current_prefs["time_period"] = time_period
        elif triggered_input == "chart-type-dropdown":
            current_prefs["chart_type"] = chart_type
        elif triggered_input == "price-series-checklist":
            current_prefs["price_series"] = price_series
        elif triggered_input == "moving-avg-checklist":
            current_prefs["moving_avgs"] = moving_avgs
        elif triggered_input == "bb-checklist":
            current_prefs["bb"] = bb
        elif triggered_input == "other-indicators-checklist":
            current_prefs["other_indicators"] = other_indicators
        else:
            logger.warning(
                f"Unknown input triggered save_preferences: {triggered_input}"
            )
            return dash.no_update

        return current_prefs

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
            return dash.no_update, dash.no_update, dash.no_update

        # Инициализация данных if needed
        if custom_indicators is None:
            custom_indicators = {"moving_avgs": []}

        if "moving_avgs" not in custom_indicators:
            custom_indicators["moving_avgs"] = []

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if not current_options:
            current_options = []

        if button_id == "add-sma-button" and sma_period:
            # Проверка if SMA already exists
            sma_value = f"sma_{sma_period}"
            sma_exists = any(opt["value"] == sma_value for opt in current_options)

            if not sma_exists and sma_period >= 2:
                custom_indicators["moving_avgs"].append(
                    {"type": "sma", "period": sma_period}
                )
                # Добавление к опций
                current_options.append(
                    {"label": f"SMA-{sma_period}", "value": sma_value}
                )
                if current_ma_values is None:
                    current_ma_values = []
                current_ma_values.append(sma_value)

        elif button_id == "add-ema-button" and ema_period:
            ema_value = f"ema_{ema_period}"
            ema_exists = any(opt["value"] == ema_value for opt in current_options)

            if not ema_exists and ema_period >= 2:
                custom_indicators["moving_avgs"].append(
                    {"type": "ema", "period": ema_period}
                )
                current_options.append(
                    {"label": f"EMA-{ema_period}", "value": ema_value}
                )
                if current_ma_values is None:
                    current_ma_values = []
                current_ma_values.append(ema_value)

        current_options.sort(
            key=lambda x: (x["value"].split("_")[0], int(x["value"].split("_")[1]))
        )

        return current_options, current_ma_values, custom_indicators

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
    def update_dashboard_custom_indicators(
        bb_period,
        bb_stddev,
        rsi_period,
        custom_indicators,
    ):
        # Инициализация данных if needed - preserve existing скользящих средних
        if custom_indicators is None:
            custom_indicators = {
                "moving_avgs": [],
                "bb_period": 20,
                "bb_stddev": 2,
                "rsi_period": 14,
            }
        elif "moving_avgs" not in custom_indicators:
            custom_indicators["moving_avgs"] = []

        # Context к determine which input triggered  Колбэк
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update

        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if input_id == "bb-period-input" and bb_period is not None:
            custom_indicators["bb_period"] = bb_period
        elif input_id == "bb-stddev-input" and bb_stddev is not None:
            custom_indicators["bb_stddev"] = bb_stddev
        elif input_id == "rsi-period-input" and rsi_period is not None:
            custom_indicators["rsi_period"] = rsi_period

        return custom_indicators

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
            ctx = callback_context
            if not ctx.triggered:
                trigger_id = None
            else:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            logger.info(f"Dashboard update triggered by: {trigger_id}")

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

            if not timeframe or timeframe not in ["1H", "4H", "1D"]:
                timeframe = "1D"  # Default to daily timeframe

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

            logger.info(
                f"Getting data with params: metals={metals}, source={source}, start_date={start_date}, end_date={end_date}, timeframe={timeframe}"
            )

            # Получение данных
            df, error = get_price_data(
                metals, source, start_date, end_date, timeframe=timeframe
            )

            if error:
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

            logger.info(f"DataFrame head before plotting: {df.head().to_dict()}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")

            df = df.copy()

            for metal in df["metal_type"].unique():
                metal_df = calculate_indicators(
                    df[df["metal_type"] == metal].copy(), metal, custom_indicators
                )
                for col in metal_df.columns:
                    if col not in df.columns:
                        df[col] = None
                    df.loc[df["metal_type"] == metal, col] = metal_df[col].values

            stored_data = df.to_json(date_format="iso", orient="split")

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
                if isinstance(result, dict):
                    main_fig = result["main_fig"]
                else:
                    main_fig = result
            volatility_fig = create_volatility_chart(df)

            stats_component = create_statistics_summary(df)

            table_component = create_data_table(df, timeframe=timeframe)

            return (
                stored_data,
                main_fig,
                volatility_fig,
                stats_component,
                table_component,
            )

        except Exception as e:
            logger.exception(f"Error in update_dashboard: {str(e)}")

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

    @app.callback(
        Output("custom-indicators-store", "data"),
        Input("dashboard-container", "children"),
        prevent_initial_call=False,
    )
    def initialize_dashboard_custom_indicators(_):
        # Установка up по умолчанию индикаторов
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

    @app.callback(
        [
            Output("moving-avg-checklist", "options"),
            Output("moving-avg-checklist", "value"),
        ],
        Input("custom-indicators-store", "data"),
        prevent_initial_call=False,
    )
    def initialize_dashboard_moving_avg_options(custom_indicators):
        standard_sma_periods = [10, 20, 50]
        standard_ema_periods = [10, 20, 50]

        moving_avgs = []
        custom_sma = None
        custom_ema = None

        if custom_indicators and "moving_avgs" in custom_indicators:
            moving_avgs = custom_indicators["moving_avgs"]

            for ma in moving_avgs:
                if ma["type"] == "sma" and ma["period"] not in standard_sma_periods:
                    custom_sma = ma["period"]
                elif ma["type"] == "ema" and ma["period"] not in standard_ema_periods:
                    custom_ema = ma["period"]

        options = []

        for period in standard_sma_periods:
            options.append({"label": f"SMA {period}", "value": f"sma_{period}"})

        for period in standard_ema_periods:
            options.append({"label": f"EMA {period}", "value": f"ema_{period}"})

        if custom_sma:
            options.append(
                {
                    "label": f"SMA {custom_sma}",
                    "value": f"sma_{custom_sma}",
                }
            )

        # Добавление custom EMA if it exists
        if custom_ema:
            options.append(
                {
                    "label": f"EMA {custom_ema}",
                    "value": f"ema_{custom_ema}",
                }
            )

        options.sort(
            key=lambda x: (x["value"].split("_")[0], int(x["value"].split("_")[1]))
        )

        values = []
        if custom_sma:
            values.append(f"sma_{custom_sma}")
        if custom_ema:
            values.append(f"ema_{custom_ema}")

        return options, values

    @app.callback(
        Output("other-indicators-checklist", "value", allow_duplicate=True),
        Input("custom-indicators-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def initialize_dashboard_other_indicators(custom_indicators):
        return ["rsi"]

    @app.callback(
        Output("other-indicators-checklist", "options"),
        Input("rsi-period-input", "value"),
        prevent_initial_call=False,
    )
    def update_dashboard_rsi_label(rsi_period):
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
        # Добавление extra space when RSI is displayed к prevent overlapping
        if "rsi" in other_indicators:
            return {"height": "210px"}
        return {"height": "10px"}  # Small height even when no RSI

    @app.callback(
        Output("main-chart", "style"),
        [Input("other-indicators-checklist", "value")],
    )
    def update_chart_height(other_indicators):
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
    def update_dashboard_moving_averages(sma_period, ema_period, custom_indicators):
        context = callback_context
        trigger_id = context.triggered[0]["prop_id"].split(".")[0]

        # Инициализация данных if needed - preserve existing скользящих средних
        if custom_indicators is None:
            custom_indicators = {"moving_avgs": []}

        if "moving_avgs" not in custom_indicators:
            custom_indicators["moving_avgs"] = []

        moving_avgs = custom_indicators["moving_avgs"]

        standard_sma_periods = [5, 10, 20, 50, 100, 200]
        standard_ema_periods = [5, 10, 20, 50, 100, 200]

        if trigger_id == "sma-period-input" and sma_period is not None:
            sma_exists = any(
                ma["type"] == "sma" and ma["period"] == sma_period for ma in moving_avgs
            )

            # Добавление new SMA if it doesn't exist and isn't a standard one
            if not sma_exists and sma_period not in standard_sma_periods:
                moving_avgs.append({"type": "sma", "period": sma_period})

        elif trigger_id == "ema-period-input" and ema_period is not None:
            # Проверка if EMA with this периода already exists
            ema_exists = any(
                ma["type"] == "ema" and ma["period"] == ema_period for ma in moving_avgs
            )

            if not ema_exists and ema_period not in standard_ema_periods:
                moving_avgs.append({"type": "ema", "period": ema_period})

        custom_indicators["moving_avgs"] = moving_avgs

        return custom_indicators

    @app.callback(
        Output({"type": "download-metal-excel", "metal": ALL}, "data"),
        Input({"type": "export-metal-button", "metal": ALL}, "n_clicks"),
        [
            State("chart-data-store", "data"),
            State({"type": "filtered-data-table", "metal": ALL}, "rowData"),
            State("timeframe-dropdown", "value"),  # Add timeframe state
        ],
    )
    def export_metal_data_excel(
        n_clicks_list, stored_data, filtered_data_list, timeframe
    ):
        ctx = callback_context

        if not ctx.triggered:
            return [None] * len(n_clicks_list)

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        triggered_metal = json.loads(button_id)["metal"]

        outputs = [None] * len(n_clicks_list)

        # Ensure timeframe has a valid value
        if not timeframe or timeframe not in ["1H", "4H", "1D"]:
            timeframe = "1D"  # Default to daily timeframe

        for i, button_prop in enumerate(ctx.inputs_list[0]):
            metal = button_prop["id"]["metal"]
            if metal == triggered_metal:
                if n_clicks_list[i] > 0:
                    try:
                        from loguru import logger

                        logger.info(
                            f"Excel export triggered for {metal} with timeframe {timeframe}: n_clicks={n_clicks_list[i]}"
                        )

                        if (
                            filtered_data_list
                            and i < len(filtered_data_list)
                            and filtered_data_list[i]
                        ):
                            # We have filtered данных available
                            logger.info(
                                f"Using filtered data for export: {len(filtered_data_list[i])} records"
                            )

                            filtered_df = pd.DataFrame(filtered_data_list[i])

                            if "date" in filtered_df.columns:
                                filtered_df = filtered_df.sort_values(
                                    "date", ascending=False
                                )

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
                            logger.info(
                                "Filtered data not available, using original data"
                            )

                            df = pd.read_json(stored_data, orient="split")

                            # Фильтрация для this metal's данных
                            metal_df = df[df["metal_type"] == metal].copy()

                            if metal_df.empty:
                                logger.warning(
                                    f"No {metal} data found for Excel export"
                                )
                                continue

                            if timeframe == "1D":
                                metal_df["date"] = metal_df["timestamp"].dt.date

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

                                processed_df["price_change"] = (
                                    processed_df["close_price"]
                                    - processed_df["open_price"]
                                )
                                processed_df["price_change_pct"] = (
                                    processed_df["price_change"]
                                    / processed_df["open_price"]
                                ) * 100

                                processed_df = processed_df.sort_values(
                                    "date", ascending=False
                                )

                                filename = f"{metal.lower()}_daily_price_data.xlsx"
                                sheet_name = f"{metal} Daily Prices"
                            else:
                                # для hourly timeframes, keep  original временная метка
                                processed_df = metal_df.copy()

                                processed_df = processed_df.rename(
                                    columns={"timestamp": "date"}
                                )

                                processed_df["price_change"] = (
                                    processed_df["close_price"]
                                    - processed_df["open_price"]
                                )
                                processed_df["price_change_pct"] = (
                                    processed_df["price_change"]
                                    / processed_df["open_price"]
                                ) * 100

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
    volatility_df = df.copy()

    # Ensure we're using a clean DataFrame
    if not isinstance(volatility_df.index, pd.RangeIndex):
        volatility_df = volatility_df.reset_index()

    if "volatility" not in volatility_df.columns:
        metals = volatility_df["metal_type"].unique()
        for metal in metals:
            metal_df = volatility_df[volatility_df["metal_type"] == metal].copy()

            metal_df = metal_df.sort_values("timestamp")

            metal_df["price_change"] = metal_df["close_price"].diff()
            metal_df["volatility"] = (
                metal_df["price_change"]
                .rolling(window=20, min_periods=1)
                .std()
                .fillna(0)
            )

            volatility_df.loc[volatility_df["metal_type"] == metal, "volatility"] = (
                metal_df["volatility"].values
            )
            volatility_df.loc[volatility_df["metal_type"] == metal, "price_change"] = (
                metal_df["price_change"].values
            )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Добавление price lines для each metal
    for metal in volatility_df["metal_type"].unique():
        metal_df = volatility_df[volatility_df["metal_type"] == metal].copy()
        metal_df = metal_df.sort_values("timestamp")

        fig.add_trace(
            go.Scatter(
                x=metal_df["timestamp"],
                y=metal_df["close_price"],
                name=f"{metal} Price",
                line=dict(width=2),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=metal_df["timestamp"],
                y=metal_df["volatility"],
                name=f"{metal} Volatility",
                line=dict(width=1, dash="dot"),
                opacity=0.7,
            ),
            secondary_y=True,
        )

    # Обновление layout
    fig.update_layout(
        title="Price and Volatility Comparison",
        xaxis_title="Date",
        legend_title="Metrics",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
    )

    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Volatility (StdDev)", secondary_y=True)

    return fig


def create_statistics_summary(df):
    metals = df["metal_type"].unique()

    cards = []

    for metal in metals:
        metal_df = df[df["metal_type"] == metal].copy()

        try:
            current_price = metal_df["close_price"].iloc[-1]
            avg_price = metal_df["close_price"].mean()
            min_price = metal_df["close_price"].min()
            max_price = metal_df["close_price"].max()

            first_price = metal_df["close_price"].iloc[0]
            price_change = current_price - first_price
            price_change_pct = (price_change / first_price) * 100

            daily_returns = metal_df["close_price"].pct_change() * 100
            volatility = daily_returns.std()

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
        # Создание a copy of  dataframe к avoid modifications
        plot_df = df.copy()

        if "date" not in plot_df.columns and "timestamp" in plot_df.columns:
            plot_df["date"] = plot_df["timestamp"]

        metals = plot_df["metal_type"].unique()

        show_rsi = "rsi" in other_indicators

        if len(metals) > 1 and show_rsi:
            # Создание a figure with subplots для RSI
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3],
                subplot_titles=["Metals Price Comparison", "RSI"],
            )

            for metal in metals:
                metal_df = plot_df[plot_df["metal_type"] == metal].copy()

                metal_df = metal_df.dropna(subset=["close_price"])

                if not metal_df.empty:
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

                    for ma in moving_avgs:
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
                                continue

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

                    if "bb" in bollinger_bands:
                        if all(
                            col in metal_df.columns
                            for col in ["bb_upper", "bb_middle", "bb_lower"]
                        ):
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

                            # средняя полоса
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

                    # Добавление RSI
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

            rsi_data = plot_df.dropna(subset=["rsi"])
            if not rsi_data.empty:
                min_date = rsi_data["date"].min()
                max_date = rsi_data["date"].max()

                # Добавление overbought/oversold lines
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

                # Добавление 50 level line
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

            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

            fig.update_layout(
                height=800 if show_rsi else 600,
                legend=dict(
                    orientation="h", y=1.02, yanchor="bottom", xanchor="center", x=0.5
                ),
                hovermode="x unified",
                xaxis=dict(
                    title="Date",
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(title="Price"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            return fig

        if show_rsi:
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
            main_fig = make_subplots(rows=1, cols=1)

        y_min, y_max = float("inf"), float("-inf")

        for metal in metals:
            metal_df = plot_df[plot_df["metal_type"] == metal]

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

            # Добавление скользящих средних
            for ma in moving_avgs:
                if isinstance(ma, dict):
                    ma_type = ma["type"]
                    period = ma["period"]
                    column_name = f"{ma_type}_{period}"
                    ma_type_upper = ma_type.upper()
                else:
                    column_name = ma
                    try:
                        ma_type, period = ma.split("_")
                        ma_type_upper = ma_type.upper()
                    except (ValueError, AttributeError):
                        continue

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

            if "bb" in bollinger_bands:
                if all(
                    col in metal_df.columns
                    for col in ["bb_upper", "bb_middle", "bb_lower"]
                ):
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

            # Добавление RSI if selected
            if show_rsi and "rsi" in metal_df.columns:
                rsi_df = metal_df.dropna(subset=["rsi"])

                if not rsi_df.empty:
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

        if show_rsi:
            date_range = plot_df["date"]
            min_date = date_range.min()
            max_date = date_range.max()

            # Добавление overbought line (70)
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

            # Добавление oversold line (30)
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

            main_fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        # Обновление layout для  main figure
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
            return main_fig
        else:
            return main_fig

    except Exception as e:
        from loguru import logger

        logger.exception(f"Error creating advanced chart: {str(e)}")
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
    display_df = df.copy()

    # Преобразование временная метка к date для grouping
    if timeframe == "1D":
        display_df["date"] = display_df["timestamp"].dt.date

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

        # Сортировка by date (newest first) and metal type
        result_df = result_df.sort_values(
            ["date", "metal_type"], ascending=[False, True]
        )

        # Date label для table header
        date_label = "Date"
    else:
        result_df = display_df.copy()

        result_df = result_df.rename(columns={"timestamp": "date"})

        result_df = result_df.sort_values(
            ["date", "metal_type"], ascending=[False, True]
        )

        date_label = f"Timestamp ({timeframe})"

    result_df["price_change"] = result_df["close_price"] - result_df["open_price"]
    result_df["price_change_pct"] = result_df["price_change"] / result_df["open_price"]

    tables = []

    # Создание an exports section для all металлов
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

    export_buttons = []

    stores = []

    for metal in result_df["metal_type"].unique():
        metal_data = result_df[result_df["metal_type"] == metal]

        metal_records = metal_data.to_dict("records")

        store_id = f"filtered-data-{metal.lower().replace(' ', '-')}"
        stores.append(dcc.Store(id=store_id, data=metal_records))

        # Title with timeframe info
        title = f"{metal} Price Data ({timeframe})"

        columnDefs = [
            {"headerName": date_label, "field": "date"},
            {
                "headerName": "Open",
                "field": "open_price",
                "valueFormatter": {"function": "d3.format('$.2f')(params.value)"},
            },
            {
                "headerName": "High",
                "field": "high_price",
                "valueFormatter": {"function": "d3.format('$.2f')(params.value)"},
            },
            {
                "headerName": "Low",
                "field": "low_price",
                "valueFormatter": {"function": "d3.format('$.2f')(params.value)"},
            },
            {
                "headerName": "Close",
                "field": "close_price",
                "valueFormatter": {"function": "d3.format('$.2f')(params.value)"},
            },
            {
                "headerName": "Change",
                "field": "price_change",
                "valueFormatter": {"function": "d3.format('+$.2f')(params.value)"},
                "cellStyle": {
                    "styleConditions": [
                        {"condition": "params.value < 0", "style": {"color": "red"}},
                        {"condition": "params.value > 0", "style": {"color": "green"}},
                    ]
                },
            },
            {
                "headerName": "Change %",
                "field": "price_change_pct",
                "valueFormatter": {"function": "d3.format('+.2%')(params.value)"},
                "cellStyle": {
                    "styleConditions": [
                        {"condition": "params.value < 0", "style": {"color": "red"}},
                        {"condition": "params.value > 0", "style": {"color": "green"}},
                    ]
                },
            },
        ]

        table = html.Div(
            [
                html.H5(title, className="mt-3 mb-2"),
                dag.AgGrid(
                    id={"type": "filtered-data-table", "metal": metal},
                    columnDefs=columnDefs,
                    rowData=metal_records,
                    dashGridOptions={
                        "pagination": True,
                        "paginationAutoPageSize": True,
                    },
                    defaultColDef={
                        "sortable": True,
                        "filter": True,
                        "resizable": True,
                    },
                    persistence=True,
                    persistence_type="session",
                ),
            ]
        )

        tables.append(table)

        export_button = dbc.Button(
            f"Export {metal} Data as Excel",
            id={"type": "export-metal-button", "metal": metal},
            color="success",
            className="me-2 mb-2",
            n_clicks=0,
        )

        export_buttons.append(export_button)

    # Установка all  export buttons at once
    export_section.children[1].children = export_buttons

    tables.append(export_section)

    for store in stores:
        tables.append(store)

    # Добавление multiple download components для each metal
    for metal in result_df["metal_type"].unique():
        tables.append(dcc.Download(id={"type": "download-metal-excel", "metal": metal}))

    return html.Div(tables)


def create_histogram_chart(df):
    """Create a histogram showing distribution of metal prices"""
    # Prepare данных для histogram
    histogram_data = []

    for metal in df["metal_type"].unique():
        metal_df = df[df["metal_type"] == metal].copy()
        histogram_data.append(
            go.Histogram(
                x=metal_df["close_price"],
                name=metal,
                opacity=0.7,
                nbinsx=30,
            )
        )

    layout = go.Layout(
        title="Price Distribution",
        xaxis=dict(title="Price"),
        yaxis=dict(title="Frequency"),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Создание figure
    fig = go.Figure(data=histogram_data, layout=layout)
    return fig


def create_box_plot_chart(df):
    """Create a box plot showing price distribution statistics"""
    fig = go.Figure()

    for metal in df["metal_type"].unique():
        metal_df = df[df["metal_type"] == metal].copy()
        fig.add_trace(
            go.Box(
                y=metal_df["close_price"],
                name=metal,
                boxmean=True,  # Show mean as a dashed line
                boxpoints="outliers",  # Only show outliers
            )
        )

    fig.update_layout(
        title="Price Distribution by Metal",
        yaxis=dict(title="Price"),
        hovermode="y unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_candlestick_chart(df, price_series=None):
    """Create a candlestick chart from the dataframe"""
    if price_series is None:
        price_series = ["open", "high", "low", "close"]

    # Group by metal type and Создание subplots if more than one metal
    metals = df["metal_type"].unique()

    if len(metals) > 1:
        rows = len(metals)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        for i, metal in enumerate(metals):
            metal_df = df[df["metal_type"] == metal].copy()

            fig.add_trace(
                go.Candlestick(
                    x=metal_df["timestamp"],
                    open=metal_df["open_price"],
                    high=metal_df["high_price"],
                    low=metal_df["low_price"],
                    close=metal_df["close_price"],
                    name=metal,
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )

            fig.update_yaxes(
                title_text=metal,
                row=i + 1,
                col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128, 128, 128, 0.5)",
            )

            # Configure x-axis для each subplot
            if i == rows - 1:  # Only show x-axis title on the bottom subplot
                fig.update_xaxes(
                    title_text="Date",
                    row=i + 1,
                    col=1,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                )
            else:
                fig.update_xaxes(
                    row=i + 1,
                    col=1,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                )

        # Обновление layout
        fig.update_layout(
            height=250 * rows,  # Adjust height based on number of metals
            title="OHLC Candlestick Chart",
            hovermode="x unified",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        metal = metals[0]
        metal_df = df[df["metal_type"] == metal].copy()

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=metal_df["timestamp"],
                    open=metal_df["open_price"],
                    high=metal_df["high_price"],
                    low=metal_df["low_price"],
                    close=metal_df["close_price"],
                    name=metal,
                )
            ]
        )

        # Configure axes with grid
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
        )
        fig.update_yaxes(
            title_text="Price",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(128, 128, 128, 0.5)",
        )

        # Обновление layout
        fig.update_layout(
            title=f"{metal} OHLC Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

    # Customize candlestick colors & hide rangeslider
    fig.update_layout(
        xaxis_rangeslider_visible=False,
    )

    fig.update_layout(
        template="plotly_dark"  # This templates will be overridden by CSS but provides base colors
    )

    return fig


def create_comparison_chart(df):
    """Create a percentage change comparison chart from the dataframe"""
    plot_df = df.copy()

    fig = go.Figure()

    # Получение list of металлов in  dataframe
    metals = plot_df["metal_type"].unique()

    for metal in metals:
        metal_df = plot_df[plot_df["metal_type"] == metal].copy()

        # Сортировка by date
        metal_df = metal_df.sort_values("timestamp")

        first_price = metal_df["close_price"].iloc[0]
        metal_df["pct_change"] = (
            (metal_df["close_price"] - first_price) / first_price * 100
        )

        fig.add_trace(
            go.Scatter(
                x=metal_df["timestamp"],
                y=metal_df["pct_change"],
                mode="lines",
                name=metal,
            )
        )

    fig.add_shape(
        type="line",
        x0=plot_df["timestamp"].min(),
        x1=plot_df["timestamp"].max(),
        y0=0,
        y1=0,
        line=dict(color="grey", width=1, dash="dash"),
    )

    # Обновление layout
    fig.update_layout(
        title="Metal Price Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Percentage Change (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_scatter_chart(df):
    """Create a scatter chart showing metal prices over time

    This implementation shows the actual price data points over time for each metal,
    rather than the volatility vs return analysis that might be expected from a scatter plot.
    It provides a clear view of price movements with both markers for individual data points
    and connecting lines to show the trend.
    """
    plot_df = df.copy()

    # Создание empty figure
    fig = go.Figure()

    for metal in plot_df["metal_type"].unique():
        metal_df = plot_df[plot_df["metal_type"] == metal].copy()

        metal_df = metal_df.sort_values("timestamp")

        fig.add_trace(
            go.Scatter(
                x=metal_df["timestamp"],
                y=metal_df["close_price"],
                mode="markers+lines",
                marker=dict(
                    size=8,
                    opacity=0.7,
                ),
                name=metal,
            )
        )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
        title="Date",
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
        title="Price",
    )

    # Обновление layout
    fig.update_layout(
        title="Metal Prices Scatter Plot",
        hovermode="closest",
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def create_heatmap_chart(df):
    """Create a correlation heatmap for metal prices"""
    plot_df = df.copy()

    pivot_df = plot_df.pivot_table(
        index="timestamp", columns="metal_type", values="close_price"
    )

    corr_matrix = pivot_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmin=-1,  # Min value for correlation
            zmax=1,  # Max value for correlation
            colorscale="RdBu",
            colorbar=dict(title="Correlation"),
            text=np.around(corr_matrix.values, decimals=2),  # Add text values
            texttemplate="%{text:.2f}",
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Metal Price Correlation Matrix",
        xaxis=dict(title="Metal"),
        yaxis=dict(title="Metal"),
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig
