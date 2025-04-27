from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from loguru import logger


from database.init_database import MetalType, DataSource, MetalPrice
from settings import DB_URL

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

# Define layout
layout = html.Div(
    [
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
            ],
            brand="Precious Metals Analytics",
            brand_href="/dashboard/",
            color="primary",
            dark=True,
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
                        # Time Period Selection
                        dbc.Col(
                            [
                                html.Label("Time Period"),
                                dcc.Dropdown(
                                    id="time-period-dropdown",
                                    options=[
                                        {"label": "Last 24 Hours", "value": "1D"},
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
                            width=4,
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
                # Chart Type Selection
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Chart Type"),
                                dcc.Dropdown(
                                    id="chart-type-dropdown",
                                    options=[
                                        {
                                            "label": "Price Over Time",
                                            "value": "time_series",
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
                                    value="time_series",
                                ),
                            ],
                            width=4,
                        ),
                        # Additional Options
                        dbc.Col(
                            [
                                html.Label("Moving Average (Days)"),
                                dcc.Input(
                                    id="moving-avg-input",
                                    type="number",
                                    min=0,
                                    max=30,
                                    step=1,
                                    value=0,
                                ),
                            ],
                            width=4,
                        ),
                        # Update Button
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button(
                                    "Update Dashboard",
                                    id="update-dashboard-button",
                                    color="primary",
                                    className="mt-2",
                                ),
                            ],
                            width=4,
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
                                                            style={"height": "60vh"},
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
                # Download Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Export Charts"),
                                        dbc.CardBody(
                                            [
                                                dbc.Button(
                                                    "Save as JPEG",
                                                    id="save-jpeg-button",
                                                    color="success",
                                                    className="me-2",
                                                ),
                                                dbc.Button(
                                                    "Save as PDF",
                                                    id="save-pdf-button",
                                                    color="success",
                                                    className="me-2",
                                                ),
                                                html.Div(
                                                    id="export-notification",
                                                    className="mt-2",
                                                ),
                                                # Add download components with unique IDs
                                                dcc.Download(
                                                    id="dashboard-download-jpeg"
                                                ),
                                                dcc.Download(
                                                    id="dashboard-download-pdf"
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=12,
                        ),
                    ]
                ),
                # Hidden components
                dcc.Store(id="chart-data-store"),
                # Update interval for automatic refresh
                dcc.Interval(
                    id="auto-refresh-interval",
                    interval=5 * 60 * 1000,  # 5 minutes in milliseconds
                    n_intervals=0,
                ),
            ],
            className="mt-4",
        ),
    ]
)


# Helper function to get price data
def get_price_data(metals, source, start_date, end_date):
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

        # Set timestamp as index
        # df.set_index("timestamp", inplace=True)

        return df, None

    except Exception as e:
        logger.error(f"Error getting price data: {e}")
        return None, f"Error: {str(e)}"

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

    # Main callback to update all charts
    @app.callback(
        [
            Output("chart-data-store", "data"),
            Output("main-chart", "figure"),
            Output("volatility-chart", "figure"),
            Output("price-statistics", "children"),
        ],
        [
            Input("update-dashboard-button", "n_clicks"),
            Input("auto-refresh-interval", "n_intervals"),
        ],
        [
            State("metals-dropdown", "value"),
            State("source-dropdown", "value"),
            State("time-period-dropdown", "value"),
            State("dashboard-custom-date-range", "start_date"),
            State("dashboard-custom-date-range", "end_date"),
            State("chart-type-dropdown", "value"),
            State("moving-avg-input", "value"),
        ],
    )
    def update_dashboard(
        n_clicks,
        n_intervals,
        metals,
        source,
        time_period,
        custom_start_date,
        custom_end_date,
        chart_type,
        moving_avg_days,
    ):
        # Calculate date range based on time period
        end_date = datetime.now()

        if time_period == "custom":
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

        # Get data
        df, error = get_price_data(metals, source, start_date, end_date)

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
            return None, empty_fig, empty_fig, html.Div(error)

        # Log the head of the dataframe to check the data
        logger.info(f"DataFrame head before plotting:\\n{df.head()}")

        # Store data for potential export
        stored_data = df.to_json(date_format="iso", orient="split")

        # Calculate moving average if needed
        if moving_avg_days > 0:
            # Group by metal type and calculate moving average
            for metal in df["metal_type"].unique():
                mask = df["metal_type"] == metal
                df.loc[mask, "ma"] = (
                    df.loc[mask, "close_price"].rolling(window=moving_avg_days).mean()
                )

        # Create main chart based on chart type
        if chart_type == "time_series":
            main_fig = create_time_series_chart(df, moving_avg_days)
        elif chart_type == "histogram":
            main_fig = create_histogram_chart(df)
        elif chart_type == "box_plot":
            main_fig = create_box_plot_chart(df)
        elif chart_type == "candlestick":
            main_fig = create_candlestick_chart(df)
        elif chart_type == "comparison":
            main_fig = create_comparison_chart(df)
        elif chart_type == "scatter":
            main_fig = create_scatter_chart(df)
        else:
            main_fig = create_time_series_chart(df, moving_avg_days)

        # Create volatility chart
        volatility_fig = create_volatility_chart(df)

        # Create statistics summary
        stats_component = create_statistics_summary(df)

        return stored_data, main_fig, volatility_fig, stats_component

    # Add export callbacks
    @app.callback(
        Output("dashboard-download-jpeg", "data"),
        Input("save-jpeg-button", "n_clicks"),
        State("chart-data-store", "data"),
    )
    def download_jpeg(n_clicks, stored_data):
        if n_clicks is None or n_clicks == 0:
            return None

        try:
            # Return a blank image for now (this would need to be enhanced to create an actual chart image)
            return dict(content="", filename="precious_metals_chart.jpeg")
        except Exception as e:
            logger.error(f"Error exporting JPEG: {e}")
            return None

    @app.callback(
        Output("dashboard-download-pdf", "data"),
        Input("save-pdf-button", "n_clicks"),
        State("chart-data-store", "data"),
    )
    def download_pdf(n_clicks, stored_data):
        if n_clicks is None or n_clicks == 0:
            return None

        try:
            # Return a blank PDF for now (this would need to be enhanced to create an actual chart PDF)
            return dict(content="", filename="precious_metals_chart.pdf")
        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            return None


# Chart creation functions
def create_time_series_chart(df, moving_avg_days=0):
    # Make a copy of the dataframe to avoid modifying the original
    # print(df.head())
    # df["close_price"] = df["close_price"].astype(float)
    # print(df["close_price"].head())
    # plot_df = df.copy()

    # fig = px.line(
    #     plot_df,
    #     x="timestamp",
    #     y="close_price",
    #     color="metal_type",
    #     title="Price Over Time",
    #     labels={
    #         "close_price": "Price (USD)",
    #         "timestamp": "Date",
    #         "metal_type": "Metal",
    #     },
    # )
    df = pd.DataFrame(dict(x=[1, 3, 2, 4], y=[1, 2, 3, 4]))
    # fig = px.line(df, x="x", y="y", title="Unsorted Input")
    # fig.show()

    df = df.sort_values(by="x")
    fig = px.line(df, x="x", y="y", title="Sorted Input")

    # Add moving average if needed
    # if moving_avg_days > 0 and "ma" in plot_df.columns:
    #     for metal in plot_df["metal_type"].unique():
    #         metal_df = plot_df[plot_df["metal_type"] == metal]
    #         fig.add_scatter(
    #             x=metal_df["timestamp"],
    #             y=metal_df["ma"],
    #             mode="lines",
    #             line=dict(dash="dash"),
    #             name=f"{metal} {moving_avg_days}-day MA",
    #         )

    # fig.update_layout(
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    #     margin=dict(l=40, r=40, t=40, b=40),
    #     hovermode="x unified",
    # )

    # # Ensure y-axis shows actual values, not incremental ones
    # fig.update_yaxes(title="Price (USD)", type="linear", autorange=True)

    return fig


def create_histogram_chart(df):
    fig = px.histogram(
        df,
        x="close_price",
        color="metal_type",
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
    # Group by day for better visualization
    df["date"] = df["timestamp"].dt.date

    fig = px.box(
        df,
        x="metal_type",
        y="close_price",
        title="Price Distribution by Metal",
        labels={"close_price": "Price (USD)", "metal_type": "Metal"},
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def create_candlestick_chart(df):
    # Create a candlestick chart for each metal
    metals = df["metal_type"].unique()

    if len(metals) == 1:
        # For single metal, create a simple candlestick chart
        metal_df = df[df["metal_type"] == metals[0]]

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=metal_df["timestamp"],
                    open=metal_df["open_price"],
                    high=metal_df["high_price"],
                    low=metal_df["low_price"],
                    close=metal_df["close_price"],
                    name=metals[0],
                )
            ]
        )

        fig.update_layout(
            title=f"{metals[0]} Price (Candlestick)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            margin=dict(l=40, r=40, t=40, b=40),
        )
    else:
        # For multiple metals, create subplots
        fig = make_subplots(
            rows=len(metals),
            cols=1,
            subplot_titles=[f"{metal} Price" for metal in metals],
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        for i, metal in enumerate(metals, 1):
            metal_df = df[df["metal_type"] == metal]

            fig.add_trace(
                go.Candlestick(
                    x=metal_df["timestamp"],
                    open=metal_df["open_price"],
                    high=metal_df["high_price"],
                    low=metal_df["low_price"],
                    close=metal_df["close_price"],
                    name=metal,
                ),
                row=i,
                col=1,
            )

        fig.update_layout(
            title="Metal Prices (Candlestick)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=300 * len(metals),
            margin=dict(l=40, r=40, t=40, b=40),
        )

    return fig


def create_comparison_chart(df):
    # Normalize prices to compare percentage changes
    metals = df["metal_type"].unique()

    # Create a new DataFrame for normalized prices
    norm_df = pd.DataFrame()

    for metal in metals:
        metal_df = df[df["metal_type"] == metal].copy()

        # Get the first price for normalization
        first_price = metal_df["close_price"].iloc[0]

        # Calculate percentage change
        metal_df["normalized_price"] = (metal_df["close_price"] / first_price - 1) * 100

        # Add to the normalized DataFrame
        if norm_df.empty:
            norm_df = metal_df
        else:
            norm_df = pd.concat([norm_df, metal_df])

    # Create the chart
    fig = px.line(
        norm_df,
        x="timestamp",
        y="normalized_price",
        color="metal_type",
        title="Price Comparison (% Change)",
        labels={
            "normalized_price": "Change (%)",
            "timestamp": "Date",
            "metal_type": "Metal",
        },
    )

    # Add horizontal line at 0%
    fig.add_shape(
        type="line",
        line=dict(dash="dash", width=1, color="gray"),
        y0=0,
        y1=0,
        x0=0,
        x1=1,
        xref="paper",
        yref="y",
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
    )

    return fig


def create_scatter_chart(df):
    # Calculate daily returns for each metal
    daily_returns = pd.DataFrame()

    for metal in df["metal_type"].unique():
        metal_df = df[df["metal_type"] == metal].copy()
        metal_df = metal_df.sort_values("timestamp")

        # Calculate daily returns
        metal_df["daily_return"] = metal_df["close_price"].pct_change() * 100

        # Add to the daily returns DataFrame
        if daily_returns.empty:
            daily_returns = metal_df
        else:
            daily_returns = pd.concat([daily_returns, metal_df])

    # Drop NaN values (first day will have NaN return)
    daily_returns = daily_returns.dropna(subset=["daily_return"])

    # Create scatter plot
    fig = px.scatter(
        daily_returns,
        x="timestamp",
        y="daily_return",
        color="metal_type",
        title="Daily Price Volatility",
        labels={
            "daily_return": "Daily Return (%)",
            "timestamp": "Date",
            "metal_type": "Metal",
        },
    )

    # Add horizontal line at 0%
    fig.add_shape(
        type="line",
        line=dict(dash="dash", width=1, color="gray"),
        y0=0,
        y1=0,
        x0=0,
        x1=1,
        xref="paper",
        yref="y",
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
    )

    return fig


def create_volatility_chart(df):
    # Calculate daily returns for each metal
    volatility_data = []

    for metal in df["metal_type"].unique():
        metal_df = df[df["metal_type"] == metal].copy()
        metal_df = metal_df.sort_values("timestamp")

        # Calculate daily price change
        metal_df["daily_change"] = metal_df["high_price"] - metal_df["low_price"]
        metal_df["daily_change_pct"] = (
            metal_df["daily_change"] / metal_df["close_price"] * 100
        )

        # Calculate rolling volatility (7-day standard deviation of returns)
        if len(metal_df) > 7:
            metal_df["volatility"] = (
                metal_df["daily_change_pct"].rolling(window=7).std()
            )

            # Add to volatility data
            for _, row in metal_df.dropna(subset=["volatility"]).iterrows():
                volatility_data.append(
                    {
                        "metal_type": metal,
                        "timestamp": row["timestamp"],
                        "volatility": row["volatility"],
                    }
                )

    # Create DataFrame from volatility data
    volatility_df = pd.DataFrame(volatility_data)

    if volatility_df.empty:
        # Return empty figure if no data
        return {
            "data": [],
            "layout": {
                "title": "Not enough data to calculate volatility",
                "xaxis": {"title": ""},
                "yaxis": {"title": ""},
            },
        }

    # Create volatility chart
    fig = px.line(
        volatility_df,
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

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
    )

    return fig


def create_statistics_summary(df):
    # Create a statistics summary for each metal
    metals = df["metal_type"].unique()

    # Create a list of cards for each metal
    cards = []

    for metal in metals:
        metal_df = df[df["metal_type"] == metal]

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
                                        html.Td(f"${min_price:.2f} / ${max_price:.2f}"),
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

    # Arrange cards in rows
    rows = []
    for i in range(0, len(cards), 2):
        row_cards = cards[i : i + 2]
        row = dbc.Row([dbc.Col(card, width=6) for card in row_cards])
        rows.append(row)

    return html.Div(rows)
