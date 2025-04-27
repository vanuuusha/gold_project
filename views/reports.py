from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
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
                dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard/")),
                dbc.NavItem(
                    dbc.NavLink("Data Collection", href="/dashboard/data-collection")
                ),
                dbc.NavItem(
                    dbc.NavLink("Database Management", href="/dashboard/database")
                ),
                dbc.NavItem(
                    dbc.NavLink("Reports", href="/dashboard/reports", active=True)
                ),
            ],
            brand="Precious Metals Analytics",
            brand_href="/dashboard/",
            color="primary",
            dark=True,
        ),
        dbc.Container(
            [
                html.H1("Reports", className="my-4"),
                html.P("Generate text and tabular reports from collected data."),
                html.Hr(),
                # Report Type Selection
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Report Type"),
                                dcc.Dropdown(
                                    id="report-type-dropdown",
                                    options=[
                                        {
                                            "label": "Price Summary",
                                            "value": "price_summary",
                                        },
                                        {
                                            "label": "Price Comparison",
                                            "value": "price_comparison",
                                        },
                                        {
                                            "label": "Daily Price Change",
                                            "value": "daily_change",
                                        },
                                        {
                                            "label": "Monthly Averages",
                                            "value": "monthly_avg",
                                        },
                                        {
                                            "label": "Volatility Analysis",
                                            "value": "volatility",
                                        },
                                    ],
                                    value="price_summary",
                                ),
                            ],
                            width=6,
                        ),
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
                    ],
                    className="mb-3",
                ),
                # Custom Date Range (initially hidden)
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="reports-custom-date-range-container",
                                    children=[
                                        html.Label("Custom Date Range"),
                                        dcc.DatePickerRange(
                                            id="reports-custom-date-range",
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
                # Metal and Source Selection
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
                                    value=[MetalType.GOLD.name, MetalType.SILVER.name],
                                    multi=True,
                                ),
                            ],
                            width=6,
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
                            width=6,
                        ),
                    ],
                    className="mb-3",
                ),
                # Generate Report Button
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Generate Report",
                                    id="generate-report-button",
                                    color="primary",
                                    className="mt-2 mb-4",
                                ),
                            ],
                            width={"size": 6, "offset": 3},
                        ),
                    ]
                ),
                # Report Container
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Report"),
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    id="loading-report",
                                                    type="circle",
                                                    children=[
                                                        html.Div(id="report-container"),
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
                # Export Options
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Export Options"),
                                        dbc.CardBody(
                                            [
                                                dbc.Button(
                                                    "Export as CSV",
                                                    id="export-csv-button",
                                                    color="success",
                                                    className="me-2",
                                                ),
                                                dbc.Button(
                                                    "Export as Excel",
                                                    id="export-excel-button",
                                                    color="success",
                                                    className="me-2",
                                                ),
                                                html.Div(
                                                    id="export-notification",
                                                    className="mt-2",
                                                ),
                                                dcc.Download(id="reports-download-csv"),
                                                dcc.Download(
                                                    id="reports-download-excel"
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
                # Hidden storage
                dcc.Store(id="report-data-store"),
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
        Output("reports-custom-date-range-container", "style"),
        [Input("time-period-dropdown", "value")],
    )
    def toggle_custom_date_range(time_period):
        if time_period == "custom":
            return {"display": "block"}
        return {"display": "none"}

    # Update end date of custom range when time period changes
    @app.callback(
        [
            Output("reports-custom-date-range", "start_date"),
            Output("reports-custom-date-range", "end_date"),
        ],
        [Input("time-period-dropdown", "value")],
    )
    def update_custom_date_range(time_period):
        end_date = datetime.now().date()

        if time_period == "7D":
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
            import dash

            return dash.no_update, dash.no_update

        return start_date, end_date

    # Generate report
    @app.callback(
        [Output("report-container", "children"), Output("report-data-store", "data")],
        [Input("generate-report-button", "n_clicks")],
        [
            State("report-type-dropdown", "value"),
            State("metals-dropdown", "value"),
            State("source-dropdown", "value"),
            State("time-period-dropdown", "value"),
            State("reports-custom-date-range", "start_date"),
            State("reports-custom-date-range", "end_date"),
        ],
    )
    def generate_report(
        n_clicks,
        report_type,
        metals,
        source,
        time_period,
        custom_start_date,
        custom_end_date,
    ):
        if n_clicks is None:
            return html.Div("Click 'Generate Report' to create a report."), None

        # Calculate date range based on time period
        end_date = datetime.now()

        if time_period == "custom":
            start_date = datetime.strptime(custom_start_date, "%Y-%m-%d")
            end_date = datetime.strptime(custom_end_date, "%Y-%m-%d")
        else:
            days_map = {"7D": 7, "30D": 30, "90D": 90, "180D": 180, "365D": 365}
            days = days_map.get(time_period, 30)
            start_date = end_date - timedelta(days=days)

        # Get data
        df, error = get_price_data(metals, source, start_date, end_date)

        if error:
            return html.Div(dbc.Alert(error, color="warning")), None

        # Store data for export
        stored_data = df.to_json(date_format="iso", orient="split")

        # Generate report based on type
        if report_type == "price_summary":
            report_content = generate_price_summary_report(df)
        elif report_type == "price_comparison":
            report_content = generate_price_comparison_report(df)
        elif report_type == "daily_change":
            report_content = generate_daily_change_report(df)
        elif report_type == "monthly_avg":
            report_content = generate_monthly_avg_report(df)
        elif report_type == "volatility":
            report_content = generate_volatility_report(df)
        else:
            report_content = generate_price_summary_report(df)

        return report_content, stored_data

    # Export to CSV
    @app.callback(
        Output("reports-download-csv", "data"),
        [Input("export-csv-button", "n_clicks")],
        [State("report-data-store", "data"), State("report-type-dropdown", "value")],
    )
    def export_csv(n_clicks, stored_data, report_type):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(stored_data, orient="split")

            # Process the dataframe based on report type
            if report_type == "price_summary":
                export_df = create_price_summary_df(df)
            elif report_type == "price_comparison":
                export_df = create_price_comparison_df(df)
            elif report_type == "daily_change":
                export_df = create_daily_change_df(df)
            elif report_type == "monthly_avg":
                export_df = create_monthly_avg_df(df)
            elif report_type == "volatility":
                export_df = create_volatility_df(df)
            else:
                export_df = df

            return dcc.send_data_frame(
                export_df.to_csv, f"{report_type}_report.csv", index=False
            )
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return None

    # Export to Excel
    @app.callback(
        Output("reports-download-excel", "data"),
        [Input("export-excel-button", "n_clicks")],
        [State("report-data-store", "data"), State("report-type-dropdown", "value")],
    )
    def export_excel(n_clicks, stored_data, report_type):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(stored_data, orient="split")

            # Process the dataframe based on report type
            if report_type == "price_summary":
                export_df = create_price_summary_df(df)
            elif report_type == "price_comparison":
                export_df = create_price_comparison_df(df)
            elif report_type == "daily_change":
                export_df = create_daily_change_df(df)
            elif report_type == "monthly_avg":
                export_df = create_monthly_avg_df(df)
            elif report_type == "volatility":
                export_df = create_volatility_df(df)
            else:
                export_df = df

            return dcc.send_data_frame(
                export_df.to_excel, f"{report_type}_report.xlsx", index=False
            )
        except Exception as e:
            logger.error(f"Error exporting Excel: {e}")
            return None


# Report generation functions
def generate_price_summary_report(df):
    # Create summary statistics for each metal
    report_df = create_price_summary_df(df)

    # Create report content
    report_content = [
        html.H3("Price Summary Report"),
        html.P(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        html.Hr(),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in report_df.columns],
            data=report_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
        ),
        html.Hr(),
        html.P(
            "This report provides a summary of prices for the selected metals during the specified time period."
        ),
    ]

    return html.Div(report_content)


def create_price_summary_df(df):
    # Group by metal type
    metals = df["metal_type"].unique()

    # Create summary data
    summary_data = []

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

        # Add to summary data
        summary_data.append(
            {
                "Metal": metal,
                "Latest Price (USD)": round(current_price, 2),
                "Average Price (USD)": round(avg_price, 2),
                "Min Price (USD)": round(min_price, 2),
                "Max Price (USD)": round(max_price, 2),
                "Price Change (USD)": round(price_change, 2),
                "Price Change (%)": round(price_change_pct, 2),
                "Volatility (%)": round(volatility, 2),
                "Data Points": len(metal_df),
                "Start Date": metal_df["timestamp"].min().strftime("%Y-%m-%d"),
                "End Date": metal_df["timestamp"].max().strftime("%Y-%m-%d"),
            }
        )

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    return summary_df


def generate_price_comparison_report(df):
    # Create comparison data
    report_df = create_price_comparison_df(df)

    # Create report content
    report_content = [
        html.H3("Price Comparison Report"),
        html.P(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        html.Hr(),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in report_df.columns],
            data=report_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
            sort_action="native",
        ),
        html.Hr(),
        html.P(
            "This report compares the performance of different metals during the specified time period."
        ),
    ]

    return html.Div(report_content)


def create_price_comparison_df(df):
    # Group by date and metal type, keeping only the close price
    df["date"] = df["timestamp"].dt.date
    pivoted = df.pivot_table(
        index="date", columns="metal_type", values="close_price", aggfunc="last"
    ).reset_index()

    # Calculate daily returns for each metal
    metals = df["metal_type"].unique()
    for metal in metals:
        if metal in pivoted.columns:
            pivoted[f"{metal} Daily Return (%)"] = pivoted[metal].pct_change() * 100

    # Format date column
    pivoted["date"] = pivoted["date"].astype(str)

    # Ensure all metal columns come before all return columns
    cols = ["date"] + list(metals) + [f"{metal} Daily Return (%)" for metal in metals]
    pivoted = pivoted[cols]

    return pivoted


def generate_daily_change_report(df):
    # Create daily change data
    report_df = create_daily_change_df(df)

    # Create report content
    report_content = [
        html.H3("Daily Price Change Report"),
        html.P(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        html.Hr(),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in report_df.columns],
            data=report_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
            sort_action="native",
            sort_by=[{"column_id": "Date", "direction": "desc"}],
        ),
        html.Hr(),
        html.P("This report shows the daily price changes for the selected metals."),
    ]

    return html.Div(report_content)


def create_daily_change_df(df):
    # Group by date and metal type
    df["Date"] = df["timestamp"].dt.date

    # Create daily change data
    daily_data = []

    for metal in df["metal_type"].unique():
        metal_df = df[df["metal_type"] == metal].copy()

        # Group by date, keeping first and last values
        daily_df = (
            metal_df.groupby("Date")
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

        # Calculate daily changes
        for _, row in daily_df.iterrows():
            daily_change = row["close_price"] - row["open_price"]
            daily_change_pct = (daily_change / row["open_price"]) * 100
            range_pct = (
                (row["high_price"] - row["low_price"]) / row["open_price"]
            ) * 100

            daily_data.append(
                {
                    "Date": row["Date"],
                    "Metal": metal,
                    "Open (USD)": round(row["open_price"], 2),
                    "High (USD)": round(row["high_price"], 2),
                    "Low (USD)": round(row["low_price"], 2),
                    "Close (USD)": round(row["close_price"], 2),
                    "Change (USD)": round(daily_change, 2),
                    "Change (%)": round(daily_change_pct, 2),
                    "Range (%)": round(range_pct, 2),
                }
            )

    # Create DataFrame and sort by date (newest first)
    daily_df = pd.DataFrame(daily_data)
    daily_df = daily_df.sort_values(["Date", "Metal"], ascending=[False, True])

    # Convert date to string for display
    daily_df["Date"] = daily_df["Date"].astype(str)

    return daily_df


def generate_monthly_avg_report(df):
    # Create monthly average data
    report_df = create_monthly_avg_df(df)

    # Create report content
    report_content = [
        html.H3("Monthly Average Prices Report"),
        html.P(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        html.Hr(),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in report_df.columns],
            data=report_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
            sort_action="native",
        ),
        html.Hr(),
        html.P("This report shows the monthly average prices for the selected metals."),
    ]

    return html.Div(report_content)


def create_monthly_avg_df(df):
    # Extract year and month
    df["Year"] = df["timestamp"].dt.year
    df["Month"] = df["timestamp"].dt.month

    # Create monthly averages
    monthly_data = []

    for metal in df["metal_type"].unique():
        metal_df = df[df["metal_type"] == metal]

        # Group by year and month
        monthly_avg = (
            metal_df.groupby(["Year", "Month"])
            .agg(
                {
                    "close_price": "mean",
                    "high_price": "max",
                    "low_price": "min",
                    "timestamp": "count",
                }
            )
            .reset_index()
        )

        # Calculate monthly statistics
        for _, row in monthly_avg.iterrows():
            # Create month name
            month_name = datetime(int(row["Year"]), int(row["Month"]), 1).strftime(
                "%B %Y"
            )

            monthly_data.append(
                {
                    "Month": month_name,
                    "Metal": metal,
                    "Average Price (USD)": round(row["close_price"], 2),
                    "Highest Price (USD)": round(row["high_price"], 2),
                    "Lowest Price (USD)": round(row["low_price"], 2),
                    "Price Range (USD)": round(row["high_price"] - row["low_price"], 2),
                    "Data Points": row["timestamp"],
                }
            )

    # Create DataFrame and sort by month
    monthly_df = pd.DataFrame(monthly_data)

    # Add a hidden column for sorting
    monthly_df["SortKey"] = monthly_df["Month"].apply(
        lambda x: datetime.strptime(x, "%B %Y")
    )
    monthly_df = monthly_df.sort_values(["SortKey", "Metal"], ascending=[False, True])

    # Drop the sort key
    monthly_df = monthly_df.drop(columns=["SortKey"])

    return monthly_df


def generate_volatility_report(df):
    # Create volatility data
    report_df = create_volatility_df(df)

    # Create report content
    report_content = [
        html.H3("Volatility Analysis Report"),
        html.P(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        html.Hr(),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in report_df.columns],
            data=report_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
            sort_action="native",
        ),
        html.Hr(),
        html.P(
            "This report analyzes the volatility of the selected metals over different time periods."
        ),
    ]

    return html.Div(report_content)


def create_volatility_df(df):
    # Calculate volatility for different time periods
    volatility_data = []

    for metal in df["metal_type"].unique():
        metal_df = df[df["metal_type"] == metal].copy()
        metal_df = metal_df.sort_values("timestamp")

        # Calculate daily returns
        metal_df["daily_return"] = metal_df["close_price"].pct_change() * 100

        # Calculate volatility over different periods
        if len(metal_df) >= 7:  # At least a week of data
            week_vol = metal_df["daily_return"].tail(7).std()
        else:
            week_vol = None

        if len(metal_df) >= 30:  # At least a month of data
            month_vol = metal_df["daily_return"].tail(30).std()
        else:
            month_vol = None

        if len(metal_df) >= 90:  # At least 3 months of data
            quarter_vol = metal_df["daily_return"].tail(90).std()
        else:
            quarter_vol = None

        # Calculate overall volatility
        overall_vol = metal_df["daily_return"].std()

        # Calculate price range as percentage of average price
        price_range_pct = (
            (metal_df["high_price"].max() - metal_df["low_price"].min())
            / metal_df["close_price"].mean()
        ) * 100

        # Calculate largest single-day move
        max_daily_move = metal_df["daily_return"].abs().max()

        volatility_data.append(
            {
                "Metal": metal,
                "7-Day Volatility (%)": (
                    round(week_vol, 2) if week_vol is not None else "N/A"
                ),
                "30-Day Volatility (%)": (
                    round(month_vol, 2) if month_vol is not None else "N/A"
                ),
                "90-Day Volatility (%)": (
                    round(quarter_vol, 2) if quarter_vol is not None else "N/A"
                ),
                "Overall Volatility (%)": round(overall_vol, 2),
                "Price Range (%)": round(price_range_pct, 2),
                "Largest Daily Move (%)": round(max_daily_move, 2),
                "Data Points": len(metal_df),
                "Start Date": metal_df["timestamp"].min().strftime("%Y-%m-%d"),
                "End Date": metal_df["timestamp"].max().strftime("%Y-%m-%d"),
            }
        )

    # Create DataFrame
    volatility_df = pd.DataFrame(volatility_data)

    return volatility_df
