from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
import threading
from loguru import logger

from database.init_database import MetalType, DataSource, CollectorSchedule
from database.data_collector import DataCollector
from settings import DB_URL

# Initialize collector
collector = DataCollector()

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

# Define data collection layout
layout = html.Div(
    [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard/")),
                dbc.NavItem(
                    dbc.NavLink(
                        "Data Collection",
                        href="/dashboard/data-collection",
                        active=True,
                    )
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
                html.H1("Data Collection Settings", className="my-4"),
                html.P(
                    "Configure data collection schedules and trigger manual updates."
                ),
                html.Hr(),
                # Collection Schedules Section
                html.H3("Collection Schedules", className="mt-4"),
                html.Div(id="schedules-table-container"),
                # Add New Schedule Form
                html.H3("Add New Schedule", className="mt-4"),
                dbc.Form(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Metal Type"),
                                        dcc.Dropdown(
                                            id="metal-type-dropdown",
                                            options=[
                                                {
                                                    "label": metal_type.value,
                                                    "value": metal_type.name,
                                                }
                                                for metal_type in MetalType
                                            ],
                                            value=MetalType.GOLD.name,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Data Source"),
                                        dcc.Dropdown(
                                            id="data-source-dropdown",
                                            options=[
                                                {
                                                    "label": data_source.value,
                                                    "value": data_source.name,
                                                }
                                                for data_source in DataSource
                                            ],
                                            value=DataSource.YFINANCE.name,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Interval Type"),
                                        dcc.Dropdown(
                                            id="interval-type-dropdown",
                                            options=[
                                                {"label": "Hourly", "value": "hourly"},
                                                {"label": "Daily", "value": "daily"},
                                            ],
                                            value="hourly",
                                        ),
                                    ],
                                    width=4,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Checkbox(
                                            id="is-active-checkbox",
                                            label="Active",
                                            value=True,
                                            className="mt-3",
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Add Schedule",
                                            id="add-schedule-button",
                                            color="primary",
                                            className="mt-3",
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col([html.Div(id="add-schedule-message")], width=4),
                            ]
                        ),
                    ],
                    className="mb-4",
                ),
                # Manual Data Collection Section
                html.H3("Manual Data Collection", className="mt-4"),
                dbc.Form(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Metal Type"),
                                        dcc.Dropdown(
                                            id="manual-metal-type-dropdown",
                                            options=[
                                                {
                                                    "label": metal_type.value,
                                                    "value": metal_type.name,
                                                }
                                                for metal_type in MetalType
                                            ],
                                            value=MetalType.GOLD.name,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Data Source"),
                                        dcc.Dropdown(
                                            id="manual-data-source-dropdown",
                                            options=[
                                                {
                                                    "label": data_source.value,
                                                    "value": data_source.name,
                                                }
                                                for data_source in DataSource
                                            ],
                                            value=DataSource.YFINANCE.name,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Days Back"),
                                        dcc.Input(
                                            id="days-back-input",
                                            type="number",
                                            value=30,
                                            min=1,
                                            max=365,
                                            step=1,
                                        ),
                                    ],
                                    width=4,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Collect Data Now",
                                            id="collect-data-button",
                                            color="primary",
                                            className="mt-3",
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col([html.Div(id="collect-data-message")], width=8),
                            ]
                        ),
                    ],
                    className="mb-4",
                ),
                # Collection Status
                html.H3("Collection Status", className="mt-4"),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(id="collection-status"),
                                dcc.Interval(
                                    id="status-interval",
                                    interval=5000,  # 5 seconds
                                    n_intervals=0,
                                ),
                            ]
                        )
                    ]
                ),
                # Shared components
                dcc.Store(id="trigger-refresh"),
            ],
            className="mt-4",
        ),
    ]
)


# Helper function to get current schedules from the database
def get_schedules():
    try:
        session = Session()
        schedules = session.query(CollectorSchedule).all()

        # Convert to DataFrame for table display
        data = []
        for schedule in schedules:
            data.append(
                {
                    "id": schedule.id,
                    "metal_type": schedule.metal_type.value,
                    "source": schedule.source.value,
                    "interval_type": schedule.interval_type,
                    "is_active": "Yes" if schedule.is_active == 1 else "No",
                    "last_run": (
                        schedule.last_run.strftime("%Y-%m-%d %H:%M:%S")
                        if schedule.last_run
                        else "Never"
                    ),
                    "created_at": schedule.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error getting schedules: {e}")
        return pd.DataFrame()
    finally:
        session.close()


# Register callbacks for data collection page
def register_callbacks(app):
    # Display current schedules
    @app.callback(
        Output("schedules-table-container", "children"),
        [Input("trigger-refresh", "data"), Input("status-interval", "n_intervals")],
    )
    def update_schedules_table(trigger_data, n_intervals):
        df = get_schedules()

        if df.empty:
            return html.Div("No schedules found. Add a new schedule below.")

        # Add action buttons to each row
        df["actions"] = [
            html.Div(
                [
                    dbc.Button(
                        "Run",
                        id={"type": "run-btn", "index": row["id"]},
                        color="success",
                        size="sm",
                        className="me-1",
                    ),
                    dbc.Button(
                        "Edit",
                        id={"type": "edit-btn", "index": row["id"]},
                        color="warning",
                        size="sm",
                        className="me-1",
                    ),
                    dbc.Button(
                        "Delete",
                        id={"type": "delete-btn", "index": row["id"]},
                        color="danger",
                        size="sm",
                    ),
                ],
                style={"white-space": "nowrap"},
            )
            for _, row in df.iterrows()
        ]

        # Remove ID column from display
        display_df = df.drop(columns=["id"])

        return dash_table.DataTable(
            id="schedules-table",
            columns=[{"name": col, "id": col} for col in display_df.columns],
            data=display_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
        )

    # Add new schedule
    @app.callback(
        [Output("add-schedule-message", "children"), Output("trigger-refresh", "data")],
        [Input("add-schedule-button", "n_clicks")],
        [
            State("metal-type-dropdown", "value"),
            State("data-source-dropdown", "value"),
            State("interval-type-dropdown", "value"),
            State("is-active-checkbox", "value"),
        ],
    )
    def add_schedule(n_clicks, metal_type, data_source, interval_type, is_active):
        if n_clicks is None:
            return "", None

        try:
            session = Session()

            # Check if schedule already exists
            existing = (
                session.query(CollectorSchedule)
                .filter(
                    and_(
                        CollectorSchedule.metal_type == MetalType[metal_type],
                        CollectorSchedule.source == DataSource[data_source],
                    )
                )
                .first()
            )

            if existing:
                return dbc.Alert(
                    f"Schedule for {metal_type} from {data_source} already exists.",
                    color="warning",
                ), {"time": pd.Timestamp.now().isoformat()}

            # Create new schedule
            new_schedule = CollectorSchedule(
                metal_type=MetalType[metal_type],
                source=DataSource[data_source],
                interval_type=interval_type,
                is_active=1 if is_active else 0,
            )

            session.add(new_schedule)
            session.commit()

            return dbc.Alert(
                f"Schedule added successfully for {metal_type} from {data_source}",
                color="success",
            ), {"time": pd.Timestamp.now().isoformat()}

        except Exception as e:
            logger.error(f"Error adding schedule: {e}")
            return dbc.Alert(f"Error adding schedule: {str(e)}", color="danger"), None

        finally:
            session.close()

    # Collect data manually
    @app.callback(
        Output("collect-data-message", "children"),
        [Input("collect-data-button", "n_clicks")],
        [
            State("manual-metal-type-dropdown", "value"),
            State("manual-data-source-dropdown", "value"),
            State("days-back-input", "value"),
        ],
    )
    def collect_data_manually(n_clicks, metal_type, data_source, days_back):
        if n_clicks is None:
            return ""

        try:
            # Map dropdown value to ticker
            ticker = None
            if data_source == "YFINANCE":
                ticker_map = {
                    "GOLD": "GC=F",
                    "SILVER": "SI=F",
                    "PLATINUM": "PL=F",
                    "PALLADIUM": "PA=F",
                    "COPPER": "HG=F",
                }
                ticker = ticker_map.get(metal_type)
            elif data_source == "ALPHAVANTAGE_API":
                ticker_map = {
                    "GOLD": "XAUUSD",
                    "SILVER": "XAGUSD",
                    "PLATINUM": "XPTUSD",
                    "PALLADIUM": "XPDUSD",
                    "COPPER": "COPPER",
                }
                ticker = ticker_map.get(metal_type)

            if not ticker:
                return dbc.Alert(
                    f"Unsupported combination: {metal_type} from {data_source}",
                    color="warning",
                )

            # Run collection in a separate thread
            def run_collection():
                result = collector.update(
                    ticker=ticker,
                    platform_type=data_source.lower(),
                    days_back=days_back,
                )
                logger.info(f"Manual collection result: {result}")

            # Start the collection thread
            thread = threading.Thread(target=run_collection)
            thread.daemon = True
            thread.start()

            return dbc.Alert(
                f"Data collection started for {metal_type} from {data_source} for the last {days_back} days",
                color="info",
            )

        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            return dbc.Alert(f"Error collecting data: {str(e)}", color="danger")

    # Collection status
    @app.callback(
        Output("collection-status", "children"),
        [Input("status-interval", "n_intervals")],
    )
    def update_collection_status(n_intervals):
        try:
            session = Session()

            # Get latest metal prices
            latest_prices = []
            for metal_type in MetalType:
                for source in DataSource:
                    try:
                        # Skip checking some combinations that might not exist
                        if (
                            source == DataSource.ALPHAVANTAGE_API
                            and metal_type == MetalType.COPPER
                        ):
                            continue

                        # Get the latest price for this metal and source
                        latest_price = (
                            session.query(MetalPrice)
                            .filter(
                                MetalPrice.metal_type == metal_type,
                                MetalPrice.source == source,
                            )
                            .order_by(MetalPrice.timestamp.desc())
                            .first()
                        )

                        if latest_price:
                            latest_prices.append(
                                {
                                    "metal": metal_type.value,
                                    "source": source.value,
                                    "price": latest_price.close_price,
                                    "timestamp": latest_price.timestamp.strftime(
                                        "%Y-%m-%d %H:%M"
                                    ),
                                    "age_hours": round(
                                        (
                                            pd.Timestamp.now()
                                            - pd.Timestamp(latest_price.timestamp)
                                        ).total_seconds()
                                        / 3600,
                                        1,
                                    ),
                                }
                            )
                    except Exception as inner_e:
                        logger.error(
                            f"Error getting latest price for {metal_type.value} from {source.value}: {inner_e}"
                        )

            # Create a table with the latest prices
            df = pd.DataFrame(latest_prices)

            if df.empty:
                return html.Div("No data collected yet.")

            # Format the table
            return html.Div(
                [
                    html.P(
                        f"Last update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Metal", "id": "metal"},
                            {"name": "Source", "id": "source"},
                            {"name": "Latest Price", "id": "price"},
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Age (hours)", "id": "age_hours"},
                        ],
                        data=df.to_dict("records"),
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={
                            "backgroundColor": "lightgrey",
                            "fontWeight": "bold",
                        },
                    ),
                ]
            )

        except Exception as e:
            logger.error(f"Error updating collection status: {e}")
            return html.Div(f"Error updating collection status: {str(e)}")

        finally:
            session.close()
