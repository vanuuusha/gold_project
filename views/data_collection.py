from dash import html, dcc, dash_table, dash, ALL
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
import threading
from loguru import logger
import json
from datetime import datetime

from database.init_database import MetalType, DataSource, CollectorSchedule, MetalPrice
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
                                                {"label": "Weekly", "value": "weekly"},
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


# Helper function to get latest price updates for each metal and source
def get_latest_updates():
    try:
        session = Session()

        # Get the latest timestamp for each metal and source combination
        subquery = (
            session.query(
                MetalPrice.metal_type,
                MetalPrice.source,
                func.max(MetalPrice.timestamp).label("max_timestamp"),
            )
            .group_by(MetalPrice.metal_type, MetalPrice.source)
            .subquery()
        )

        # Join with the original table to get the full records
        latest_prices = (
            session.query(
                MetalPrice.metal_type,
                MetalPrice.source,
                MetalPrice.timestamp,
                MetalPrice.close_price,
            )
            .join(
                subquery,
                and_(
                    MetalPrice.metal_type == subquery.c.metal_type,
                    MetalPrice.source == subquery.c.source,
                    MetalPrice.timestamp == subquery.c.max_timestamp,
                ),
            )
            .all()
        )

        # Convert to DataFrame for display
        data = []
        for metal_type, source, timestamp, price in latest_prices:
            data.append(
                {
                    "metal_type": metal_type.value,
                    "source": source.value,
                    "last_update": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "current_price": price,
                    "age_hours": round(
                        (pd.Timestamp.now() - pd.Timestamp(timestamp)).total_seconds()
                        / 3600,
                        1,
                    ),
                }
            )

        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error getting latest updates: {e}")
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

        # Добавляем стилизацию для активных расписаний
        style_data_conditional = [
            {
                "if": {
                    "filter_query": '{is_active} = "Yes"',
                },
                "backgroundColor": "#e6ffe6",  # Светло-зеленый фон для активных
                "fontWeight": "bold",
            }
        ]

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
                        "Activate" if row["is_active"] == "No" else "Deactivate",
                        id={"type": "toggle-btn", "index": row["id"]},
                        color="warning" if row["is_active"] == "No" else "info",
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

        # Добавляем кнопки для быстрого изменения интервала
        df["interval_actions"] = [
            html.Div(
                [
                    dbc.Button(
                        "Hourly",
                        id={"type": "interval-hourly", "index": row["id"]},
                        color=(
                            "primary"
                            if row["interval_type"] == "hourly"
                            else "secondary"
                        ),
                        size="sm",
                        className="me-1",
                        disabled=row["interval_type"] == "hourly",
                    ),
                    dbc.Button(
                        "Daily",
                        id={"type": "interval-daily", "index": row["id"]},
                        color=(
                            "primary"
                            if row["interval_type"] == "daily"
                            else "secondary"
                        ),
                        size="sm",
                        className="me-1",
                        disabled=row["interval_type"] == "daily",
                    ),
                    dbc.Button(
                        "Weekly",
                        id={"type": "interval-weekly", "index": row["id"]},
                        color=(
                            "primary"
                            if row["interval_type"] == "weekly"
                            else "secondary"
                        ),
                        size="sm",
                        disabled=row["interval_type"] == "weekly",
                    ),
                ],
                style={"white-space": "nowrap"},
            )
            for _, row in df.iterrows()
        ]

        # Remove ID column from display
        display_df = df.drop(columns=["id"])

        return html.Div(
            [
                html.H4("All Schedules", className="mb-3"),
                html.P(
                    "Only one schedule can be active per metal and source combination. Active schedules are highlighted in green."
                ),
                dash_table.DataTable(
                    id="schedules-table",
                    columns=[
                        {"name": col, "id": col}
                        for col in display_df.columns
                        if col not in ["interval_actions"]
                    ]
                    + [{"name": "Change Interval", "id": "interval_actions"}],
                    data=display_df.to_dict("records"),
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "10px"},
                    style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
                    style_data_conditional=style_data_conditional,
                ),
            ]
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
            metal_enum = MetalType[metal_type]
            source_enum = DataSource[data_source]

            # Check if schedule already exists for this metal and source
            existing = (
                session.query(CollectorSchedule)
                .filter(
                    and_(
                        CollectorSchedule.metal_type == metal_enum,
                        CollectorSchedule.source == source_enum,
                    )
                )
                .all()
            )

            # If this schedule will be active, deactivate all other schedules for this metal/source
            if is_active:
                for schedule in existing:
                    if schedule.is_active == 1:
                        schedule.is_active = 0
                        logger.info(
                            f"Deactivated existing schedule {schedule.id} for {metal_type} from {data_source}"
                        )
                session.commit()

            # Create new schedule
            new_schedule = CollectorSchedule(
                metal_type=metal_enum,
                source=source_enum,
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

    # Run schedule manually
    @app.callback(
        Output("trigger-refresh", "data", allow_duplicate=True),
        [Input({"type": "run-btn", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def run_schedule(n_clicks_list):
        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        schedule_id = json.loads(btn_id)["index"]

        try:
            session = Session()
            schedule = (
                session.query(CollectorSchedule)
                .filter(CollectorSchedule.id == schedule_id)
                .first()
            )

            if not schedule:
                logger.error(f"Schedule with ID {schedule_id} not found")
                return {"time": pd.Timestamp.now().isoformat(), "action": "error"}

            # Get the ticker for this metal and source
            ticker = None
            if schedule.source == DataSource.YFINANCE:
                ticker_map = {
                    MetalType.GOLD: "GC=F",
                    MetalType.SILVER: "SI=F",
                    MetalType.PLATINUM: "PL=F",
                    MetalType.PALLADIUM: "PA=F",
                    MetalType.COPPER: "HG=F",
                }
                ticker = ticker_map.get(schedule.metal_type)
            elif schedule.source == DataSource.ALPHAVANTAGE_API:
                ticker_map = {
                    MetalType.GOLD: "XAUUSD",
                    MetalType.SILVER: "XAGUSD",
                    MetalType.PLATINUM: "XPTUSD",
                    MetalType.PALLADIUM: "XPDUSD",
                    MetalType.COPPER: "COPPER",
                }
                ticker = ticker_map.get(schedule.metal_type)

            if not ticker:
                logger.error(
                    f"No ticker found for {schedule.metal_type.value} from {schedule.source.value}"
                )
                return {"time": pd.Timestamp.now().isoformat(), "action": "error"}

            # Run collection in a separate thread
            def run_collection():
                result = collector.update(
                    ticker=ticker,
                    platform_type=schedule.source.name.lower(),
                    days_back=(
                        7
                        if schedule.interval_type == "weekly"
                        else (1 if schedule.interval_type == "daily" else 1)
                    ),
                )

                if result:
                    # Update last_run timestamp
                    with Session() as update_session:
                        updated_schedule = (
                            update_session.query(CollectorSchedule)
                            .filter(CollectorSchedule.id == schedule_id)
                            .first()
                        )

                        if updated_schedule:
                            updated_schedule.last_run = datetime.now()
                            update_session.commit()
                            logger.info(f"Updated last_run for schedule {schedule_id}")

                logger.info(f"Manual run result for schedule {schedule_id}: {result}")

            # Start the collection thread
            thread = threading.Thread(target=run_collection)
            thread.daemon = True
            thread.start()

            return {"time": pd.Timestamp.now().isoformat(), "action": "run"}

        except Exception as e:
            logger.error(f"Error running schedule {schedule_id}: {e}")
            return {"time": pd.Timestamp.now().isoformat(), "action": "error"}
        finally:
            session.close()

    # Toggle schedule active state
    @app.callback(
        Output("trigger-refresh", "data", allow_duplicate=True),
        [Input({"type": "toggle-btn", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def toggle_schedule(n_clicks_list):
        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        schedule_id = json.loads(btn_id)["index"]

        try:
            session = Session()
            schedule = (
                session.query(CollectorSchedule)
                .filter(CollectorSchedule.id == schedule_id)
                .first()
            )

            if not schedule:
                logger.error(f"Schedule with ID {schedule_id} not found")
                return {"time": pd.Timestamp.now().isoformat(), "action": "error"}

            # Toggle active state
            new_active_state = 1 if schedule.is_active == 0 else 0

            # If activating, deactivate all other schedules for this metal/source
            if new_active_state == 1:
                other_schedules = (
                    session.query(CollectorSchedule)
                    .filter(
                        and_(
                            CollectorSchedule.metal_type == schedule.metal_type,
                            CollectorSchedule.source == schedule.source,
                            CollectorSchedule.id != schedule_id,
                            CollectorSchedule.is_active == 1,
                        )
                    )
                    .all()
                )

                for other in other_schedules:
                    other.is_active = 0
                    logger.info(
                        f"Deactivated schedule {other.id} for {schedule.metal_type.value} from {schedule.source.value}"
                    )

            # Update this schedule
            schedule.is_active = new_active_state
            session.commit()

            logger.info(
                f"Schedule {schedule_id} active state toggled to {new_active_state}"
            )
            return {"time": pd.Timestamp.now().isoformat(), "action": "toggle"}

        except Exception as e:
            logger.error(f"Error toggling schedule {schedule_id}: {e}")
            return {"time": pd.Timestamp.now().isoformat(), "action": "error"}
        finally:
            session.close()

    # Delete schedule
    @app.callback(
        Output("trigger-refresh", "data", allow_duplicate=True),
        [Input({"type": "delete-btn", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def delete_schedule(n_clicks_list):
        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        schedule_id = json.loads(btn_id)["index"]

        try:
            session = Session()
            schedule = (
                session.query(CollectorSchedule)
                .filter(CollectorSchedule.id == schedule_id)
                .first()
            )

            if not schedule:
                logger.error(f"Schedule with ID {schedule_id} not found")
                return {"time": pd.Timestamp.now().isoformat(), "action": "error"}

            # Delete the schedule
            session.delete(schedule)
            session.commit()

            logger.info(f"Schedule {schedule_id} deleted")
            return {"time": pd.Timestamp.now().isoformat(), "action": "delete"}

        except Exception as e:
            logger.error(f"Error deleting schedule {schedule_id}: {e}")
            return {"time": pd.Timestamp.now().isoformat(), "action": "error"}
        finally:
            session.close()

    # Collection status
    @app.callback(
        Output("collection-status", "children"),
        [Input("status-interval", "n_intervals")],
    )
    def update_collection_status(n_intervals):
        try:
            # Get the latest price updates
            latest_updates_df = get_latest_updates()

            # Get active schedules
            session = Session()
            active_schedules = (
                session.query(CollectorSchedule)
                .filter(CollectorSchedule.is_active == 1)
                .all()
            )

            active_schedules_data = []
            for schedule in active_schedules:
                # Определяем цвет и статус обновления
                hours_since_update = 0
                next_update_hours = 0
                update_status = "Unknown"

                if schedule.last_run:
                    # Сколько времени прошло с последнего обновления
                    hours_since_update = round(
                        (datetime.now() - schedule.last_run).total_seconds() / 3600, 1
                    )

                    # Когда будет следующее обновление
                    interval_minutes = schedule.interval_minutes
                    minutes_since_update = (
                        datetime.now() - schedule.last_run
                    ).total_seconds() / 60
                    next_update_minutes = max(
                        0, interval_minutes - minutes_since_update
                    )
                    next_update_hours = round(next_update_minutes / 60, 1)

                    # Статус обновления (просрочено/в процессе/ожидание)
                    if next_update_minutes <= 0:
                        update_status = "Overdue"
                    else:
                        update_status = "Waiting"

                active_schedules_data.append(
                    {
                        "metal_type": schedule.metal_type.value.capitalize(),
                        "source": schedule.source.value,
                        "interval": schedule.interval_type,
                        "last_run": (
                            schedule.last_run.strftime("%Y-%m-%d %H:%M:%S")
                            if schedule.last_run
                            else "Never"
                        ),
                        "hours_since_update": hours_since_update,
                        "next_update_in": f"{next_update_hours} hours",
                        "status": update_status,
                    }
                )

            active_schedules_df = pd.DataFrame(active_schedules_data)

            if latest_updates_df.empty and active_schedules_df.empty:
                return html.Div("No data collected yet and no active schedules.")

            # Create return components
            components = [
                html.P(
                    f"Last status check: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            ]

            if not active_schedules_df.empty:
                components.extend(
                    [
                        html.H4("Active Collection Schedules", className="mt-3"),
                        dash_table.DataTable(
                            columns=[
                                {"name": "Metal", "id": "metal_type"},
                                {"name": "Source", "id": "source"},
                                {"name": "Interval", "id": "interval"},
                                {"name": "Last Run", "id": "last_run"},
                                {
                                    "name": "Hours Since Update",
                                    "id": "hours_since_update",
                                },
                                {"name": "Next Update In", "id": "next_update_in"},
                                {"name": "Status", "id": "status"},
                            ],
                            data=active_schedules_df.to_dict("records"),
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "10px"},
                            style_header={
                                "backgroundColor": "lightgrey",
                                "fontWeight": "bold",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": '{status} = "Overdue"'},
                                    "backgroundColor": "#ffe6e6",  # Светло-красный для просроченных
                                    "color": "#cc0000",
                                },
                                {
                                    "if": {"filter_query": '{status} = "In Progress"'},
                                    "backgroundColor": "#e6f7ff",  # Светло-голубой для активных
                                    "color": "#0066cc",
                                },
                            ],
                        ),
                    ]
                )

            if not latest_updates_df.empty:
                components.extend(
                    [
                        html.H4("Latest Price Updates", className="mt-3"),
                        dash_table.DataTable(
                            columns=[
                                {"name": "Metal", "id": "metal_type"},
                                {"name": "Source", "id": "source"},
                                {"name": "Latest Price", "id": "current_price"},
                                {"name": "Last Update", "id": "last_update"},
                                {"name": "Age (hours)", "id": "age_hours"},
                            ],
                            data=latest_updates_df.to_dict("records"),
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "10px"},
                            style_header={
                                "backgroundColor": "lightgrey",
                                "fontWeight": "bold",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": "{age_hours} > 24"},
                                    "backgroundColor": "#ffe6e6",  # Светло-красный для устаревших данных
                                    "color": "#cc0000",
                                }
                            ],
                        ),
                    ]
                )

            return html.Div(components)

        except Exception as e:
            logger.error(f"Error updating collection status: {e}")
            return html.Div(f"Error updating collection status: {str(e)}")

        finally:
            if "session" in locals():
                session.close()

    # Callback for changing interval to hourly
    @app.callback(
        Output("trigger-refresh", "data", allow_duplicate=True),
        [Input({"type": "interval-hourly", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def change_interval_hourly(n_clicks_list):
        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        schedule_id = json.loads(btn_id)["index"]

        return change_schedule_interval(schedule_id, "hourly")

    # Callback for changing interval to daily
    @app.callback(
        Output("trigger-refresh", "data", allow_duplicate=True),
        [Input({"type": "interval-daily", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def change_interval_daily(n_clicks_list):
        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        schedule_id = json.loads(btn_id)["index"]

        return change_schedule_interval(schedule_id, "daily")

    # Callback for changing interval to weekly
    @app.callback(
        Output("trigger-refresh", "data", allow_duplicate=True),
        [Input({"type": "interval-weekly", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def change_interval_weekly(n_clicks_list):
        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        schedule_id = json.loads(btn_id)["index"]

        return change_schedule_interval(schedule_id, "weekly")

    # Helper function for interval changing
    def change_schedule_interval(schedule_id, interval_type):
        try:
            session = Session()
            schedule = (
                session.query(CollectorSchedule)
                .filter(CollectorSchedule.id == schedule_id)
                .first()
            )

            if not schedule:
                logger.error(f"Schedule with ID {schedule_id} not found")
                return {"time": pd.Timestamp.now().isoformat(), "action": "error"}

            # Update interval type
            old_interval = schedule.interval_type
            schedule.interval_type = interval_type
            session.commit()

            logger.info(
                f"Schedule {schedule_id} interval changed from {old_interval} to {interval_type}"
            )
            return {
                "time": pd.Timestamp.now().isoformat(),
                "action": f"interval-{interval_type}",
            }

        except Exception as e:
            logger.error(f"Error changing interval for schedule {schedule_id}: {e}")
            return {"time": pd.Timestamp.now().isoformat(), "action": "error"}
        finally:
            session.close()
