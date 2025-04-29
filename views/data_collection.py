from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from loguru import logger
import json

from settings import DB_URL
from database.init_database import CollectorSchedule, DataSource, MetalType
from database.data_collector import DataCollector
from views.components import create_theme_switch

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

collector = DataCollector()

# Define  layout для  данных collection page
layout = html.Div(
    [
        dcc.Store(id="theme-store", storage_type="local"),
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
            id="navbar-data-collection",
        ),
        dbc.Container(
            [
                html.H1("Scheduler Management", className="my-4"),
                html.P("View and manage collection schedules for metal prices."),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Refresh",
                                    id="refresh-schedules-button",
                                    color="primary",
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "Run All Updates",
                                    id="run-updates-button",
                                    color="success",
                                    className="me-2",
                                ),
                                html.Div(id="update-status-message", className="mt-2"),
                            ],
                            width=12,
                            className="mb-3",
                        ),
                    ],
                ),
                html.Div(id="scheduler-cards-container", className="mb-4"),
                dcc.Interval(
                    id="refresh-interval",
                    interval=60 * 1000,  # 1 minute
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Store(id="scheduler-data-store"),
            ],
            fluid=True,
            id="data-collection-container",
        ),
    ],
    id="data-collection-page",
)


# Helper functions
def get_scheduler_data():
    """Получение данных планировщика и форматирование для хранилища"""
    try:
        session = Session()
        schedules = session.query(CollectorSchedule).all()

        current_time = datetime.now()

        # Format данных для  Хранилище
        data = []
        for schedule in schedules:
            # Расчет next run time На основе updated_at and interval (для newly activated items)
            next_run = None
            seconds_until_next_run = "N/A"

            reference_time = (
                schedule.last_run if schedule.last_run else schedule.updated_at
            )

            if reference_time and schedule.is_active == 1:
                if schedule.interval_type == "hourly":
                    next_run = reference_time + timedelta(hours=1)
                elif schedule.interval_type == "daily":
                    next_run = reference_time + timedelta(days=1)
                elif schedule.interval_type == "weekly":
                    next_run = reference_time + timedelta(weeks=1)

                if next_run and next_run > current_time:
                    time_diff = next_run - current_time
                    seconds_until_next_run = f"{int(time_diff.total_seconds())} сек."
                elif next_run:
                    seconds_until_next_run = "0 сек. (просрочено)"
            elif schedule.is_active == 0:
                seconds_until_next_run = "Неактивно"

            # Format dates and times для display
            last_run_display = (
                schedule.last_run.strftime("%Y-%m-%d %H:%M:%S")
                if schedule.last_run
                else "Never"
            )

            last_triggered_display = (
                schedule.last_triggered.strftime("%Y-%m-%d %H:%M:%S")
                if schedule.last_triggered
                else "Never"
            )

            created_at_display = (
                schedule.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if schedule.created_at
                else "N/A"
            )
            updated_at_display = (
                schedule.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                if schedule.updated_at
                else "N/A"
            )

            data.append(
                {
                    "id": schedule.id,
                    "metal_type": schedule.metal_type.name,
                    "source": schedule.source.name,
                    "interval_type": schedule.interval_type,
                    "is_active": schedule.is_active,
                    "last_run": last_run_display,
                    "last_triggered": last_triggered_display,
                    "next_run": seconds_until_next_run,
                    "created_at": created_at_display,
                    "updated_at": updated_at_display,
                }
            )

        return data
    except Exception as e:
        logger.error(f"Error retrieving scheduler data: {e}")
        return []
    finally:
        session.close()


def update_scheduler_record(record_id, field, value):
    """Обновление конкретного поля в записи планировщика"""
    try:
        session = Session()
        schedule = (
            session.query(CollectorSchedule)
            .filter(CollectorSchedule.id == record_id)
            .first()
        )

        if not schedule:
            logger.error(f"Schedule with ID {record_id} not found")
            return False, f"Schedule with ID {record_id} not found"

        setattr(schedule, field, value)
        schedule.updated_at = datetime.now()  # Update the updated_at timestamp

        session.commit()
        logger.info(f"Updated schedule {record_id}: {field} = {value}")
        return True, f"Updated schedule {record_id}"
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating scheduler record: {e}")
        return False, f"Error: {str(e)}"
    finally:
        session.close()


def create_scheduler_cards(data):
    """Create card components for each scheduler item"""
    if not data:
        return html.Div("No scheduler data available.", className="text-center mt-4")

    cards = []

    for item in data:
        card = dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.H5(
                            f"{item['metal_type']} from {item['source']}",
                            className="mb-0",
                        ),
                        html.Small(f"ID: {item['id']}", className="text-muted"),
                    ]
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P("Update Frequency:"),
                                        dcc.Dropdown(
                                            id={
                                                "type": "interval-type",
                                                "index": item["id"],
                                            },
                                            options=[
                                                {"label": "Hourly", "value": "hourly"},
                                                {"label": "Monthly", "value": "daily"},
                                                {"label": "Weekly", "value": "weekly"},
                                            ],
                                            value=item["interval_type"],
                                            clearable=False,
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.P("Status:"),
                                        dcc.Dropdown(
                                            id={
                                                "type": "is-active",
                                                "index": item["id"],
                                            },
                                            options=[
                                                {"label": "Active", "value": 1},
                                                {"label": "UnActive", "value": 0},
                                            ],
                                            value=item["is_active"],
                                            clearable=False,
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
                                        html.P("Last Update:"),
                                        html.P(item["last_run"], className="text-info"),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.P("Next Update:"),
                                        html.P(
                                            item["next_run"], className="text-primary"
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
                                        html.P("Last Triggered:"),
                                        html.P(
                                            item["last_triggered"],
                                            className="text-warning",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.P("Last Time Updated:"),
                                        html.P(
                                            item["updated_at"],
                                            className="text-muted small",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            "Save Changes",
                            id={"type": "save-button", "index": item["id"]},
                            color="primary",
                            className="mt-2",
                            n_clicks=0,
                        ),
                        html.Div(
                            id={"type": "save-status", "index": item["id"]},
                            className="mt-2",
                        ),
                    ]
                ),
            ],
            className="mb-3",
            style={"border": "1px solid #dee2e6"},
        )
        cards.append(card)

    return html.Div(cards)


def register_callbacks(app):
    @app.callback(
        [
            Output("data-collection-container", "className"),
            Output("navbar-data-collection", "dark"),
            Output("navbar-data-collection", "color"),
            Output("data-collection-page", "className"),
        ],
        [Input("theme-store", "data")],
        prevent_initial_call=False,
    )
    def update_data_collection_theme(theme_data):
        dark_mode = theme_data.get("dark_mode", True) if theme_data else True

        if dark_mode:
            return (
                "dark-mode",
                True,
                "dark",
                "main-container dark-mode",
            )
        else:
            return (
                "",
                False,
                "light",
                "main-container light-mode",
            )

    @app.callback(
        Output("theme-switch", "value", allow_duplicate=True),
        [Input("theme-store", "data")],
        prevent_initial_call="initial_duplicate",
    )
    def sync_theme_switch(theme_data):
        return theme_data.get("dark_mode", True) if theme_data else True

    @app.callback(
        Output("scheduler-data-store", "data"),
        [
            Input("refresh-schedules-button", "n_clicks"),
            Input("refresh-interval", "n_intervals"),
        ],
    )
    def refresh_data_collection_scheduler_data(n_clicks, n_intervals):
        """Refresh scheduler data when the refresh button is clicked or interval triggers"""
        data = get_scheduler_data()
        return data

    @app.callback(
        Output("scheduler-cards-container", "children"),
        [Input("scheduler-data-store", "data")],
    )
    def update_data_collection_scheduler_cards(data):
        """Update the scheduler cards with data from the store"""
        return create_scheduler_cards(data)

    @app.callback(
        [
            Output("update-status-message", "children"),
            Output("scheduler-data-store", "data", allow_duplicate=True),
        ],
        [Input("run-updates-button", "n_clicks")],
        prevent_initial_call=True,
    )
    def run_data_collection_updates(n_clicks):
        """Run all scheduled updates when the button is clicked"""
        if not n_clicks:
            return "", dash.no_update

        try:
            results = collector.run_scheduled_tasks()

            if not results:
                return (
                    dbc.Alert("No scheduled tasks were executed", color="info"),
                    get_scheduler_data(),
                )

            success_count = sum(1 for result in results.values() if result)
            error_count = sum(1 for result in results.values() if not result)

            if error_count == 0:
                return (
                    dbc.Alert(
                        f"Successfully executed {success_count} task(s)",
                        color="success",
                    ),
                    get_scheduler_data(),
                )
            else:
                return (
                    dbc.Alert(
                        f"Executed {success_count} task(s) with {error_count} error(s)",
                        color="warning",
                    ),
                    get_scheduler_data(),
                )
        except Exception as e:
            logger.error(f"Error running updates: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger"), dash.no_update

    @app.callback(
        [
            Output({"type": "save-status", "index": dash.ALL}, "children"),
            Output(
                "scheduler-data-store", "data", allow_duplicate=True
            ),  # Allow duplicate
        ],
        [Input({"type": "save-button", "index": dash.ALL}, "n_clicks")],
        [
            State({"type": "interval-type", "index": dash.ALL}, "value"),
            State({"type": "is-active", "index": dash.ALL}, "value"),
            State("scheduler-data-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def save_data_collection_changes(
        n_clicks_list, interval_types, is_active_values, all_data
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return [dash.no_update] * len(n_clicks_list), dash.no_update

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if not button_id:
            return [dash.no_update] * len(n_clicks_list), dash.no_update

        try:
            button_dict = json.loads(button_id)
            clicked_index = button_dict["index"]
            button_index_in_list = next(
                (
                    i
                    for i, x in enumerate(ctx.inputs_list[0])
                    if x["id"]["index"] == clicked_index
                ),
                None,
            )

            if button_index_in_list is None:
                return [dash.no_update] * len(n_clicks_list), dash.no_update

            schedule_id = clicked_index

            original_item = next(
                (item for item in all_data if item["id"] == schedule_id), None
            )
            if not original_item:
                result = dbc.Alert(
                    "Error: Could not find schedule data", color="danger"
                )
                return [
                    result if i == button_index_in_list else dash.no_update
                    for i in range(len(n_clicks_list))
                ], dash.no_update

            new_interval_type = interval_types[button_index_in_list]
            new_is_active = is_active_values[button_index_in_list]

            changes_made = False
            success_count = 0
            error_count = 0

            if new_interval_type != original_item["interval_type"]:
                success, _ = update_scheduler_record(
                    schedule_id, "interval_type", new_interval_type
                )
                changes_made = True
                if success:
                    success_count += 1
                else:
                    error_count += 1

            # Проверка для is_active changes
            if new_is_active != original_item["is_active"]:
                success, _ = update_scheduler_record(
                    schedule_id, "is_active", new_is_active
                )
                changes_made = True
                if success:
                    success_count += 1
                else:
                    error_count += 1

            if success_count == 0 and error_count == 0:
                return [dash.no_update] * len(n_clicks_list), dash.no_update
            elif error_count == 0:
                result = dbc.Alert(
                    f"Successfully updated {success_count} field(s)", color="success"
                )
                return [
                    result if i == button_index_in_list else dash.no_update
                    for i in range(len(n_clicks_list))
                ], get_scheduler_data()
            else:
                result = dbc.Alert(
                    f"Updated {success_count} field(s) with {error_count} error(s)",
                    color="warning",
                )
                # Refresh данных even with errors, as some changes may have succeeded
                return [
                    result if i == button_index_in_list else dash.no_update
                    for i in range(len(n_clicks_list))
                ], get_scheduler_data()

        except Exception as e:
            logger.error(f"Error saving changes: {e}")
            result = dbc.Alert(f"Error: {str(e)}", color="danger")
            return [
                result if i == 0 else dash.no_update for i in range(len(n_clicks_list))
            ], dash.no_update
