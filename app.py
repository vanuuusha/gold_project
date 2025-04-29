from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import flask
from flask import Flask, redirect, url_for
from datetime import datetime, timedelta
import os
from loguru import logger
import sys
import threading
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Импорт моделей базы данных и сборщика данных
from database.init_database import (
    MetalType,
    DataSource,
    MetalPrice,
    CollectorSchedule,
    Base,
    init_database,
    init_schedules,
)
from database.data_collector import DataCollector
from settings import APP_CONFIG, DB_URL

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")

collector = DataCollector()


def scheduler_thread():
    """Фоновый поток, периодически проверяющий запланированные задачи"""
    logger.info("Scheduler background thread started")
    collector = DataCollector()
    collector._run_task_scheduler()


# Инициализация сервера Flask
server = Flask(__name__)
server.secret_key = "mysecretkey12345"

app = Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    routes_pathname_prefix="/dashboard/",
    title="Precious Metals Analytics",
)

_original_dispatch = app.dispatch


def _custom_dispatch(self, request_body, request_options=None):
    try:
        return _original_dispatch(self, request_body, request_options)
    except KeyError as e:
        raise


app.dispatch = _custom_dispatch.__get__(app, app.__class__)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
        dcc.Store(id="theme-store", storage_type="local"),
    ],
    id="_body",
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/dashboard/" or pathname == "/dashboard" or pathname == "/":
        from views.visual_dashboard import layout

        return layout
    elif pathname == "/dashboard/data-collection":
        from views.data_collection import layout

        return layout
    elif pathname == "/dashboard/database":
        from views.database_management import layout

        return layout
    else:
        from views.visual_dashboard import layout

        return layout


@server.route("/")
def index():
    return redirect("/dashboard/")


@server.route("/initialize-database")
@server.route("/initialize-database/<reset_schedules>")
def initialize_db(reset_schedules=None):
    try:
        force_reset = reset_schedules == "true"
        init_database(force_reset_schedules=force_reset)
        return "Database initialized successfully!" + (
            " Schedules were reset." if force_reset else ""
        )
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return f"Error initializing database: {str(e)}"


@server.route("/initialize-schedules")
@server.route("/initialize-schedules/<force_reset>")
def initialize_schedules(force_reset=None):
    try:
        engine = create_engine(DB_URL)
        Session = sessionmaker(bind=engine)
        session = Session()

        do_reset = force_reset == "true"
        init_schedules(session, force_reset=do_reset)

        session.close()
        return "Schedules initialized successfully!" + (
            " (with reset)" if do_reset else ""
        )
    except Exception as e:
        logger.error(f"Scheduler initialization error: {e}")
        return f"Error initializing schedules: {str(e)}"


from views import register_callbacks


# Создание callback для темы
@app.callback(
    [
        Output("theme-store", "data", allow_duplicate=True),
        Output("_body", "className"),
        Output("_body", "data-theme"),
    ],
    [Input("theme-switch", "value")],
    prevent_initial_call="initial_duplicate",
)
def update_global_theme(dark_mode):
    if dark_mode is None:
        dark_mode = True

    theme_data = {"dark_mode": dark_mode}
    body_class = "dark-mode-active" if dark_mode else ""

    return theme_data, body_class, ""


register_callbacks(app)

app.clientside_callback(
    """
    function(data) {
        if (data && data.dark_mode) {
            document.documentElement.classList.add('dark-mode-active');
            document.body.classList.add('dark-mode-active');
            document.body.classList.remove('light-mode-active');
        } else {
            document.documentElement.classList.remove('dark-mode-active');
            document.body.classList.remove('dark-mode-active');
            document.body.classList.add('light-mode-active');
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("_body", "data-theme", allow_duplicate=True),
    Input("theme-store", "data"),
    prevent_initial_call=True,
)

if __name__ == "__main__":

    if APP_CONFIG.get("enable_scheduler", True):
        scheduler = threading.Thread(target=scheduler_thread)
        scheduler.daemon = True
        scheduler.start()
        logger.info("Started background scheduler thread")

    server.run(
        host=APP_CONFIG.get("host", "0.0.0.0"),
        port=APP_CONFIG.get("port", 5001),
        debug=APP_CONFIG.get("debug", True),
    )
