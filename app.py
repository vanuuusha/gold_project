from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import flask
from flask import Flask, render_template, redirect, url_for
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from loguru import logger
import sys

# Import database models and data collector
from database.init_database import (
    MetalType, DataSource, MetalPrice, CollectorSchedule, 
    User, UserRequest, Base, init_database
)
from database.data_collector import DataCollector
from settings import APP_CONFIG, DB_URL

# Set up loguru logger
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
logger.add("logs/app.log", rotation="10 MB", level="INFO")

# Initialize Flask server
server = Flask(__name__)
server.secret_key = "mysecretkey12345"  # Change this in production

# Initialize Dash app
app = Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,  # This is important for callbacks from different files
    routes_pathname_prefix='/dashboard/'
)

# Create collector
collector = DataCollector()

# Define the main layout with navigation
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Define the home layout
home_layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard/dashboard")),
            dbc.NavItem(dbc.NavLink("Data Collection", href="/dashboard/data-collection")),
            dbc.NavItem(dbc.NavLink("Database Management", href="/dashboard/database")),
            dbc.NavItem(dbc.NavLink("Reports", href="/dashboard/reports")),
        ],
        brand="Precious Metals Analytics",
        brand_href="/dashboard/",
        color="primary",
        dark=True,
    ),
    dbc.Container([
        html.H1("Welcome to Precious Metals Analytics", className="my-4"),
        html.P("This application collects and analyzes precious metal prices from various sources."),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Quick Overview"),
                    dbc.CardBody([
                        html.P("View the current status of data collection and latest price information."),
                        dbc.Button("View Dashboard", color="primary", href="/dashboard/dashboard"),
                    ])
                ], className="mb-4"),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Collection"),
                    dbc.CardBody([
                        html.P("Configure data collection schedules and trigger manual updates."),
                        dbc.Button("Data Collection Settings", color="primary", href="/dashboard/data-collection"),
                    ])
                ], className="mb-4"),
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Database Management"),
                    dbc.CardBody([
                        html.P("Manage database tables and view stored data."),
                        dbc.Button("Database Management", color="primary", href="/dashboard/database"),
                    ])
                ], className="mb-4"),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Reports"),
                    dbc.CardBody([
                        html.P("Generate visual and textual reports from collected data."),
                        dbc.Button("Generate Reports", color="primary", href="/dashboard/reports"),
                    ])
                ], className="mb-4"),
            ], width=6),
        ]),
    ], className="mt-4"),
])

# Callback to update the page content based on URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/dashboard/' or pathname == '/dashboard':
        return home_layout
    elif pathname == '/dashboard/data-collection':
        # This will be imported from data_collection.py
        from views.data_collection import layout
        return layout
    elif pathname == '/dashboard/database':
        # This will be imported from database_management.py
        from views.database_management import layout
        return layout
    elif pathname == '/dashboard/reports':
        # This will be imported from reports.py
        from views.reports import layout
        return layout
    elif pathname == '/dashboard/dashboard':
        # This will be imported from visual_dashboard.py
        from views.visual_dashboard import layout
        return layout
    else:
        return home_layout

# Flask routes for non-Dash pages
@server.route('/')
def index():
    return redirect('/dashboard/')

@server.route('/initialize-database')
def initialize_db():
    try:
        init_database()
        return "Database initialized successfully!"
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return f"Error initializing database: {str(e)}"

# Import views and register callbacks after the app is defined
from views import register_callbacks

# Register all callbacks
register_callbacks(app)

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Start the application
    server.run(
        host=APP_CONFIG.get('host', '0.0.0.0'),
        port=APP_CONFIG.get('port', 5001),
        debug=APP_CONFIG.get('debug', True)
    )