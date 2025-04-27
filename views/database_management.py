from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import sqlalchemy
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    DateTime,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from loguru import logger
import json

from settings import DB_URL
from database.init_database import (
    Base,
    MetalType,
    DataSource,
    MetalPrice,
    CollectorSchedule,
    User,
    UserRequest,
    init_database,
)

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

# Define database management layout
layout = html.Div(
    [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard/")),
                dbc.NavItem(
                    dbc.NavLink("Data Collection", href="/dashboard/data-collection")
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        "Database Management", href="/dashboard/database", active=True
                    )
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
                html.H1("Database Management", className="my-4"),
                html.P("View and manage database tables."),
                html.Hr(),
                # Database Control Section
                html.H3("Database Controls", className="mt-4"),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Initialize Database",
                                                    id="init-db-button",
                                                    color="danger",
                                                    className="me-2",
                                                ),
                                                dbc.Button(
                                                    "Refresh Tables",
                                                    id="refresh-tables-button",
                                                    color="primary",
                                                    className="me-2",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                html.Div(id="db-control-message", className="mt-3"),
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                # Table Selection Section
                html.H3("Table Selection", className="mt-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Table"),
                                dcc.Dropdown(
                                    id="table-dropdown",
                                    options=[
                                        {
                                            "label": "Metal Prices",
                                            "value": "metal_prices",
                                        },
                                        {
                                            "label": "Collection Schedules",
                                            "value": "collector_schedules",
                                        },
                                        {"label": "Users", "value": "users"},
                                        {
                                            "label": "User Requests",
                                            "value": "user_requests",
                                        },
                                    ],
                                    value="metal_prices",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Rows to Display"),
                                dcc.Input(
                                    id="rows-input",
                                    type="number",
                                    value=100,
                                    min=10,
                                    max=1000,
                                    step=10,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Filter"),
                                dcc.Input(
                                    id="filter-input",
                                    type="text",
                                    placeholder="Enter filter text",
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-3",
                ),
                # Filtering Options for Metal Prices
                html.Div(
                    id="metal-prices-filters",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Metal Type"),
                                        dcc.Dropdown(
                                            id="metal-type-filter",
                                            options=[
                                                {
                                                    "label": metal_type.value,
                                                    "value": metal_type.name,
                                                }
                                                for metal_type in MetalType
                                            ],
                                            multi=True,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Data Source"),
                                        dcc.Dropdown(
                                            id="data-source-filter",
                                            options=[
                                                {
                                                    "label": data_source.value,
                                                    "value": data_source.name,
                                                }
                                                for data_source in DataSource
                                            ],
                                            multi=True,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Date Range"),
                                        dcc.DatePickerRange(
                                            id="date-range-filter",
                                            start_date_placeholder_text="Start Date",
                                            end_date_placeholder_text="End Date",
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
                                            "Apply Filters",
                                            id="apply-filters-button",
                                            color="primary",
                                            className="mt-3",
                                        ),
                                        dbc.Button(
                                            "Clear Filters",
                                            id="clear-filters-button",
                                            color="secondary",
                                            className="mt-3 ms-2",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    style={"display": "none"},
                ),
                # Table Display Section
                html.H3("Table Data", className="mt-4"),
                html.Div(id="table-data-container", className="mb-4"),
                # Data Export Section
                html.H3("Export Data", className="mt-4"),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
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
                                            ]
                                        ),
                                    ]
                                ),
                                html.Div(id="export-message", className="mt-3"),
                                dcc.Download(id="db-download-csv"),
                                dcc.Download(id="db-download-excel"),
                            ]
                        )
                    ]
                ),
                # Hidden divs for storing data
                dcc.Store(id="table-data-store"),
                dcc.Store(id="filter-state"),
            ],
            className="mt-4",
        ),
    ]
)


# Helper function to get table data
def get_table_data(table_name, limit=100, filters=None):
    try:
        session = Session()

        # Map table name to SQLAlchemy model
        table_model_map = {
            "metal_prices": MetalPrice,
            "collector_schedules": CollectorSchedule,
            "users": User,
            "user_requests": UserRequest,
        }

        model = table_model_map.get(table_name)
        if not model:
            return pd.DataFrame(), f"Table {table_name} not found"

        # Start query
        query = session.query(model)

        # Apply filters
        if filters and table_name == "metal_prices":
            # Apply metal type filter
            if filters.get("metal_types"):
                metal_types = [MetalType[mt] for mt in filters["metal_types"]]
                query = query.filter(MetalPrice.metal_type.in_(metal_types))

            # Apply data source filter
            if filters.get("data_sources"):
                data_sources = [DataSource[ds] for ds in filters["data_sources"]]
                query = query.filter(MetalPrice.source.in_(data_sources))

            # Apply date range filter
            if filters.get("start_date"):
                query = query.filter(MetalPrice.timestamp >= filters["start_date"])

            if filters.get("end_date"):
                query = query.filter(MetalPrice.timestamp <= filters["end_date"])

        # Apply limit
        if limit:
            query = query.limit(limit)

        # Execute query
        results = query.all()

        # Convert to DataFrame
        data = []
        for result in results:
            # Convert SQLAlchemy model to dict
            item = {}
            for column in inspect(model).columns.keys():
                value = getattr(result, column)

                # Handle enum types
                if isinstance(value, (MetalType, DataSource)):
                    value = value.value

                item[column] = value

            data.append(item)

        df = pd.DataFrame(data)

        # Handle empty results
        if df.empty:
            return df, "No data found"

        return df, None

    except Exception as e:
        logger.error(f"Error getting table data: {e}")
        return pd.DataFrame(), f"Error: {str(e)}"

    finally:
        session.close()


# Register callbacks for database management page
def register_callbacks(app):
    # Show/hide metal prices filters based on selected table
    @app.callback(
        Output("metal-prices-filters", "style"), [Input("table-dropdown", "value")]
    )
    def toggle_metal_prices_filters(table_name):
        if table_name == "metal_prices":
            return {"display": "block"}
        return {"display": "none"}

    # Load table data
    @app.callback(
        [
            Output("table-data-store", "data"),
            Output("table-data-container", "children"),
        ],
        [
            Input("table-dropdown", "value"),
            Input("rows-input", "value"),
            Input("apply-filters-button", "n_clicks"),
        ],
        [
            State("metal-type-filter", "value"),
            State("data-source-filter", "value"),
            State("date-range-filter", "start_date"),
            State("date-range-filter", "end_date"),
        ],
    )
    def load_table_data(
        table_name, limit, apply_clicks, metal_types, data_sources, start_date, end_date
    ):
        # Prepare filters
        filters = None
        if table_name == "metal_prices" and any(
            [metal_types, data_sources, start_date, end_date]
        ):
            filters = {
                "metal_types": metal_types,
                "data_sources": data_sources,
                "start_date": start_date,
                "end_date": end_date,
            }

        # Get table data
        df, error = get_table_data(table_name, limit, filters)

        if error:
            return None, html.Div(
                [
                    dbc.Alert(error, color="warning"),
                ]
            )

        if df.empty:
            return None, html.Div("No data available")

        # Store data for export
        stored_data = df.to_json(date_format="iso", orient="split")

        # Format specific columns for display
        if table_name == "metal_prices":
            if "close_price" in df.columns:
                df["close_price"] = df["close_price"].round(2)
            if "open_price" in df.columns:
                df["open_price"] = df["open_price"].round(2)
            if "high_price" in df.columns:
                df["high_price"] = df["high_price"].round(2)
            if "low_price" in df.columns:
                df["low_price"] = df["low_price"].round(2)

        # Create data table
        data_table = dash_table.DataTable(
            id="data-table",
            columns=[{"name": col, "id": col} for col in df.columns],
            data=df.to_dict("records"),
            page_size=25,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
        )

        return stored_data, html.Div([html.P(f"Showing {len(df)} records"), data_table])

    # Initialize database
    @app.callback(
        Output("db-control-message", "children"), [Input("init-db-button", "n_clicks")]
    )
    def initialize_database(n_clicks):
        if n_clicks is None:
            return ""

        try:
            init_database()
            return dbc.Alert("Database initialized successfully!", color="success")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return dbc.Alert(f"Error initializing database: {str(e)}", color="danger")

    # Export to CSV
    @app.callback(
        Output("db-download-csv", "data"),
        [Input("export-csv-button", "n_clicks")],
        [State("table-data-store", "data"), State("table-dropdown", "value")],
    )
    def export_csv(n_clicks, stored_data, table_name):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(stored_data, orient="split")
            return dcc.send_data_frame(df.to_csv, f"{table_name}.csv", index=False)
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return None

    # Export to Excel
    @app.callback(
        Output("db-download-excel", "data"),
        [Input("export-excel-button", "n_clicks")],
        [State("table-data-store", "data"), State("table-dropdown", "value")],
    )
    def export_excel(n_clicks, stored_data, table_name):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(stored_data, orient="split")
            return dcc.send_data_frame(df.to_excel, f"{table_name}.xlsx", index=False)
        except Exception as e:
            logger.error(f"Error exporting Excel: {e}")
            return None

    # Clear filters
    @app.callback(
        [
            Output("metal-type-filter", "value"),
            Output("data-source-filter", "value"),
            Output("date-range-filter", "start_date"),
            Output("date-range-filter", "end_date"),
        ],
        [Input("clear-filters-button", "n_clicks")],
    )
    def clear_filters(n_clicks):
        if n_clicks is None:
            return None, None, None, None
        return None, None, None, None
