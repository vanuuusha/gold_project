from dash import html, dcc, dash_table, callback_context
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
from io import StringIO

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
                html.P("View and manage metal prices data."),
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
                                                    "Refresh Data",
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
                # Data Settings Section
                html.H3("Data Settings", className="mt-4"),
                dbc.Row(
                    [
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
                                    style={"height": "38px", "width": "100%"},
                                    placeholder="10-1000 rows",
                                ),
                            ],
                            width=6,
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
                                            options=[],  # Will be populated dynamically
                                            multi=True,
                                            style={"height": "38px"},
                                        ),
                                    ],
                                    width=4,
                                    className="mb-3",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Data Source"),
                                        dcc.Dropdown(
                                            id="data-source-filter",
                                            options=[],  # Will be populated dynamically
                                            multi=True,
                                            style={"height": "38px"},
                                        ),
                                    ],
                                    width=4,
                                    className="mb-3",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Date Range"),
                                        dcc.DatePickerRange(
                                            id="date-range-filter",
                                            start_date_placeholder_text="Start Date",
                                            end_date_placeholder_text="End Date",
                                            style={"width": "100%"},
                                            className="date-range-picker",
                                        ),
                                    ],
                                    width=4,
                                    className="mb-3",
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
                                            className="me-2",
                                        ),
                                        dbc.Button(
                                            "Clear Filters",
                                            id="clear-filters-button",
                                            color="secondary",
                                            className="ms-2",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                    ],
                ),
                # Table Display Section
                html.H3("Metal Prices Data", className="mt-4"),
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
                                                dbc.Button(
                                                    "Save Changes",
                                                    id="save-changes-button",
                                                    color="primary",
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
                # Modal for edit confirmation
                dbc.Modal(
                    [
                        dbc.ModalHeader("Confirm Changes"),
                        dbc.ModalBody(
                            "Are you sure you want to save these changes to the database?"
                        ),
                        dbc.ModalFooter(
                            [
                                dbc.Button(
                                    "Cancel",
                                    id="cancel-edit-button",
                                    className="me-2",
                                    color="secondary",
                                ),
                                dbc.Button(
                                    "Save", id="confirm-edit-button", color="primary"
                                ),
                            ]
                        ),
                    ],
                    id="edit-confirmation-modal",
                    centered=True,
                ),
                # Hidden divs for storing data
                dcc.Store(id="table-data-store"),
                dcc.Store(id="edited-data-store"),
                dcc.Store(id="filter-state"),
            ],
            className="mt-4",
        ),
    ]
)


# Helper function to get table data
def get_table_data(limit=100, filters=None):
    try:
        session = Session()

        # Start query
        query = session.query(MetalPrice)

        # Apply filters
        if filters:
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
            for column in inspect(MetalPrice).columns.keys():
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


# Helper function to update database with edited data
def update_table_data(edited_data):
    try:
        session = Session()

        # Add debug info
        logger.debug(f"Update data type: {type(edited_data)}")

        # Safely convert JSON to DataFrame using StringIO
        try:
            df = pd.read_json(StringIO(edited_data), orient="split")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"DataFrame shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to parse JSON data: {e}")
            return f"Error parsing data: {str(e)}"

        # Identify primary key
        try:
            inspector = inspect(engine)
            pk_constraint = inspector.get_pk_constraint(MetalPrice.__tablename__)
            logger.debug(f"PK constraint: {pk_constraint}")

            # Handle different return formats of get_pk_constraint
            if (
                isinstance(pk_constraint, dict)
                and "constrained_columns" in pk_constraint
            ):
                pk_columns = pk_constraint["constrained_columns"]
            else:
                # Direct list or alternative format
                logger.warning(f"Unexpected pk_constraint format: {pk_constraint}")
                pk_columns = []

            if not pk_columns:
                # Try to get primary key info directly from the model
                pk_columns = []
                for column in MetalPrice.__table__.columns:
                    if column.primary_key:
                        pk_columns.append(column.name)

            if not pk_columns:
                pk_columns = ["id"]  # Default primary key

            logger.debug(f"Primary key columns: {pk_columns}")
        except Exception as e:
            logger.error(f"Failed to identify primary keys: {e}")
            # Try to get primary key info directly from the model as fallback
            try:
                pk_columns = []
                for column in MetalPrice.__table__.columns:
                    if column.primary_key:
                        pk_columns.append(column.name)
                if pk_columns:
                    logger.info(f"Retrieved primary keys from model: {pk_columns}")
                else:
                    pk_columns = ["id"]  # Default to id if there's an error
            except Exception as inner_e:
                logger.error(f"Failed to get primary keys from model: {inner_e}")
                pk_columns = ["id"]  # Default to id if there's an error

        # Update each row
        count = 0
        for _, row in df.iterrows():
            try:
                # Create a filter condition based on primary key(s)
                filter_condition = {}
                for pk in pk_columns:
                    if pk in row:
                        filter_condition[pk] = row[pk]
                    else:
                        logger.warning(f"Primary key {pk} not found in row")

                if not filter_condition:
                    logger.warning(f"No primary key values found in row, skipping")
                    continue

                # Get the record from the database
                record = session.query(MetalPrice).filter_by(**filter_condition).first()

                if not record:
                    logger.warning(f"No record found with filter: {filter_condition}")
                    continue

                # Update non-primary key columns
                for column, value in row.items():
                    if column not in pk_columns:
                        # Handle enum fields
                        if column == "metal_type" and isinstance(value, str):
                            try:
                                # Try to convert string to enum
                                setattr(record, column, MetalType[value])
                            except (KeyError, ValueError):
                                # If value is not enum name but display value, find by value
                                enum_found = False
                                for enum_val in MetalType:
                                    if enum_val.value == value:
                                        setattr(record, column, enum_val)
                                        enum_found = True
                                        break
                                if not enum_found:
                                    logger.warning(
                                        f"Could not map {value} to MetalType enum"
                                    )
                        elif column == "source" and isinstance(value, str):
                            try:
                                # Try to convert string to enum
                                setattr(record, column, DataSource[value])
                            except (KeyError, ValueError):
                                # If value is not enum name but display value, find by value
                                enum_found = False
                                for enum_val in DataSource:
                                    if enum_val.value == value:
                                        setattr(record, column, enum_val)
                                        enum_found = True
                                        break
                                if not enum_found:
                                    logger.warning(
                                        f"Could not map {value} to DataSource enum"
                                    )
                        # Handle numeric fields
                        elif column in [
                            "open_price",
                            "high_price",
                            "low_price",
                            "close_price",
                        ]:
                            # Convert to float or None
                            if pd.isna(value) or value == "":
                                setattr(record, column, None)
                            else:
                                try:
                                    setattr(record, column, float(value))
                                except (ValueError, TypeError):
                                    logger.warning(
                                        f"Could not convert {value} to float for {column}"
                                    )
                        # Handle boolean/integer flags
                        elif column in ["is_checked", "is_market_closed"]:
                            try:
                                setattr(record, column, int(value))
                            except (ValueError, TypeError):
                                if isinstance(value, bool):
                                    setattr(record, column, 1 if value else 0)
                                else:
                                    logger.warning(
                                        f"Could not convert {value} to int for {column}"
                                    )
                        # Handle dates
                        elif column == "timestamp" and isinstance(value, str):
                            try:
                                # Parse datetime string
                                from dateutil import parser

                                setattr(record, column, parser.parse(value))
                            except Exception as e:
                                logger.warning(f"Could not parse date {value}: {e}")
                        else:
                            # Default handling
                            setattr(record, column, value)

                count += 1
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue

        # Commit changes
        session.commit()
        return f"Successfully updated {count} records"

    except Exception as e:
        session.rollback()
        logger.error(f"Error updating table data: {e}")
        return f"Error: {str(e)}"

    finally:
        session.close()


# Register callbacks for database management page
def register_callbacks(app):
    # Initialize filter options on page load
    @app.callback(
        [
            Output("metal-type-filter", "options"),
            Output("data-source-filter", "options"),
        ],
        [Input("refresh-tables-button", "n_clicks")],
    )
    def load_filter_options(refresh_clicks):
        try:
            session = Session()

            # Get unique metal types from the database
            metal_types = session.query(MetalPrice.metal_type).distinct().all()
            metal_type_options = []
            for mt in metal_types:
                metal_type = mt[0]
                if isinstance(metal_type, MetalType):
                    metal_type_options.append(
                        {"label": metal_type.value, "value": metal_type.name}
                    )

            # Fallback to enum values if no records
            if not metal_type_options:
                metal_type_options = [
                    {"label": metal_type.value, "value": metal_type.name}
                    for metal_type in MetalType
                ]

            # Get unique data sources from the database
            data_sources = session.query(MetalPrice.source).distinct().all()
            data_source_options = []
            for ds in data_sources:
                data_source = ds[0]
                if isinstance(data_source, DataSource):
                    data_source_options.append(
                        {"label": data_source.value, "value": data_source.name}
                    )

            # Fallback to enum values if no records
            if not data_source_options:
                data_source_options = [
                    {"label": data_source.value, "value": data_source.name}
                    for data_source in DataSource
                ]

            return metal_type_options, data_source_options

        except Exception as e:
            logger.error(f"Error loading filter options: {e}")
            # Fallback to enum values
            metal_type_options = [
                {"label": metal_type.value, "value": metal_type.name}
                for metal_type in MetalType
            ]
            data_source_options = [
                {"label": data_source.value, "value": data_source.name}
                for data_source in DataSource
            ]
            return metal_type_options, data_source_options
        finally:
            session.close()

    # Load table data
    @app.callback(
        [
            Output("table-data-store", "data"),
            Output("table-data-container", "children"),
        ],
        [
            Input("rows-input", "value"),
            Input("apply-filters-button", "n_clicks"),
            Input("refresh-tables-button", "n_clicks"),
        ],
        [
            State("metal-type-filter", "value"),
            State("data-source-filter", "value"),
            State("date-range-filter", "start_date"),
            State("date-range-filter", "end_date"),
        ],
    )
    def load_table_data(
        limit,
        apply_clicks,
        refresh_clicks,
        metal_types,
        data_sources,
        start_date,
        end_date,
    ):
        # Prepare filters
        filters = None
        if any([metal_types, data_sources, start_date, end_date]):
            filters = {
                "metal_types": metal_types,
                "data_sources": data_sources,
                "start_date": start_date,
                "end_date": end_date,
            }

        # Get table data
        df, error = get_table_data(limit, filters)

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
        if "close_price" in df.columns:
            df["close_price"] = df["close_price"].round(2)
        if "open_price" in df.columns:
            df["open_price"] = df["open_price"].round(2)
        if "high_price" in df.columns:
            df["high_price"] = df["high_price"].round(2)
        if "low_price" in df.columns:
            df["low_price"] = df["low_price"].round(2)

        # Determine editable columns (exclude primary keys)
        inspector = inspect(engine)

        editable_columns = {}
        try:
            # Get primary key columns with better error handling
            pk_constraint = inspector.get_pk_constraint(MetalPrice.__tablename__)
            logger.debug(f"Table display - PK constraint: {pk_constraint}")

            # Handle different return formats of get_pk_constraint
            if (
                isinstance(pk_constraint, dict)
                and "constrained_columns" in pk_constraint
            ):
                pk_columns = pk_constraint["constrained_columns"]
            else:
                # Direct list or alternative format
                logger.warning(
                    f"Table display - Unexpected pk_constraint format: {pk_constraint}"
                )
                pk_columns = []

            # Try to get from model if not found
            if not pk_columns:
                for column in MetalPrice.__table__.columns:
                    if column.primary_key:
                        pk_columns.append(column.name)

            # Default to id if still not found
            if not pk_columns:
                pk_columns = ["id"]

            logger.debug(f"Table display - Primary key columns: {pk_columns}")

            for col in df.columns:
                editable_columns[col] = col not in pk_columns
        except Exception as e:
            logger.error(f"Error determining editable columns: {e}")
            # Default to id as primary key if can't determine
            editable_columns = {col: col != "id" for col in df.columns}

        # Create data table with editing enabled
        data_table = dash_table.DataTable(
            id="data-table",
            columns=[
                {"name": col, "id": col, "editable": editable_columns.get(col, False)}
                for col in df.columns
            ],
            data=df.to_dict("records"),
            page_size=25,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            editable=True,
            row_deletable=False,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "minWidth": "150px",
                "maxWidth": "300px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_header={
                "backgroundColor": "lightgrey",
                "fontWeight": "bold",
                "whiteSpace": "normal",
                "height": "auto",
            },
            css=[
                {
                    "selector": ".dash-cell div.dash-cell-value",
                    "rule": "display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;",
                }
            ],
            tooltip_data=[
                {
                    column: {"value": str(value), "type": "markdown"}
                    for column, value in row.items()
                }
                for row in df.to_dict("records")
            ],
            tooltip_duration=None,
        )

        return stored_data, html.Div(
            [
                html.P(f"Showing {len(df)} records"),
                data_table,
                html.Div(id="edit-status", className="mt-3"),
            ]
        )

    # Store edited data
    @app.callback(
        Output("edited-data-store", "data"),
        [Input("data-table", "data")],
        [State("table-data-store", "data")],
    )
    def store_edited_data(table_data, original_data):
        if not table_data or not original_data:
            return None

        try:
            # Create DataFrame from edited table data
            edited_df = pd.DataFrame(table_data)

            # Log some debugging info
            logger.debug(f"Edited DataFrame columns: {edited_df.columns.tolist()}")
            logger.debug(f"Edited DataFrame shape: {edited_df.shape}")

            # Ensure we have a valid DataFrame before converting to JSON
            if edited_df.empty:
                logger.warning("Empty DataFrame in store_edited_data")
                return None

            # Convert to JSON with ISO date format to maintain compatibility
            return edited_df.to_json(date_format="iso", orient="split")

        except Exception as e:
            logger.error(f"Error in store_edited_data: {e}")
            return None

    # Show edit confirmation modal
    @app.callback(
        Output("edit-confirmation-modal", "is_open"),
        [
            Input("save-changes-button", "n_clicks"),
            Input("cancel-edit-button", "n_clicks"),
            Input("confirm-edit-button", "n_clicks"),
        ],
        [State("edit-confirmation-modal", "is_open")],
    )
    def toggle_edit_modal(save_clicks, cancel_clicks, confirm_clicks, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "save-changes-button" and save_clicks:
            return True
        elif button_id in ["cancel-edit-button", "confirm-edit-button"]:
            return False

        return is_open

    # Save edited data to database
    @app.callback(
        Output("edit-status", "children"),
        [Input("confirm-edit-button", "n_clicks")],
        [State("edited-data-store", "data")],
    )
    def save_edited_data(confirm_clicks, edited_data):
        if not confirm_clicks or edited_data is None:
            return None

        logger.debug(f"Starting to save edited data. Data type: {type(edited_data)}")

        if not edited_data:
            return dbc.Alert("No data to save", color="warning")

        try:
            # Verify data can be parsed before sending to update function
            test_df = pd.read_json(StringIO(edited_data), orient="split")
            logger.debug(f"Validated data with shape: {test_df.shape}")
        except Exception as e:
            logger.error(f"Invalid data format in save_edited_data: {e}")
            return dbc.Alert(f"Invalid data format: {str(e)}", color="danger")

        result = update_table_data(edited_data)
        if "Error" in result:
            logger.error(f"Error in save_edited_data: {result}")
            return dbc.Alert(result, color="danger")
        else:
            logger.info(f"Successfully saved data: {result}")
            return dbc.Alert(result, color="success")

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
        [State("table-data-store", "data")],
    )
    def export_csv(n_clicks, stored_data):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(StringIO(stored_data), orient="split")
            return dcc.send_data_frame(df.to_csv, "metal_prices.csv", index=False)
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return None

    # Export to Excel
    @app.callback(
        Output("db-download-excel", "data"),
        [Input("export-excel-button", "n_clicks")],
        [State("table-data-store", "data")],
    )
    def export_excel(n_clicks, stored_data):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(StringIO(stored_data), orient="split")
            return dcc.send_data_frame(df.to_excel, "metal_prices.xlsx", index=False)
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
