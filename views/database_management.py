from dash import html, dcc, dash_table, dash, callback_context
from dash.dependencies import Input, Output, State
from loguru import logger
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd
import json
from io import StringIO
from datetime import datetime, timedelta, date
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

from settings import DB_URL
from database.init_database import (
    Base,
    MetalType,
    DataSource,
    MetalPrice,
    CollectorSchedule,
    init_database,
)
from views.components import create_theme_switch

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

layout = html.Div(
    [
        dcc.Store(id="theme-store", storage_type="local"),
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
            id="navbar-database",
        ),
        dbc.Container(
            [
                html.H1("Database Management", className="my-4"),
                html.P("View and manage metal prices data."),
                html.Hr(),
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
                html.H3("Metal Prices Data", className="mt-4"),
                html.Div(id="table-data-container", className="mb-4"),
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
                dcc.Store(id="table-data-store"),
                dcc.Store(id="edited-data-store"),
                dcc.Store(id="filter-state"),
            ],
            id="database-container",
        ),
    ],
    id="database-page",
)


def get_table_data(limit=100, filters=None):
    try:
        session = Session()

        query = session.query(MetalPrice)

        # Apply filters
        if filters:
            # Apply metal type Фильтрация
            if filters.get("metal_types"):
                metal_types = [MetalType[mt] for mt in filters["metal_types"]]
                query = query.filter(MetalPrice.metal_type.in_(metal_types))

            if filters.get("data_sources"):
                data_sources = [DataSource[ds] for ds in filters["data_sources"]]
                query = query.filter(MetalPrice.source.in_(data_sources))

            if filters.get("start_date"):
                query = query.filter(MetalPrice.timestamp >= filters["start_date"])

            if filters.get("end_date"):
                query = query.filter(MetalPrice.timestamp <= filters["end_date"])

        if limit:
            query = query.limit(limit)

        results = query.all()

        data = []
        for result in results:
            item = {}
            for column in inspect(MetalPrice).columns.keys():
                value = getattr(result, column)

                # Обработка enum types
                if isinstance(value, (MetalType, DataSource)):
                    value = value.value

                item[column] = value

            data.append(item)

        df = pd.DataFrame(data)

        if df.empty:
            return df, "No data found"

        return df, None

    except Exception as e:
        logger.error(f"Error getting table data: {e}")
        return pd.DataFrame(), f"Error: {str(e)}"

    finally:
        session.close()


def update_table_data(edited_data):
    try:
        session = Session()

        logger.debug(f"Update data type: {type(edited_data)}")

        try:
            df = pd.read_json(StringIO(edited_data), orient="split")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"DataFrame shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to parse JSON data: {e}")
            return f"Error parsing data: {str(e)}"

        try:
            inspector = inspect(engine)
            pk_constraint = inspector.get_pk_constraint(MetalPrice.__tablename__)
            logger.debug(f"PK constraint: {pk_constraint}")

            if (
                isinstance(pk_constraint, dict)
                and "constrained_columns" in pk_constraint
            ):
                pk_columns = pk_constraint["constrained_columns"]
            else:
                logger.warning(f"Unexpected pk_constraint format: {pk_constraint}")
                pk_columns = []

            if not pk_columns:
                pk_columns = []
                for column in MetalPrice.__table__.columns:
                    if column.primary_key:
                        pk_columns.append(column.name)

            if not pk_columns:
                pk_columns = ["id"]  # Default primary key

            logger.debug(f"Primary key columns: {pk_columns}")
        except Exception as e:
            logger.error(f"Failed to identify primary keys: {e}")
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

        count = 0
        for _, row in df.iterrows():
            try:
                # Создание a Фильтрация condition На основе primary key(s)
                filter_condition = {}
                for pk in pk_columns:
                    if pk in row:
                        filter_condition[pk] = row[pk]
                    else:
                        logger.warning(f"Primary key {pk} not found in row")

                if not filter_condition:
                    logger.warning(f"No primary key values found in row, skipping")
                    continue

                # Получение  record из  базы данных
                record = session.query(MetalPrice).filter_by(**filter_condition).first()

                if not record:
                    logger.warning(f"No record found with filter: {filter_condition}")
                    continue

                for column, value in row.items():
                    if column not in pk_columns:
                        if column == "metal_type" and isinstance(value, str):
                            try:
                                setattr(record, column, MetalType[value])
                            except (KeyError, ValueError):
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
                                setattr(record, column, DataSource[value])
                            except (KeyError, ValueError):
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
                        elif column in [
                            "open_price",
                            "high_price",
                            "low_price",
                            "close_price",
                        ]:
                            if pd.isna(value) or value == "":
                                setattr(record, column, None)
                            else:
                                try:
                                    setattr(record, column, float(value))
                                except (ValueError, TypeError):
                                    logger.warning(
                                        f"Could not convert {value} to float for {column}"
                                    )
                        # Обработка boolean/integer flags
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
                        elif column == "timestamp" and isinstance(value, str):
                            try:
                                from dateutil import parser

                                setattr(record, column, parser.parse(value))
                            except Exception as e:
                                logger.warning(f"Could not parse date {value}: {e}")
                        else:
                            setattr(record, column, value)

                count += 1
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue

        session.commit()
        return f"Successfully updated {count} records"

    except Exception as e:
        session.rollback()
        logger.error(f"Error updating table data: {e}")
        return f"Error: {str(e)}"

    finally:
        session.close()


# Регистрация callbacks для базы данных management page
def register_callbacks(app):
    # Колбэк для темы страницы управления базой данных
    @app.callback(
        [
            Output("database-container", "className"),
            Output("navbar-database", "dark"),
            Output("navbar-database", "color"),
            Output("database-page", "className"),
        ],
        [Input("theme-store", "data")],
        prevent_initial_call=False,
    )
    def update_database_theme(theme_data):
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
        [
            Output("metal-type-filter", "options"),
            Output("data-source-filter", "options"),
        ],
        [Input("refresh-tables-button", "n_clicks")],
    )
    def load_database_filter_options(refresh_clicks):
        try:
            session = Session()

            metal_types = session.query(MetalPrice.metal_type).distinct().all()
            metal_type_options = []
            for mt in metal_types:
                metal_type = mt[0]
                if isinstance(metal_type, MetalType):
                    metal_type_options.append(
                        {"label": metal_type.value, "value": metal_type.name}
                    )

            if not metal_type_options:
                metal_type_options = [
                    {"label": metal_type.value, "value": metal_type.name}
                    for metal_type in MetalType
                ]

            data_sources = session.query(MetalPrice.source).distinct().all()
            data_source_options = []
            for ds in data_sources:
                data_source = ds[0]
                if isinstance(data_source, DataSource):
                    data_source_options.append(
                        {"label": data_source.value, "value": data_source.name}
                    )

            if not data_source_options:
                data_source_options = [
                    {"label": data_source.value, "value": data_source.name}
                    for data_source in DataSource
                ]

            return metal_type_options, data_source_options

        except Exception as e:
            logger.error(f"Error loading filter options: {e}")
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
    def load_database_table_data(
        limit,
        apply_clicks,
        refresh_clicks,
        metal_types,
        data_sources,
        start_date,
        end_date,
    ):
        # Подготовка фильтров
        filters = None
        if any([metal_types, data_sources, start_date, end_date]):
            filters = {
                "metal_types": metal_types,
                "data_sources": data_sources,
                "start_date": start_date,
                "end_date": end_date,
            }

        df, error = get_table_data(limit, filters)

        if error:
            return None, html.Div(
                [
                    dbc.Alert(error, color="warning"),
                ]
            )

        if df.empty:
            return None, html.Div("No data available")

        stored_data = df.to_json(date_format="iso", orient="split")

        if "close_price" in df.columns:
            df["close_price"] = df["close_price"].round(2)
        if "open_price" in df.columns:
            df["open_price"] = df["open_price"].round(2)
        if "high_price" in df.columns:
            df["high_price"] = df["high_price"].round(2)
        if "low_price" in df.columns:
            df["low_price"] = df["low_price"].round(2)

        inspector = inspect(engine)

        editable_columns = {}
        try:
            pk_constraint = inspector.get_pk_constraint(MetalPrice.__tablename__)
            logger.debug(f"Table display - PK constraint: {pk_constraint}")

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

            if not pk_columns:
                for column in MetalPrice.__table__.columns:
                    if column.primary_key:
                        pk_columns.append(column.name)

            if not pk_columns:
                pk_columns = ["id"]

            logger.debug(f"Table display - Primary key columns: {pk_columns}")

            for col in df.columns:
                editable_columns[col] = col not in pk_columns
        except Exception as e:
            logger.error(f"Error determining editable columns: {e}")
            editable_columns = {col: col != "id" for col in df.columns}

        data_table = dag.AgGrid(
            id="data-table",
            columnDefs=[
                {
                    "headerName": col,
                    "field": col,
                    "editable": editable_columns.get(col, False),
                }
                for col in df.columns
            ],
            rowData=df.to_dict("records"),
            dashGridOptions={
                "pagination": True,
                "paginationPageSize": 25,
            },
            defaultColDef={
                "sortable": True,
                "filter": True,
                "resizable": True,
                "minWidth": 150,
            },
        )

        return stored_data, html.Div(
            [
                html.P(f"Showing {len(df)} records"),
                data_table,
                html.Div(id="edit-status", className="mt-3"),
            ]
        )

    # Хранилище edited данных
    @app.callback(
        Output("edited-data-store", "data"),
        [Input("data-table", "rowData")],
        [State("table-data-store", "data")],
    )
    def store_database_edited_data(table_data, original_data):
        if not table_data or not original_data:
            return None

        try:
            edited_df = pd.DataFrame(table_data)

            # Log some debugging info
            logger.debug(f"Edited DataFrame columns: {edited_df.columns.tolist()}")
            logger.debug(f"Edited DataFrame shape: {edited_df.shape}")

            if edited_df.empty:
                logger.warning("Empty DataFrame in store_database_edited_data")
                return None

            # Преобразование к JSON with ISO date format к maintain compatibility
            return edited_df.to_json(date_format="iso", orient="split")

        except Exception as e:
            logger.error(f"Error in store_database_edited_data: {e}")
            return None

    @app.callback(
        Output("edit-confirmation-modal", "is_open"),
        [
            Input("save-changes-button", "n_clicks"),
            Input("cancel-edit-button", "n_clicks"),
            Input("confirm-edit-button", "n_clicks"),
        ],
        [State("edit-confirmation-modal", "is_open")],
    )
    def toggle_database_edit_modal(save_clicks, cancel_clicks, confirm_clicks, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "save-changes-button" and save_clicks:
            return True
        elif button_id in ["cancel-edit-button", "confirm-edit-button"]:
            return False

        return is_open

    @app.callback(
        Output("edit-status", "children"),
        [Input("confirm-edit-button", "n_clicks")],
        [State("edited-data-store", "data")],
    )
    def save_database_edited_data(confirm_clicks, edited_data):
        if not confirm_clicks or edited_data is None:
            return None

        logger.debug(f"Starting to save edited data. Data type: {type(edited_data)}")

        if not edited_data:
            return dbc.Alert("No data to save", color="warning")

        try:
            # Verify данных can be parsed before sending к Обновление Функция
            test_df = pd.read_json(StringIO(edited_data), orient="split")
            logger.debug(f"Validated data with shape: {test_df.shape}")
        except Exception as e:
            logger.error(f"Invalid data format in save_database_edited_data: {e}")
            return dbc.Alert(f"Invalid data format: {str(e)}", color="danger")

        result = update_table_data(edited_data)
        if "Error" in result:
            logger.error(f"Error in save_database_edited_data: {result}")
            return dbc.Alert(result, color="danger")
        else:
            logger.info(f"Successfully saved data: {result}")
            return dbc.Alert(result, color="success")

    @app.callback(
        Output("db-control-message", "children"), [Input("init-db-button", "n_clicks")]
    )
    def initialize_database_schema(n_clicks):
        if n_clicks is None:
            return ""

        try:
            init_database()
            return dbc.Alert("Database initialized successfully!", color="success")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return dbc.Alert(f"Error initializing database: {str(e)}", color="danger")

    @app.callback(
        Output("db-download-csv", "data"),
        [Input("export-csv-button", "n_clicks")],
        [State("table-data-store", "data")],
    )
    def export_database_csv(n_clicks, stored_data):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(StringIO(stored_data), orient="split")
            return dcc.send_data_frame(df.to_csv, "metal_prices.csv", index=False)
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return None

    @app.callback(
        Output("db-download-excel", "data"),
        [Input("export-excel-button", "n_clicks")],
        [State("table-data-store", "data")],
    )
    def export_database_excel(n_clicks, stored_data):
        if n_clicks is None or stored_data is None:
            return None

        try:
            df = pd.read_json(StringIO(stored_data), orient="split")
            return dcc.send_data_frame(df.to_excel, "metal_prices.xlsx", index=False)
        except Exception as e:
            logger.error(f"Error exporting Excel: {e}")
            return None

    @app.callback(
        [
            Output("metal-type-filter", "value"),
            Output("data-source-filter", "value"),
            Output("date-range-filter", "start_date"),
            Output("date-range-filter", "end_date"),
        ],
        [Input("clear-filters-button", "n_clicks")],
    )
    def clear_database_filters(n_clicks):
        if n_clicks is None:
            return None, None, None, None
        return None, None, None, None
