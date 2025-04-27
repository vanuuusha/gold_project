import dash
from views import data_collection, database_management, visual_dashboard, reports


def register_callbacks(app):
    """Register callbacks from all view modules."""
    data_collection.register_callbacks(app)
    database_management.register_callbacks(app)
    visual_dashboard.register_callbacks(app)
    reports.register_callbacks(app)
