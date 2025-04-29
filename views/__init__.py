import dash
from views import data_collection, database_management, visual_dashboard


def register_callbacks(app):
    # Импорт и регистрация колбэков из всех модулей представления
    from views.visual_dashboard import (
        register_callbacks as register_dashboard_callbacks,
    )
    from views.data_collection import (
        register_callbacks as register_collection_callbacks,
    )
    from views.database_management import (
        register_callbacks as register_database_callbacks,
    )

    register_dashboard_callbacks(app)
    register_collection_callbacks(app)
    register_database_callbacks(app)
