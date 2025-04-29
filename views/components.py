from dash import html
import dash_bootstrap_components as dbc


def create_theme_switch():
    """Создание компонента переключения темы для тёмного/светлого режима"""
    return html.Div(
        [
            dbc.Switch(
                id="theme-switch",
                label="Dark Mode",
                value=True,
                className="ms-auto",
            ),
        ],
        className="ms-auto d-flex align-items-center no-print",
        style={"marginRight": "20px"},
    )
