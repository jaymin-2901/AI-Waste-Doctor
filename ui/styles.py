"""
Qt Stylesheet (QSS) for AI Waste Doctor Desktop Application.
Features a modern Science-Fair environmental theme.
"""

from ui.theme import ThemeManager, get_app_style

# Expose a function to get style instead of a static string
def get_style(is_dark: bool | None = None) -> str:
    if is_dark is None:
        return get_app_style(ThemeManager.get_theme()["name"] == "dark")
    return get_app_style(is_dark)

# For backward compatibility during refactoring
APP_STYLE = get_app_style(is_dark=True)
