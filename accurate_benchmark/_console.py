from typing import Final

from rich.console import Console
from rich.style import Style
from rich.theme import Theme


def _create_console() -> Console:
    THEME: Final[Theme] = Theme(
        {
            "log.path": Style(color="yellow", dim=True),
            "log.time": Style(color="bright_cyan", bold=True),
            "repr.number": Style(color="bright_cyan", bold=True),
            "rule.line": Style(dim=True),
            "primary": Style(color="green", bold=True),
            "secondary": Style(color="cyan")
        }
    )
    return Console(theme=THEME, log_path=False)
