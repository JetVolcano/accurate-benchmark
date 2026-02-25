from copy import copy
from typing import Final

from rich.console import Console
from rich.style import Style
from rich.theme import Theme


def _create_console() -> Console:
    THEME: Final[Theme] = Theme(
        {
            "log.path": Style(color="yellow", dim=True),
            "log.time": Style(color="bright_cyan", bold=True),
            "repr.call": Style(color="bright_magenta", bold=True),
            "repr.number": Style(color="bright_cyan", bold=True),
            "rule.line": Style(dim=True),
            "primary": Style(color="bright_green", bold=True),
            "secondary": Style(color="cyan"),
        }
    )
    return Console(theme=THEME, log_path=False)


class _BenchmarkConsole:
    def __init__(self) -> None:
        self.__console: Console = _create_console()

    def set(self, console: Console | None = None) -> None:
        """Sets the console, if no console is provided it will reset to the default custom console

        Parameters
        ----------
        console : Console | None, optional, default=None
            The console to set
        """

        self.__console = copy(console) or _create_console()

    def reset(self) -> None:
        """Resets the console to the default custom console"""

        self.__console = _create_console()

    def clear(self) -> None:
        """Sets the console to the default console that rich provides"""

        self.__console = Console()

    def get(self) -> Console:
        """Returns the console

        Returns
        -------
        Console
            The console
        """

        return self.__console


__all__: list[str] = [
    "_BenchmarkConsole",
    "_create_console",
]
