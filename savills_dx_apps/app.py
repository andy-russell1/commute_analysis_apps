from __future__ import annotations

from shell.layout import configure_page
from shell.router import render_shell
from shell.state import init_shell_state
from shell.theme import apply_shell_theme


def main() -> None:
    configure_page()
    init_shell_state()
    apply_shell_theme()
    render_shell()


if __name__ == "__main__":
    main()
