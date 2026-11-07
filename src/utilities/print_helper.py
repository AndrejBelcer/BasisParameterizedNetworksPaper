from typing import Iterable

from rich.console import Console
from rich.table import Table
from rich.text import Text

# Create a single, global console instance to be used across the project.
_console = Console()

def PrintGreen(text: str):
    """Prints text to the console in green."""
    _console.print(text, style="green")

def PrintYellow(text: str):
    """Prints text to the console in yellow, typically for warnings."""
    _console.print(text, style="yellow")

def PrintRed(text: str):
    """Prints text to the console in red, typically for errors."""
    _console.print(text, style="red")

def PrintSummary(summary: str):
    """Prints a formatted summary block to the console."""
    _console.print("\n" + "="*80, style="cyan")
    _console.print(Text(f"--- {summary} ---", style="bold cyan"))
    _console.print("="*80 + "\n", style="cyan")


def PrintFields(rows: Iterable[tuple[str, object]]):
    """Print labeled values with column widths fitted to the console."""
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column()

    for label, value in rows:
        table.add_row(Text(f"{label}:"), Text(str(value)))

    _console.print(table)


def PrintTable(headers: Iterable[str], rows: Iterable[tuple[object, ...]]):
    """Print a comparison table without manually padded columns."""
    table = Table(box=None, padding=(0, 2), pad_edge=False, header_style="bold cyan")

    for index, header in enumerate(headers):
        table.add_column(Text(str(header)), justify="left" if index == 0 else "right")

    for row in rows:
        table.add_row(*(Text(str(value)) for value in row))

    _console.print(table)


def FormatFields(rows: Iterable[tuple[str, object]], indent: str = "") -> str:
    """Format labeled values for plain-text reports."""
    values = [(str(label), str(value)) for label, value in rows]
    if not values:
        return ""

    label_width = max(len(label) for label, _ in values)
    return "\n".join(f"{indent}{label:<{label_width}}: {value}" for label, value in values)


def PrintStatus(text: str, kind: str = "info"):
    """Print an important message using a restrained status color."""
    styles = {"info": "", "success": "green", "warning": "yellow", "error": "red"}
    _console.print(Text(str(text), style=styles[kind]))


def Human_Format(num: int) -> str:
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
