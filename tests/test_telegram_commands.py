"""The command registry and the router must name the same set.

Until 2026-09-07 three hand-maintained lists described the bot's surface and
all three disagreed: the router dispatched ~27 commands, `_MENU_COMMANDS`
registered 12, `/help` listed ~22. `/record` was in the ☰ menu but missing
from `/help`; `/match`, `/fill`, `/league`, `/parlays`, `/summary` and
`/clear` were reachable but advertised nowhere.

`_COMMANDS` is now the single definition and `/help` + `_MENU_COMMANDS` are
derived from it. This file is what keeps it honest: it parses the router's own
dispatch branches out of the source and asserts set equality. Add a command to
the router without registering it (or vice versa) and this fails.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.pipeline import telegram_bot as TB

BOT_SRC = Path(TB.__file__).read_text(encoding="utf-8")


def _router_commands() -> set[str]:
    """Every command name the dispatch chain compares `cmd` against.

    Matches the three shapes the router actually uses:
        cmd == "/live"
        cmd in ("/sfide", "/h2h")
        cmd.startswith("/player")
    """
    # NOTE: the character class MUST allow digits — `/h2h` is a real alias and
    # an `[a-z]+` class silently misses it, which is how the first run of this
    # file "found" a registered-but-unreachable command that was reachable all
    # along. A name-shaped signal is a hypothesis about content.
    cmd_re = r"(/[a-z0-9_]+)"
    names: set[str] = set()
    for m in re.finditer(r'cmd\s*==\s*"' + cmd_re + r'"', BOT_SRC):
        names.add(m.group(1))
    for m in re.finditer(r'cmd\s+in\s+\(([^)]*)\)', BOT_SRC):
        names.update(re.findall(r'"' + cmd_re + r'"', m.group(1)))
    for m in re.finditer(r'cmd\.startswith\("' + cmd_re + r'"\)', BOT_SRC):
        names.add(m.group(1))
    return {n.lstrip("/") for n in names}


def test_the_router_parser_actually_finds_commands():
    """Guard the guard: a regex that matches nothing would make every
    assertion below vacuously pass."""
    found = _router_commands()
    assert len(found) >= 20, f"router parser found only {found} — regex is broken"
    # Known-present anchors, one per dispatch shape.
    assert "live" in found       # cmd == "/live"
    assert "sfide" in found      # cmd in ("/sfide", "/h2h")
    assert "h2h" in found        # ...and the digit-bearing alias next to it
    assert "player" in found     # cmd.startswith("/player")


def test_every_registered_command_is_reachable_in_the_router():
    missing = sorted(TB.ALL_COMMAND_NAMES - _router_commands())
    assert not missing, (
        f"registered but the router never dispatches them: {missing} — "
        "either add a branch or drop them from _COMMANDS"
    )


def test_every_router_command_is_registered():
    unregistered = sorted(_router_commands() - TB.ALL_COMMAND_NAMES)
    assert not unregistered, (
        f"reachable but undocumented: {unregistered} — add them to _COMMANDS "
        "(in_menu=False is fine) so /help lists them"
    )


def test_menu_is_derived_and_within_telegram_limits():
    menu = TB._MENU_COMMANDS
    assert menu, "the ☰ menu must not be empty"
    assert len(menu) <= 100, "Telegram setMyCommands accepts at most 100"
    for entry in menu:
        assert set(entry) == {"command", "description"}
        assert entry["description"], f"/{entry['command']} has a blank menu label"
        assert not entry["command"].startswith("/"), "setMyCommands wants bare names"
        assert entry["command"] in TB.ALL_COMMAND_NAMES


def test_legacy_world_cup_commands_are_not_in_the_menu():
    """WC2026 is over. The commands still answer, but they must not occupy
    slots in the ☰ menu that the money surface needs."""
    legacy = {c["command"] for c in TB._COMMANDS if c["group"] == TB._GROUP_LEGACY}
    assert legacy, "the legacy group should still describe the WC surface"
    in_menu = {e["command"] for e in TB._MENU_COMMANDS} & legacy
    assert not in_menu, f"legacy WC commands in the menu: {sorted(in_menu)}"


def test_help_lists_every_fanta_and_betting_command():
    """The /record regression: in the menu, absent from /help."""
    help_text = TB._handle_help()
    for c in TB._COMMANDS:
        if c["group"] in (TB._GROUP_FANTA, TB._GROUP_BETTING):
            assert f"/{c['command']}" in help_text, (
                f"/{c['command']} is registered but /help does not mention it"
            )


def test_help_mentions_record_specifically():
    """A true-positive anchor: this is the command that was actually missing,
    so a refactor that silently empties the loop above is caught here."""
    assert "/record" in TB._handle_help()


@pytest.mark.parametrize("field", ["command", "help", "group"])
def test_registry_rows_are_complete(field):
    for c in TB._COMMANDS:
        assert c.get(field), f"{c.get('command', '?')} is missing '{field}'"


def test_no_duplicate_command_names():
    seen: list[str] = []
    for c in TB._COMMANDS:
        seen.append(c["command"])
        seen.extend(c.get("aliases", []))
    dupes = {n for n in seen if seen.count(n) > 1}
    assert not dupes, f"a name is claimed twice: {sorted(dupes)}"
