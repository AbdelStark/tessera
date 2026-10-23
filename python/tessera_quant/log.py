# SPDX-License-Identifier: Apache-2.0 OR MIT
"""`tessera-quant`'s stderr lines (05 §2): readable by default, JSON lines with `--log-format json`.

An event has a name, `tessera.quant.<event>` in snake case, a message template and typed fields:
integers, booleans and strings of public data only (05 §3). A `{name}` in the template stands for
the field of that name. A readable line is the substituted message. A JSON line is one
`tessera.log-event/1` object (`spec/schemas/log-event-v1.schema.json`): `ts`, `level`, `target`,
`event`, `span_path` (empty: the front-end has no spans) and `fields`, which hold the substituted
`message` with the typed fields.
"""

import json
import logging
import re
import sys
import time

FORMATS = ("human", "json")
LEVELS = {
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
_LOGGER = logging.getLogger("tessera_quant")
_SNAKE = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")
_NAMES = {
    logging.ERROR: "error",
    logging.WARNING: "warn",
    logging.INFO: "info",
    logging.DEBUG: "debug",
}

Field = int | bool | str


def render(template: str, fields: dict[str, Field]) -> str:
    """The template with every `{name}` of a field replaced by its text; other braces stay."""

    def text(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in fields:
            return match.group(0)
        value = fields[name]
        return json.dumps(value) if isinstance(value, bool) else str(value)

    return re.sub(r"\{([^{}]*)\}", text, template)


class _Formatter(logging.Formatter):
    def __init__(self, fmt: str) -> None:
        super().__init__()
        self.json = fmt == "json"

    def format(self, record: logging.LogRecord) -> str:
        event: str = getattr(record, "tessera_event", "tessera.quant.message")
        fields: dict[str, Field] = dict(getattr(record, "tessera_fields", {}))
        message = render(str(record.msg), fields)
        if not self.json:
            return message
        seconds, fraction = divmod(record.created, 1)
        ts = (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
            + f".{int(fraction * 1e6):06d}Z"
        )
        line: dict[str, object] = {
            "ts": ts,
            "level": _NAMES.get(record.levelno, "error"),
            "target": record.name,
            "event": event,
            "span_path": [],
            "fields": fields | {"message": message},
        }
        return json.dumps(line, ensure_ascii=False, sort_keys=True)


def configure(fmt: str = "human", level: str = "info") -> None:
    """Writes `tessera_quant`'s events to stderr, one line each, from `level` up."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter(fmt))
    _LOGGER.handlers[:] = [handler]
    _LOGGER.setLevel(LEVELS[level])
    _LOGGER.propagate = False


def event(
    level: str, name: str, template: str, target: str = "tessera_quant", **fields: object
) -> None:
    """Emits `tessera.quant.<name>` at `level` (`error`, `warn`, `info`, `debug`).

    A field is an integer, a boolean or a string; anything else is refused, never coerced.
    """
    if not _SNAKE.fullmatch(name):
        raise ValueError(f"event {name!r} is not snake case")
    typed: dict[str, Field] = {}
    for key, value in fields.items():
        if not _SNAKE.fullmatch(key) or not isinstance(value, int | str):
            raise TypeError(f"field {key!r} is not public data of a typed kind")
        typed[key] = value
    logging.getLogger(target).log(
        LEVELS[level],
        template,
        extra={"tessera_event": f"tessera.quant.{name}", "tessera_fields": typed},
    )
