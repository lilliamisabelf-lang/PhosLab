"""CLI: validate a protocol YAML against the schema in scripts.protocol.

Usage:
    uv run --project percept_mapper python percept_mapper/scripts/validate_protocol.py
    uv run --project percept_mapper python percept_mapper/scripts/validate_protocol.py path/to/protocol.yaml

Exits 0 when the protocol parses and passes all structural checks.
Exits 1 (printing the list of issues) when validation fails.

This is the small-tool half of Layer 3 step 3b. It lets a protocol
author catch typos and unsupported values before handing the YAML to
the (future) thin runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.protocol import (  # noqa: E402
    PROTOCOL_SCHEMA_VERSION,
    load_protocol,
    validate_protocol,
)


_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "protocol.yaml"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_PATH
    if not path.exists():
        print(f"[validate_protocol] ✗ file not found: {path}")
        return 1

    print(f"[validate_protocol] loading {path}")
    try:
        protocol = load_protocol(path)
    except Exception as e:
        print(f"[validate_protocol] ✗ parse error: {e}")
        return 1

    print(
        f"[validate_protocol] name={protocol.name!r} version={protocol.version!r} "
        f"schema_version={protocol.schema_version} (latest={PROTOCOL_SCHEMA_VERSION})"
    )
    print(f"[validate_protocol] response_mode={protocol.response_mode}")
    print(f"[validate_protocol] phases ({len(protocol.phases)}):")
    for i, ph in enumerate(protocol.phases):
        print(f"  {i}. {ph.name} → screen={ph.screen}, gate={ph.gate}({ph.value})")

    issues = validate_protocol(protocol)
    if issues:
        print("\n[validate_protocol] ✗ structural issues:")
        for it in issues:
            print(f"  - {it}")
        return 1

    print("\n[validate_protocol] ✓ valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
