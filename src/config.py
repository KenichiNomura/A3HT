#!/usr/bin/env python3
"""Dependency-free config loader for A3HT. Reads config.toml next to this file."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent  # repo root (parent of src/)
_CONFIG_PATH = ROOT / "config.toml"
_cache: Optional[Dict[str, Any]] = None


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        items = [_parse_value(x) for x in raw[1:-1].split(",") if x.strip()]
        return items
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_toml(text: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    section: Dict[str, Any] = cfg
    in_array = False
    array_key = ""
    array_buf = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # multi-line array continuation
        if in_array:
            array_buf += line
            if "]" in line:
                section[array_key] = _parse_value(array_buf)
                in_array = False
            continue

        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            cfg.setdefault(name, {})
            section = cfg[name]
            continue

        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # strip inline comments
            if not val.startswith('"') and "#" in val:
                val = val[:val.index("#")].strip()
            if val.startswith("[") and "]" not in val:
                in_array = True
                array_key = key
                array_buf = val
            else:
                section[key] = _parse_value(val)

    return cfg


def load(path: Optional[Path] = None) -> Dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    p = Path(path) if path else _CONFIG_PATH
    _cache = _parse_toml(p.read_text(encoding="utf-8"))
    return _cache


def get(key_path: str, default: Any = None) -> Any:
    cfg = load()
    parts = key_path.split(".")
    node: Any = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def shell_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")


def _resolve(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    return str(p if p.is_absolute() else ROOT / p)


def _shell_env() -> List[str]:
    cfg = load()
    c = cfg.get("campaign", {})
    campaign_name = c.get("name", "default")
    runs_root = str(ROOT / "campaigns" / campaign_name / "my_runs")
    state_dir = str(ROOT / "campaigns" / campaign_name / ".queue_state")
    lammps_dir = _resolve(cfg.get("paths", {}).get("lammps_dir", ""))
    python3 = cfg.get("paths", {}).get("python", "python3")
    s = cfg.get("structure", {})
    q = cfg.get("queue", {})
    alcf = cfg.get("alcf", {})

    lines = [
        f'A3HT_RUNS_ROOT="{shell_escape(runs_root)}"',
        f'A3HT_STATE_DIR="{shell_escape(state_dir)}"',
        f'A3HT_STRUCTURE_BASE_ANGLE_DEG="{shell_escape(s.get("base_angle_deg", 90.0))}"',
        f'A3HT_STRUCTURE_ANGLE_DISTURB_DEG="{shell_escape(s.get("angle_disturb_deg", 30.0))}"',
        f'A3HT_STRUCTURE_TILT_MAX_DEG="{shell_escape(s.get("tilt_max_deg", 30.0))}"',
        f'A3HT_ALCF_MODEL="{shell_escape(alcf.get("model", ""))}"',
        f'A3HT_TARGET_JOBS="{shell_escape(q.get("target_jobs", 10))}"',
        f'A3HT_JOB_NAME="{shell_escape(q.get("job_name", "a3ht"))}"',
        f'A3HT_INITIAL_SEED="{shell_escape(c.get("initial_seed", 1000))}"',
        f'LAMMPS_DIR="{shell_escape(lammps_dir)}"',
        f'A3HT_PYTHON3="{shell_escape(python3)}"',
    ]
    return lines


def _runtime_lib_path() -> str:
    cfg = load()
    dirs = cfg.get("runtime_libs", {}).get("dirs", [])
    resolved = [_resolve(d) for d in dirs]
    existing_ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    if existing_ldpath:
        resolved.append(existing_ldpath)
    return ":".join(resolved)


if __name__ == "__main__":
    if "--shell-env" in sys.argv:
        print("\n".join(_shell_env()))
    elif "--ld-library-path" in sys.argv:
        print(_runtime_lib_path())
    elif len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        val = get(sys.argv[1])
        if val is None:
            sys.exit(f"key not found: {sys.argv[1]}")
        print(val)
    else:
        import json
        print(json.dumps(load(), indent=2))
