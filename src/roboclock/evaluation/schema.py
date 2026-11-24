"""Episode result schema shared by every domain, and normalization of recorded results.

A result row is one evaluated episode. ``evaluate()`` writes rows to
``episodes.jsonl``; recordings in ``data/`` keep a row under ``metrics`` or, for
some sports demos, in a domain-native layout that the domain's ``result_adapter``
converts. ``aggregate()`` only accepts rows that pass ``validate_row``.
"""

import math
from pathlib import Path

CATEGORIES = ("temporal_window", "event_reactive", "throughput")

# Task IDs number tasks by class (A = window, B = reactive, C = throughput) in every
# domain, so a class letter is an accepted category spelling as well.
CATEGORY_ALIASES = {
    "A": "temporal_window",
    "window": "temporal_window",
    "temporal_window": "temporal_window",
    "B": "event_reactive",
    "reactive": "event_reactive",
    "event_reactive": "event_reactive",
    "C": "throughput",
    "throughput": "throughput",
}

# Fields aggregate() reads; older rows without environment/seed remain summarizable.
SCORING_FIELDS = {
    "task_id": str,
    "difficulty": str,
    "category": str,
    "task_success": bool,
    "manipulation_success": bool,
    "temporal_success": bool,
    "joint_success": bool,
    "temporal_violations": list,
}
REQUIRED_FIELDS = {"environment": str, "seed": int, **SCORING_FIELDS}

# Optional numeric metrics averaged by aggregate(); None means not applicable.
NUMERIC_METRICS = (
    # Stage score 0-100 (roboclock.evaluation.score); absent in rows written before it existed.
    "score",
    "completion_time",
    "reaction_latency",
    "first_motion_latency",
    "temporal_error",
    "operations_per_minute",
    "accuracy",
    "drop_rate",
    "collision_count",
    "number_of_actions",
    "mean_position_error",
    "mean_orientation_error",
)


def canonical_category(value, task_id=None):
    """Map a category spelling, or failing that the task's class letter, to CATEGORIES."""
    if isinstance(value, str) and value in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[value]
    letter = str(task_id or "")[:1]
    if value is None and letter and letter in "ABC":
        return CATEGORY_ALIASES[letter]
    raise ValueError(f"Unknown temporal category {value!r} for task {task_id!r}")


def validate_row(row: dict, fields=REQUIRED_FIELDS) -> list[str]:
    """Return every schema problem in ``row``; an empty list means it is valid."""
    errors = []
    for key, kind in fields.items():
        if key not in row:
            errors.append(f"missing {key}")
        elif kind is int and (isinstance(row[key], bool) or not isinstance(row[key], int)):
            errors.append(f"{key} must be an integer, got {row[key]!r}")
        elif kind is not int and not isinstance(row[key], kind):
            errors.append(f"{key} must be {kind.__name__}, got {type(row[key]).__name__}")
    if isinstance(row.get("category"), str) and row["category"] not in CATEGORIES:
        errors.append(f"category must be one of {', '.join(CATEGORIES)}, got {row['category']!r}")
    for key in NUMERIC_METRICS:
        value = row.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            errors.append(f"{key} must be numeric or null, got {value!r}")
    if row.get("joint_success") and not (row.get("task_success") and row.get("temporal_success")):
        errors.append("joint_success requires task_success and temporal_success")
    tiers = row.get("joint_success_by_tier")
    if tiers is not None:
        temporal = row.get("temporal_success_by_tier")
        if not isinstance(tiers, dict) or not isinstance(temporal, dict):
            errors.append("joint_success_by_tier and temporal_success_by_tier must both be objects")
        else:
            for tier, joint in tiers.items():
                if joint and not (row.get("task_success") and temporal.get(tier)):
                    errors.append(
                        f"joint_success_by_tier[{tier}] requires task_success and "
                        f"temporal_success_by_tier[{tier}]"
                    )
            if row.get("difficulty") in tiers and tiers[row["difficulty"]] != row.get(
                "joint_success"
            ):
                errors.append("joint_success must equal joint_success_by_tier[difficulty]")
    return errors


def require_valid(rows, *, fields=REQUIRED_FIELDS, source="result rows"):
    problems = [
        f"{row.get('environment', '?')}/{row.get('task_id', '?')} (row {index}): {error}"
        for index, row in enumerate(rows)
        for error in validate_row(row, fields)
    ]
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise ValueError(f"{source} do not match the result schema: {shown}{more}")


def normalize_row(row: dict, environment: str | None = None) -> dict:
    """Canonical spelling of a row that already carries the core scoring fields."""
    row = dict(row)
    if environment is not None:
        row["environment"] = environment
    row["task_id"] = str(row["task_id"])
    row["category"] = canonical_category(row.get("category"), row["task_id"])
    for key in ("task_success", "manipulation_success", "temporal_success", "joint_success"):
        if key in row:
            row[key] = bool(row[key])
    if "manipulation_success" not in row and "task_success" in row:
        row["manipulation_success"] = row["task_success"]
    if isinstance(row.get("seed"), float) and row["seed"].is_integer():
        row["seed"] = int(row["seed"])
    for key in NUMERIC_METRICS:
        if isinstance(row.get(key), float) and not math.isfinite(row[key]):
            row[key] = None
    return row


def _environment_of(document, row, path, names):
    for value in (row.get("environment"), document.get("environment"), row.get("domain")):
        if isinstance(value, str) and value in names:
            return value
    if path is not None:
        # data/<environment>/... and results/<environment>/... name their domain.
        for part in reversed(Path(path).resolve().parts):
            if part in names:
                return part
    return None


def normalize_result(document: dict, *, environment=None, path=None) -> dict | None:
    """Convert one recorded result document into a schema-valid row.

    Accepted layouts: a result row (``episodes.jsonl`` line), a recording sidecar or
    result file holding the row under ``metrics``, or any layout that the domain's
    ``EnvironmentSpec.result_adapter`` converts. Adapters receive every document and
    return None for files that are not episode results (manifests, raw evidence).
    Returns None for such documents.
    """
    from roboclock.envs.registry import environment_names, get_environment, import_symbol

    metrics = document.get("metrics")
    row = metrics if isinstance(metrics, dict) else document
    environment = environment or _environment_of(document, row, path, environment_names())
    where = path or "result"
    try:
        if environment is not None and get_environment(environment).result_adapter:
            row = import_symbol(get_environment(environment).result_adapter)(document)
        elif not ("task_success" in row and "task_id" in row):
            return None
        if row is None:
            return None
        if environment is None:
            raise ValueError("cannot tell which environment it belongs to; pass --env")
        row = normalize_row(row, environment)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{where}: {exc!r}") from exc
    row.setdefault("source", str(path) if path is not None else None)
    return row


def collect_results(paths, *, environment=None):
    """Normalized rows from result files, JSONL episode files and directories.

    Named files must hold results. Directories are searched recursively; JSON files
    that are not results (manifests, configs, offline validations) are returned as
    skipped.
    """
    import json

    rows, skipped, seen = [], [], set()
    for path in map(Path, paths):
        named = not path.is_dir()
        files = (
            [path]
            if named
            else sorted(
                p for p in path.rglob("*") if p.suffix in (".json", ".jsonl") and p.is_file()
            )
        )
        for file in files:
            # Collections link the same sidecar from several folders.
            if file.resolve() in seen:
                continue
            seen.add(file.resolve())
            text = file.read_text(encoding="utf-8")
            if file.suffix == ".jsonl":
                documents = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                document = json.loads(text)
                documents = [document] if isinstance(document, dict) else []
            results = [normalize_result(d, environment=environment, path=file) for d in documents]
            results = [row for row in results if row is not None]
            if not results:
                if named:
                    raise ValueError(f"{file} contains no episode result")
                skipped.append(str(file))
            rows.extend(results)
    return rows, skipped
