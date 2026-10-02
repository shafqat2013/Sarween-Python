"""Human-readable movement logs and independent, editable review labels."""

import copy
import csv
import io
import json
import uuid
from pathlib import Path

from app_paths import atomic_write_json, atomic_write_bytes
from tracking_evaluation import normalize_cell, parse_time_seconds, format_time, validate_case


def timeline_rows(data, track="detected"):
    fps = max(1.0, float(data.get("fps", 30)))
    entries = data.get("deliveryEvents" if track == "foundry" else "trackingEvents", [])
    previous = {}
    rows = []
    for index, event in enumerate(entries):
        mini = event.get("mini", event.get("miniId"))
        cell = normalize_cell(event.get("cell", event.get("to_cell")))
        if not mini or not cell:
            continue
        seconds = max(0, int(event.get("frame", 0))) / fps
        row = {"id": f"{track}-{index}", "mini": str(mini), "time_seconds": seconds,
               "from_cell": previous.get(mini), "to_cell": cell, "track": track,
               "source": event.get("source", "detection"), "decision": "unreviewed"}
        if track == "foundry":
            row["decision"] = "confirmed" if event.get("type") == "tokenMoveApplied" else "failed"
            row["source"] = "acknowledgement"
        else:
            row["decision"] = "detected"
        rows.append(row)
        if row["decision"] != "failed":
            previous[mini] = cell
    return rows


def detected_rows(events):
    return [{**copy.deepcopy(event), "id": f"detected-{index}", "track": "detected", "decision": "detected"}
            for index, event in enumerate(events)]


def export_rows(rows, base):
    base = Path(base)
    rows = list(rows)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("Timestamp", "Seconds", "Mini", "From", "To", "Track", "Status", "Cause"))
    lines = ["Sarween movement transcript", "Timestamps are positions in the encoded video.",
             "Detected movements are predictions, not an exhaustive record of physical movement.", ""]
    def safe(value):
        value = str(value or "")
        return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value
    for row in sorted(rows, key=lambda item: item["time_seconds"]):
        fields = (format_time(row["time_seconds"]), f"{row['time_seconds']:.3f}", row["mini"],
                  row.get("from_cell") or "?", row["to_cell"], row.get("track", "detected"),
                  row.get("decision", "detected"), row.get("source", "detection"))
        writer.writerow([safe(value) for value in fields])
        lines.append(f"{fields[0]}  {fields[2]}  {fields[3]} -> {fields[4]}  [{fields[5]} / {fields[6]} / {fields[7]}]")
    atomic_write_bytes(str(base) + ".csv", stream.getvalue().encode(), backup=False)
    atomic_write_bytes(str(base) + ".txt", ("\n".join(lines) + "\n").encode(), backup=False)
    atomic_write_json(str(base) + ".json", {"schemaVersion": 1, "movements": rows}, backup=False)


def export_recording(timeline_path):
    path = Path(timeline_path)
    data = json.loads(path.read_text())
    base = path.with_name(path.name.removesuffix(".tracking.json") + ".movements")
    export_rows(timeline_rows(data) + timeline_rows(data, "foundry"), base)
    return base


def validate_label(row, *, duration=None, grid=None):
    result = copy.deepcopy(row)
    mini = str(result.get("mini", "")).strip()
    if not mini or len(mini) > 100:
        raise ValueError("A mini name is required (100 characters maximum)")
    result.update(mini=mini, time_seconds=parse_time_seconds(result.get("time_seconds")),
                  from_cell=normalize_cell(result.get("from_cell")), to_cell=normalize_cell(result.get("to_cell")))
    if result["to_cell"] is None:
        raise ValueError("A destination cell is required")
    if duration is not None and result["time_seconds"] > duration:
        raise ValueError("Movement time is outside this video")
    if grid:
        from tracking_evaluation import a1_to_row_col
        for cell in (result["from_cell"], result["to_cell"]):
            if cell:
                r, c = a1_to_row_col(cell)
                if c >= grid[0] or r >= grid[1]:
                    raise ValueError(f"{cell} is outside the selected grid")
    result.setdefault("id", uuid.uuid4().hex)
    result.update(track="reviewed", decision="confirmed")
    result.setdefault("source", "detection")
    result.setdefault("provenance", "manual-video-review")
    return result


def review_case(video, labels, *, duration, options, complete=False):
    if not complete:
        raise ValueError("Mark the video review complete before creating a regression case")
    confirmed = sorted([validate_label(row, duration=duration) for row in labels], key=lambda row: row["time_seconds"])
    case = {"name": Path(video).stem + "-reviewed", "video": str(Path(video).resolve()),
            "label_source": "user-reviewed-video", "allow_unexpected": False,
            "ignore_after": duration, "tolerance_seconds": 1.0,
            "expectations": [{"mini": row["mini"], "from": row.get("from_cell"),
                              "to": row["to_cell"], "at": row["time_seconds"],
                              "source": row.get("source", "detection")} for row in confirmed]}
    if any(row.get("provenance") == "confirmed-prediction" for row in confirmed):
        case["label_source"] = "user-reviewed-predictions"
    if not confirmed:
        case["stationary"] = True
    for source, target in (("profiles_path", "profiles"), ("timeline_path", "timeline"),
                           ("grid_w", "grid_cols"), ("grid_h", "grid_rows"),
                           ("warp_w", "warp_w"), ("warp_h", "warp_h"), ("marker_mode", "marker_mode")):
        if options.get(source) is not None:
            case[target] = str(options[source]) if isinstance(options[source], Path) else options[source]
    case["restore_recorded_profiles"] = options.get("restore_recorded_profiles", True)
    validate_case(case)
    return {"cases": [case]}
