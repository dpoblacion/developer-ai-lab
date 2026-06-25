"""Central watchdog + reaper for paid RunPod pods.

Pure decision functions (read_state/select_*/abort_reason) are unit-tested with no network
or threads. ``PodGuard`` drives them over a real pod lifecycle and is injected with a
clock / terminate_fn / progress samplers so its watchdog is testable deterministically.
"""

import json
import os


def read_state(path):
    """Return the list of tracked-pod entries, or [] if missing/corrupt."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []


def write_state(path, entries):
    """Write entries as JSON, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f)


def add_entry(path, entry):
    entries = read_state(path)
    entries.append(entry)
    write_state(path, entries)


def remove_entry(path, pod_id):
    entries = [e for e in read_state(path) if e.get("pod_id") != pod_id]
    write_state(path, entries)
