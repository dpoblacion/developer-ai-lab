"""Panic button: terminate RunPod pods (all, or older than REAP_AGE). `make reap`."""

import os
import time

from scripts.lib.dotenv import load_dotenv
from scripts.lib.pod_guard import REAP_AGE, DEFAULT_STATE_PATH, read_state, select_to_reap


def choose(pods, state, now, reap_age):
    created = {e["pod_id"]: e.get("created_at") for e in state}
    return select_to_reap([p["id"] for p in pods], created, now, reap_age)


def main():
    load_dotenv()
    import runpod
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    pods = runpod.get_pods()
    ids = choose(pods, read_state(DEFAULT_STATE_PATH), time.time(), REAP_AGE)
    for pid in ids:
        print(f"terminating {pid}")
        runpod.terminate_pod(pid)
    print(f"reaped {len(ids)} pod(s); {len(pods) - len(ids)} left")


if __name__ == "__main__":
    main()
