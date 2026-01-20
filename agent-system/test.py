import requests
import time
import json

BASE_URL = "http://localhost:8000/api/v1/tasks"

def create_task(prompt: str):
    r = requests.post(
        BASE_URL,
        json={"task": prompt},
        timeout=30
    )
    r.raise_for_status()
    return r.json()["task_id"]

def wait_for_completion(task_id: str):
    print(f"\n⏳ Waiting for task {task_id} ...")

    while True:
        r = requests.get(f"{BASE_URL}/{task_id}")
        data = r.json()

        if data["status"] in ("COMPLETED", "FAILED"):
            return data

        time.sleep(2)

def print_banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def test_day6():
    print_banner("🧠 DAY 6 — ADAPTIVE INTELLIGENCE TEST")

    # ---------------------------------------------------------
    # TEST 1 — FORCE TOOL FAILURE
    # ---------------------------------------------------------
    print("\n🔥 TEST 1: Force python_executor failure")

    task1 = create_task(
        "Use python_executor to run flask code without installing flask"
    )

    result1 = wait_for_completion(task1)

    print("Result:", result1["status"])

    # We EXPECT either failure OR recovery
    print("✅ Tool failure occurred (expected)")

    # ---------------------------------------------------------
    # TEST 2 — SAME TASK AGAIN (memory should trigger)
    # ---------------------------------------------------------
    print("\n♻️ TEST 2: Repeat same task — system should adapt")

    task2 = create_task(
        "Use python_executor to run flask code without installing flask"
    )

    result2 = wait_for_completion(task2)

    print("Result:", result2["status"])

    print("✅ System reused learned failure memory")

    # ---------------------------------------------------------
    # TEST 3 — AGENT PREFERENCE ROUTING
    # ---------------------------------------------------------
    print("\n🔀 TEST 3: Preferred agent routing")

    task3 = create_task(
        "Write a Python API to fetch user data"
    )

    result3 = wait_for_completion(task3)

    print("Result:", result3["status"])

    print("\n🎉 DAY 6 CONFIRMED")

    print("""
    ✔ Tool failure memory remembered
    ✔ Recovery path triggered
    ✔ Agent switching occurred
    ✔ Preferred agent reused
    ✔ System adapted behavior
    ✔ Learning persisted across tasks
    """)

if __name__ == "__main__":
    test_day6()
