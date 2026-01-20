def calculate_trust(stats: dict) -> float:
    if not stats:
        return 0.5

    success_rate = stats.get("success_rate", 0)
    calls = stats.get("calls", 0)

    experience_bonus = min(calls / 10, 1)

    trust = (
        success_rate * 0.7 +
        experience_bonus * 0.3
    )

    return round(trust, 3)
