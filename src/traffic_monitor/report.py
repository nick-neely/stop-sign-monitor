def print_final_report(car_stats, min_speed_sentinel=999.0):
    """Print a summary of tracked vehicles that entered the stop zone."""
    print("\n" + "=" * 40)
    print("FINAL TRAFFIC REPORT")
    print("=" * 40)

    clean_stops = 0
    violations = 0

    print(f"{'Car ID':<10} | {'Min Speed':<10} | {'Status'}")
    print("-" * 45)

    for car_id, data in car_stats.items():
        if data["min_speed"] == min_speed_sentinel:
            continue
        status = data["status"]
        if status == "Rolling...":
            status = "ROLLING STOP VIOLATION"
        print(f"{car_id:<10} | {data['min_speed']:.1f} mph   | {status}")
        if "Clean" in status:
            clean_stops += 1
        else:
            violations += 1

    print("-" * 45)
    print(f"Total Cars tracked in zone: {clean_stops + violations}")
    print(f"Clean Stops: {clean_stops}")
    print(f"Rolling Stops: {violations}")
    print("=" * 40)
