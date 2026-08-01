import json
from pathlib import Path
from typing import Any, Dict, List


def build_segment_metrics(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = data.get("segments", [])
    metrics = []
    previous_end = None
    previous_temperature = None

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        temperature = float(segment.get("temperature", 0.0))

        duration = end - start
        temperature_change = None
        if previous_temperature is not None:
            temperature_change = temperature - previous_temperature

        time_since_previous_end = None
        if previous_end is not None:
            time_since_previous_end = start - previous_end

        metrics.append(
            {
                "id": segment.get("id"),
                "start": start,
                "end": end,
                "duration": duration,
                "temperature": temperature,
                "temperature_change": temperature_change,
                "time_since_previous_end": time_since_previous_end,
            }
        )

        previous_end = end
        previous_temperature = temperature

    return metrics


def main() -> None:
    input_path = Path(__file__).with_name("transcripts") / "output.json"
    output_path = Path(__file__).with_name("segment_metrics.json")

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metrics = build_segment_metrics(data)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Wrote {len(metrics)} segment metrics to {output_path}")


if __name__ == "__main__":
    main()
