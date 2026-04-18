import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from predict import ARTICLE_NAMES, CONFIDENCE_THRESHOLD, classify_gold_ornament, parse_llm_response

load_dotenv()


def _image_to_data_url(image_path: str) -> str:
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_file))
    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def classify_with_openai(
    image_path: str,
    model: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Add it to .env or set it in your shell.")

    client = OpenAI(api_key=key)
    articles_list = "\n".join(f'- "{name}"' for name in ARTICLE_NAMES)
    prompt = f"""Analyze this image of a gold ornament. You MUST classify the ornament type using ONLY one of the following article names (copy the name exactly):

{articles_list}

Respond with a single JSON object (no other text) containing exactly:
{{
  "ornament_type": {{ "value": "<exactly one of the article names above, copied character-for-character>", "confidence": <0-1> }},
  "weight_grams": {{ "value": <estimated weight in grams, number>, "confidence": <0-1> }},
  "wastage_percent": {{ "value": <estimated wastage percentage 0-100>, "confidence": <0-1> }}
}}
"""

    image_data_url = _image_to_data_url(image_path)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )

    raw_text = response.output_text or ""
    if not raw_text:
        raise ValueError("OpenAI returned no text.")

    data = parse_llm_response(raw_text)
    result = {
        "raw": data,
        "above_threshold": {},
        "confidence_threshold": confidence_threshold,
        "all_values": {},
    }

    for key_name in ("ornament_type", "weight_grams", "wastage_percent"):
        item = data.get(key_name)
        if isinstance(item, dict):
            val = item.get("value")
            conf = item.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else 0.0
            result["all_values"][key_name] = {"value": val, "confidence": conf}
            if conf >= confidence_threshold:
                result["above_threshold"][key_name] = {"value": val, "confidence": conf}

    return result


def _print_side_by_side(gemini_result: dict, openai_result: dict, gemini_ms: int, openai_ms: int) -> None:
    print("\n=== Side-by-Side Result ===")
    print(f"Gemini latency: {gemini_ms} ms")
    print(f"OpenAI latency: {openai_ms} ms\n")

    print("Gemini (all values):")
    print(json.dumps(gemini_result.get("all_values", {}), indent=2))
    print("\nOpenAI (all values):")
    print(json.dumps(openai_result.get("all_values", {}), indent=2))

    print("\nGemini (above threshold):")
    print(json.dumps(gemini_result.get("above_threshold", {}), indent=2))
    print("\nOpenAI (above threshold):")
    print(json.dumps(openai_result.get("above_threshold", {}), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gemini and OpenAI ornament predictions side-by-side.")
    parser.add_argument("image_path", help="Path to ornament image")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model name (for display only)")
    parser.add_argument("--openai-model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold (0-1)")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not set. Add it to .env or set it in your shell.")

    start = time.perf_counter()
    gemini_result = classify_gold_ornament(
        args.image_path,
        api_key=gemini_key,
        confidence_threshold=args.threshold,
    )
    gemini_ms = int((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    openai_result = classify_with_openai(
        args.image_path,
        model=args.openai_model,
        confidence_threshold=args.threshold,
    )
    openai_ms = int((time.perf_counter() - start) * 1000)

    print(f"Gemini model: {args.gemini_model}")
    print(f"OpenAI model: {args.openai_model}")
    _print_side_by_side(gemini_result, openai_result, gemini_ms, openai_ms)


if __name__ == "__main__":
    main()
