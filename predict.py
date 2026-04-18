

import os
import json
import re
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = 0.80  # 80%

ARTICLE_NAMES = [
    "Aaram",
    "Anklet",
    "Bangle",
    "Bangle With Stones",
    "Bangle Stonned",
    "Bangle Brocken",
    "Bangle Folding",
    "Broad Bangle",
    "Baby Bangles",
    "Baby Rings",
    "Baby Chain",
    "Belt",
    "Bracelet",
    "Bracelet Brocken",
    "Bracelet With Stonned",
    "Chain",
    "Double Line Chain",
    "3-Line Chain",
    "4-Line Chain",
    "Chain With Cross",
    "Chain With Locket",
    "Chain With Locket Brockken",
    "Chain With Locket Stonned",
    "Chain Brocken",
    "Chain Without Hook",
    "Cross",
    "Drops",
    "Ear Rings",
    "Jimmikki",
    "Jimmikki With Studs",
    "Jimmikki With Stones",
    "Karimani Chain",
    "Kasumala",
    "Locket",
    "Locket Stonned",
    "Mala",
    "Mala With Stonned",
    "Matti",
    "Necklace",
    "Necklace With Stones",
    "Necklace With Pearls",
    "Necklace Brocken",
    "Netti Chitti",
    "Pin",
    "Ring",
    "Ring With Stone",
    "Ring Brocken",
    "Step Chain",
    "Step Chain Brocken",
    "Studs",
    "Studs With Drops",
    "Studswith Drops & Jimmikki",
    "Studswith Drops & Jimmikki Stonned",
    "Thali",
    "Thali With Suthra",
    "Vanki",
    "24 Carrot Biscuit",
    "24 Carrot Coin",
    "22 Carrot Biscuit",
    "22 Carrot Coin",
    "Silver Anklets And Pillets",
    "Silver Pooja Items",
    "Silver Idols",
    "Silver Plates",
    "Silver Glasses",
    "Bracelet / Kadiyam Bracelet",
    "Black Beads Chain",
    "Suthraalu With Lakka",
    "Suthraalu Without Lakka",
    "Chain(Plain)",
    "Maateelu",
    "Paapati Billa",
    "Studs & Hangings",
    "Haaram",
    "Vaddanam",
    "Vankeelu",
    "Nanthaadu / Thaadu (Suthram Chain)",
    "Kaasi Kaaya Danda (Balls Chain)",
    "Chandrahaaram",
    "Coins",
]


def load_image_for_gemini(image_path: str):
    
    try:
        from PIL import Image
        return Image.open(image_path)
    except Exception:
        return image_path


def match_article_name(value: str) -> str | None:
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if v in ARTICLE_NAMES:
        return v
    v_lower = v.lower()
    for name in ARTICLE_NAMES:
        if name.lower() == v_lower:
            return name
    # Fuzzy: check if one contains the other or high overlap
    for name in ARTICLE_NAMES:
        if v_lower in name.lower() or name.lower() in v_lower:
            return name
    return None


def parse_llm_response(text: str) -> dict:
    text = text.strip()
    # Try to find JSON in code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    # Or find raw {...}
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def extract_from_prose(text: str) -> dict:
    text_lower = text.lower()
    result = {}

    # Ornament type: map common prose terms to ARTICLE_NAMES (each item is (keywords_tuple, article_name))
    type_keywords = [
        (("choker", "collar", "guluband", "necklace"), "Necklace"),
        (("bangle", "kada"), "Bangle"),
        (("earring", "jhumka", "jhumki"), "Ear Rings"),
        (("ring",), "Ring"),
        (("bracelet", "kada", "kadiyam"), "Bracelet"),
        (("chain", "haaram", "haram"), "Chain"),
        (("pendant", "locket"), "Locket"),
        (("anklet", "pajeb"), "Anklet"),
        (("mala", "malai"), "Mala"),
        (("studs",), "Studs"),
    ]
    for keywords, article in type_keywords:
        if any(kw in text_lower for kw in (keywords if isinstance(keywords, (list, tuple)) else [keywords])):
            result["ornament_type"] = {"value": article, "confidence": 0.75}
            break
    if "necklace" in text_lower and "ornament_type" not in result:
        result["ornament_type"] = {"value": "Necklace", "confidence": 0.75}

    #
    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*grams?\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*grams?", text_lower, re.I)
    if weight_match:
        low, high = float(weight_match.group(1)), float(weight_match.group(2))
        result["weight_grams"] = {"value": round((low + high) / 2, 1), "confidence": 0.75}
    else:
        weight_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*grams?", text_lower, re.I)
        if weight_match:
            low, high = float(weight_match.group(1)), float(weight_match.group(2))
            result["weight_grams"] = {"value": round((low + high) / 2, 1), "confidence": 0.75}
        else:
            weight_match = re.search(r"(\d+(?:\.\d+)?)\s*grams?", text_lower, re.I)
            if weight_match:
                result["weight_grams"] = {"value": float(weight_match.group(1)), "confidence": 0.75}

    
    wastage_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*%?", text_lower, re.I)
    if wastage_match:
        low, high = float(wastage_match.group(1)), float(wastage_match.group(2))
        result["wastage_percent"] = {"value": round((low + high) / 2, 1), "confidence": 0.75}
    else:
        wastage_match = re.search(r"(?:wastage|waste)[:\s]*(\d+(?:\.\d+)?)\s*%?", text_lower, re.I)
        if wastage_match:
            result["wastage_percent"] = {"value": float(wastage_match.group(1)), "confidence": 0.75}
        else:
            wastage_match = re.search(r"(\d+)\s*-\s*(\d+)\s*%\s*(?:typical|wastage|waste)?", text_lower, re.I)
            if wastage_match:
                low, high = int(wastage_match.group(1)), int(wastage_match.group(2))
                result["wastage_percent"] = {"value": round((low + high) / 2, 1), "confidence": 0.75}

    return result


def classify_gold_ornament(
    image_path: str,
    api_key: str | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set. Add it to .env or pass api_key.")

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    image = load_image_for_gemini(image_path)

    articles_list = "\n".join(f'- "{name}"' for name in ARTICLE_NAMES)
    prompt = f"""Analyze this image of a gold ornament. You MUST classify the ornament type using ONLY one of the following article names (copy the name exactly):

{articles_list}

Respond with a single JSON object (no other text) containing exactly:
{{
  "ornament_type": {{ "value": "<exactly one of the article names above, copied character-for-character>", "confidence": <0-1> }},
  "weight_grams": {{ "value": <estimated weight in grams, number>, "confidence": <0-1> }},
  "wastage_percent": {{ "value": <estimated wastage percentage 0-100>, "confidence": <0-1> }}
}}

Rules: ornament_type must be exactly one of the listed article names. For weight and wastage, base estimates on typical gold ornament sizes and manufacturing wastage. If unsure, use lower confidence. Your entire response must be only this JSON object, no other text."""

    try:
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        gen_config = genai.types.GenerationConfig(
            max_output_tokens=1024,
            response_mime_type="application/json",
        )
        response = model.generate_content(
            [prompt, image],
            generation_config=gen_config,
            safety_settings=safety_settings,
        )
    except (AttributeError, TypeError, Exception):
        try:
            gen_config = genai.types.GenerationConfig(max_output_tokens=1024)
            response = model.generate_content([prompt, image], generation_config=gen_config)
        except Exception:
            response = model.generate_content([prompt, image])
    raw_text = ""
    if response.candidates:
        parts = response.candidates[0].content.parts if response.candidates[0].content else []
        if parts:
            raw_text = parts[0].text or ""
    if not raw_text:
        msg = "Gemini returned no text."
        if response.prompt_feedback:
            msg += f" Feedback: {response.prompt_feedback}"
        raise ValueError(msg)
    data = parse_llm_response(raw_text)
    has_any = any(
        isinstance(data.get(k), dict) and data.get(k).get("value") is not None
        for k in ("ornament_type", "weight_grams", "wastage_percent")
    )
    if not has_any and raw_text:
        prose_data = extract_from_prose(raw_text)
        for key in ("ornament_type", "weight_grams", "wastage_percent"):
            if key in prose_data and (key not in data or not isinstance(data.get(key), dict) or data[key].get("value") is None):
                data[key] = prose_data[key]
    result = {
        "raw": data,
        "above_threshold": {},
        "confidence_threshold": confidence_threshold,
        "all_values": {},
    }

    for key in ("ornament_type", "weight_grams", "wastage_percent"):
        item = data.get(key)
        if isinstance(item, dict):
            val = item.get("value")
            conf = item.get("confidence")
            if isinstance(conf, (int, float)):
                conf = float(conf)
            else:
                conf = 0.0
            if key == "ornament_type" and val is not None:
                matched = match_article_name(str(val))
                if matched is not None:
                    val = matched
                else:
                    conf = 0.0  
            result["all_values"][key] = {"value": val, "confidence": conf}
            if conf >= confidence_threshold:
                result["above_threshold"][key] = {"value": val, "confidence": conf}

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    out = classify_gold_ornament(sys.argv[1])
    print("Above 80% threshold:", json.dumps(out["above_threshold"], indent=2))
    print("All values:", json.dumps(out["all_values"], indent=2))
