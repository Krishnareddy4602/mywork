# Gold Ornament Classifier

Classify gold ornament images using **Google Gemini API**. Get **ornament type**, **estimated weight (grams)**, and **wastage (%)** with confidence scores. The UI shows only predictions with **confidence ≥ 80%** (configurable).

## Setup

**Use a virtual environment** so this project’s dependencies (e.g. `google-generativeai`, `protobuf`) don’t conflict with other packages like `tensorflow-intel` or `yolov7detect`.

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies (only inside this venv):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API keys** in `.env`:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   If you only want Gemini UI/classification, `GEMINI_API_KEY` is enough.

4. **(Optional) Gemini-only key setup** (alternative):
   - **Option A:** Create a `.env` file in the project root:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
   - **Option B:** Enter the API key in the app sidebar when you run the UI.

## Run

Activate the venv first (`venv\Scripts\activate`), then:

**Web UI (recommended):**
```bash
streamlit run app.py
```

**Command-line prediction:**
```bash
python predict.py path/to/ornament_image.jpg
```

**Gemini vs OpenAI side-by-side comparison (same image, same schema):**
```bash
python compare_models.py path/to/ornament_image.jpg --openai-model gpt-4.1-mini
```

### If you see dependency conflicts (tensorflow / yolov7 / protobuf)

Install and run this project **inside its own virtual environment** (steps above). That keeps `google-generativeai` and its `protobuf`/`numpy` versions separate from TensorFlow or YOLOv7. Use the global (or another venv) only for those other projects.

## Features

- **Ornament type:** e.g. necklace, ring, bangle, pendant, earring, bracelet, chain
- **Estimated weight:** in grams (visual estimate)
- **Wastage:** estimated wastage percentage
- **Confidence threshold:** only values with confidence ≥ 80% are shown in the main result (adjustable in sidebar)
- Expandable section to view all predictions and confidence scores

## Security

Do not commit your `.env` file or share your API key. The repo includes `.env` in `.gitignore`.

## API

This app uses **Google Gemini** (gemini-2.5-flash) for image analysis. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
