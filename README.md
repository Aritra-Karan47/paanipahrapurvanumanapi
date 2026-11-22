# School Water Demand Prediction Model & Inference Service

Production-oriented system that predicts **period-wise water consumption** for individual school water coolers, including the **Recess** interval after every fourth period (P4) and **Games** periods.

| Component | Description |
|-----------|-------------|
| **Inference service** | Gradio app deployable as a Hugging Face Space |
| **SDK** | Installable Python package `cooler-demand` (PyPI-ready) |
| **ML** | Historical-average baseline (primary) + optional CatBoost P50/P90 quantiles |
| **Temperature** | Deterministic outdoor → target water temperature mapping (not ML) |

> **Important:** The one-week simulated dataset shipped with this repository is **synthetic** and is used only for software testing and baseline demonstration. It is **not** treated as sufficient for reliable production CatBoost training. The service always prefers the robust baseline until multi-week real telemetry is available.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Main Institute Application (your system)                   │
│  • Institute config, PostgreSQL, sensors, weather API       │
│  • Timetable & consumption uploads, alerts, refill events   │
└──────────────────────────┬──────────────────────────────────┘
                           │  cooler-demand SDK
                           │  predict() / forecast_day() /
                           │  get_temperature_target() /
                           │  check_shortage() / validate_data()
┌──────────────────────────▼──────────────────────────────────┐
│  Hugging Face Space (Gradio)                                │
│  • Stateless JSON API under /api/*                          │
│  • Baseline (historical average + engineered defaults)      │
│  • Optional CatBoost quantile models (when trained)         │
│  • Deterministic temperature-control table                  │
└─────────────────────────────────────────────────────────────┘
```

The model service **never** stores institute secrets, long-term history, or credentials. All institute-specific context is passed per request.

### Feature engineering

- Day of week, period slot index  
- Slot type flags: `is_recess`, `is_games`, `is_period`  
- Games intensity (None / Light / Moderate / High)  
- Cooler location attributes (capacity, floor, near_games)  
- Outdoor temperature + weather code + temperature band  
- Historical mean / std / count for the same cooler×day×slot (when provided)  
- Interaction terms (`games_x_near`, `temp_x_recess`)

### Prediction strategy

1. **Baseline (always available)**  
   Hierarchical historical average:  
   `cooler+day+slot` → `cooler+slot` → `global slot` → engineered default (capacity × slot multiplier × mild temperature factor).  
   Conservative estimate = expected × 1.25.

2. **CatBoost (optional upgrade)**  
   Independent Quantile regressors for P50 and P90.  
   Training is **refused** when fewer than 200 rows are available (configurable).  
   One-week synthetic data is deliberately treated as insufficient for production use.

### Temperature control (deterministic)

Parsed from `Water_Temperature_Control_Table.xlsx`:

| Outdoor range (°C) | Target water (°C) |
|--------------------|-------------------|
| [-10, 10)          | 18                |
| [10, 18)           | 16                |
| [18, 25)           | 14                |
| [25, 32)           | 12                |
| [32, 55]           | 10                |

---


## Institute upload formats (supported)

| File | Key columns |
|------|-------------|
| `Cooler_Water_Consumption_Record.csv` | `Day`, `Period_Number` (1–8 or Recess), `Cooler_Number` (e.g. Location01A), `Consumption_Value` |
| `School_Timetable_With_Recess.xlsx` (sheet Timetable) | `Class_section`, `day` (Mon…Sat), `period` (1–8 or Recess), `subject` (Games appears as subject) |
| `Water_Temperature_Control_Table.xlsx` (sheet Rules) | `ambient_min_c (>=)`, `ambient_max_c (<)`, `cooling_enabled`, `setpoint_*`, `target_c` (IS 1475:2024 aligned) |

Weather is supplied by the calling application (any weather API); the model service accepts `outdoor_temp_c` and a coarse `weather` label per request.


## Repository layout

```
school-water-demand/
├── app.py                 # HF Space entrypoint
├── requirements.txt
├── pyproject.toml         # cooler-demand package
├── data/                  # Synthetic demo data (NOT real training data)
│   ├── Cooler_Water_Consumption_Record.xlsx
│   ├── School_Timetable_Long_Format.xlsx
│   ├── Water_Temperature_Control_Table.xlsx
│   └── cooler_config.csv
├── fixtures/              # CSV mirrors for unit tests
├── src/
│   ├── cooler_demand/     # Installable SDK
│   │   ├── client.py
│   │   └── schemas.py
│   ├── model/
│   │   ├── baseline.py
│   │   ├── catboost_model.py
│   │   ├── features.py
│   │   ├── predictor.py
│   │   └── preprocessing.py
│   ├── temp_control/
│   │   └── controller.py
│   └── app/
│       └── app.py         # Gradio + /api/* routes
├── tests/
└── scripts/
```

---

## Quick start (local)

```bash
# Clone / unpack and enter the repo
cd school-water-demand

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # installs cooler-demand SDK

# Run unit + integration tests
pytest tests/ -v

# Launch Gradio locally
python -m src.app.app
# → http://localhost:7860
```

### Example SDK usage (against a running Space or local Gradio)

```python
from cooler_demand import CoolerDemandClient, PredictionRequest, CoolerInfo, IntervalContext, WeatherContext

client = CoolerDemandClient(base_url="http://localhost:7860")

req = {
    "cooler": {
        "cooler_id": "C04",
        "capacity_liters": 250,
        "near_games": True,
        "current_level_liters": 160,
    },
    "intervals": [
        {"period_slot": "P1", "day_of_week": "Tuesday"},
        {"period_slot": "P2", "day_of_week": "Tuesday"},
        {"period_slot": "P3", "day_of_week": "Tuesday"},
        {"period_slot": "P4", "day_of_week": "Tuesday"},
        {"period_slot": "Recess", "day_of_week": "Tuesday", "duration_minutes": 20},
        {"period_slot": "P5", "day_of_week": "Tuesday"},
        {"period_slot": "Games", "day_of_week": "Tuesday", "games_intensity": "High"},
    ],
    "weather": {"outdoor_temp_c": 31.0, "weather": "Humid"},
    "model_preference": "auto",
}

# Single-shot prediction
resp = client.predict(req)
print(resp.total_expected_liters, resp.total_conservative_liters)

# Full-day forecast + shortage
day_resp = client.forecast_day({**req, "day_of_week": "Tuesday", "current_level_liters": 160})
print(day_resp.shortage)

# Deterministic temperature target
t = client.get_temperature_target(31.0)
print(t.target_water_temp_c)   # 12.0

# Shortage check only
s = client.check_shortage({
    "cooler_id": "C04",
    "capacity_liters": 250,
    "current_level_liters": 40,
    "expected_demand_liters": 90,
    "conservative_demand_liters": 115,
})
print(s.risk_level, s.message)

client.close()
```

---

## Hugging Face Space deployment

1. Create a new Space (Gradio / Python).
2. Push this repository (or upload the zip contents).
3. Ensure `requirements.txt` and `app.py` are at the Space root.
4. The Space will run `app.py`, which launches the Gradio Blocks app and attaches `/api/*` routes.
5. Point the SDK at the Space URL:

```python
client = CoolerDemandClient(
    base_url="https://<user>-school-water-demand.hf.space",
    api_token="hf_..."   # only if the Space is gated
)
```

---

## Publishing the SDK to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
# package name: cooler-demand
```

Install elsewhere:

```bash
pip install cooler-demand
```

---

## Input / output schemas (summary)

### PredictionRequest (core fields)

```json
{
  "cooler": {
    "cooler_id": "C01",
    "capacity_liters": 200,
    "near_games": false,
    "current_level_liters": 140
  },
  "intervals": [
    {"period_slot": "Recess", "day_of_week": "Monday", "duration_minutes": 20}
  ],
  "weather": {"outdoor_temp_c": 28.0, "weather": "Clear"},
  "model_preference": "auto"
}
```

### PredictionResponse (excerpt)

```json
{
  "cooler_id": "C01",
  "predictions": [
    {
      "period_slot": "Recess",
      "expected_liters": 14.2,
      "p50_liters": 14.2,
      "p90_liters": 17.75,
      "conservative_liters": 17.75,
      "model_used": "baseline:cooler_slot",
      "confidence": "medium"
    }
  ],
  "total_expected_liters": 14.2,
  "total_conservative_liters": 17.75,
  "model_version": "water-demand-v0.1.0-baseline",
  "fallback_used": true
}
```

---

## Testing

```bash
pytest tests/ -v --tb=short
```

Coverage includes:

- Excel/CSV ingestion and column validation  
- Timetable parsing and Recess ordering  
- Games intensity features  
- All eight cooler configurations  
- Temperature boundary and extrapolation rules  
- Malformed / missing inputs and prediction fallbacks  
- Quantile output shape (when CatBoost is force-trained in unit tests)  
- Shortage risk levels and manual-refill-aware messaging  
- Schema validation and SDK client construction  
- End-to-end in-process service calls  

Synthetic fixtures under `data/` and `fixtures/` are **explicitly for software testing only**.

---

## Limitations & future real-data retraining

1. **Data volume** – One week of synthetic data cannot capture seasonal patterns, exam weeks, special events, or cooler-specific drift. Collect at least several weeks of real period-level telemetry before enabling CatBoost in production.
2. **No automatic retraining** – Retrain offline, version the models, and deploy new artifacts to the Space. Update `MODEL_VERSION` accordingly.
3. **Weather** – The service accepts outdoor temperature and a coarse weather label; the main application is responsible for calling a weather API.
4. **Sensor reconciliation** – Flow/level sensor fusion and manual refill event logging belong to the main application; the model only consumes the current level and predicted demand you supply.
5. **Multi-school** – The service is deliberately institute-agnostic. Pass cooler metadata and history on every request (or maintain a thin cache in the main app).

### Recommended retraining workflow

```python
from model.preprocessing import load_consumption_record, load_cooler_config
from model.predictor import WaterDemandPredictor

meta = load_cooler_config("path/to/cooler_config.csv")
df = load_consumption_record("path/to/real_multiweek_consumption.xlsx")

pred = WaterDemandPredictor(model_dir="artifacts/models")
pred.load_cooler_meta(meta)
info = pred.fit_from_consumption(df, train_catboost=True)  # only when rows >> 200
print(info)
# Deploy the saved .cbm files with the Space
```

---

## License

MIT
