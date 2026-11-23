### The Big Picture

Think of the system like a smart school cafeteria manager: **it looks at what students drank on the same day and period in past weeks, adjusts for how hot it is outside and whether students just played games, and predicts two numbers: what they will likely drink (Expected) and the maximum they could drink (Conservative safety buffer) so the cooler never runs dry.**

---

### 1. What Models Are Used?

We use a **Two-Tier System** (a reliable rule-based memory + a smart AI model):

```
                  ┌──────────────────────────────────────────────┐
                  │           How thirsty will kids be?          │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [ Model 1: The Memory ]                         [ Model 2: The AI ]
 Hierarchical Historical Baseline              CatBoost Quantile Regressor
   "What happened in past weeks?"              "Learns complex subtle patterns"
   (Works 100% of the time,                    (Learns interactions like:
    never fails, zero guesswork)                Hot Day + Recess + Near Field)
```

---

#### Model 1: The "Smart Memory" (Hierarchical Baseline)
This is the **primary engine**. Schools run on strict routines: every Tuesday during Recess, Cooler `Location01A` will see almost the exact same crowd. 

Instead of guessing, Model 1 uses a **4-step search**:
1. **Exact Match**: *"What did `Location01A` consume on Tuesdays during Recess over the last 5 weeks?"* $\rightarrow$ Take the average.
2. **Same Cooler, Any Day**: If Tuesday is missing: *"What does `Location01A` consume during Recess on any day?"* $\rightarrow$ Take that average.
3. **School Average**: If that cooler is new: *"What does any cooler consume during Recess across the school?"*
4. **Emergency Rule of Thumb**: If everything is missing: *"A 200-liter cooler at Recess drinks roughly 15–20 liters."*

**The Heat Adjustment**: If the weather report says it's hot (over $20^\circ\text{C}$), the model automatically increases the prediction by **+2.5% for every degree hotter**.

---

#### Model 2: The "AI Pattern Hunter" (CatBoost Quantile Model)
When you have multiple weeks of data (like the **2,160 rows** in your dataset), Model 2 activates. 

Instead of giving you just one average number, CatBoost predicts **two bounds**:
* **P50 (The Expected Demand)**: The realistic median. 50% of days use less, 50% use more.
* **P90 (The Conservative "Safe" Demand)**: A high ceiling. On 9 out of 10 days, kids will drink **less** than this number. 

> **Why P90 matters:** If you only plan for the average (P50), the cooler will run completely dry on 50% of the days! P90 gives you the safety buffer so water never runs out.

---

### 2. How the Models Use Your 4 Data Files

Here is how the system connects your data files together like puzzle pieces:

```
┌─────────────────────────────────┐
│ 1. cooler_config.csv            │ ──► Tells the system: Cooler Location04A holds 250 L
│    (8 Coolers & Locations)      │     and sits right next to the sports field.
└─────────────────────────────────┘
                 │
┌─────────────────────────────────┐
│ 2. Cooler_Water_Consumption.csv │ ──► Gives 5 weeks of history (2,160 records) of
│    (Historical Liters)          │     exactly how many liters were consumed each period.
└─────────────────────────────────┘
                 │
┌─────────────────────────────────┐
│ 3. School_Timetable.xlsx        │ ──► Tells the system: Recess starts right after Period 4
│    (Classes & Recess)           │     (everyone rushes to drink!) and Class 9-A has Games.
└─────────────────────────────────┘
                 │
┌─────────────────────────────────┐
│ 4. Water_Temperature.xlsx       │ ──► NOT an AI model. A strict physical table:
│    (IS 1475:2024 Rules)         │     If ambient temp is 33.5°C, cool water to 14°C.
└─────────────────────────────────┘     If cold outside (<20°C), turn chiller OFF.
```

---

### 3. A Real Example of How It Predicts

Suppose it is **Tuesday morning, 8:00 AM**:

1. **You ask the system**:  
   *"Predict demand for Cooler `Location04A` (capacity 250 L, currently holds 45 L from sensor). Today is Tuesday, outside temp is 34°C, Clear sky."*
2. **The system checks the timetable**:  
   It lays out the day: `P1, P2, P3, P4, Recess, P5, P6, P7, P8`.
3. **The system calculates period-by-period**:
   * During normal periods (`P1`–`P3`): Consumption is modest (~$5$ to $7$ Liters each).
   * **During `Recess` (after P4)**: Boom! The system detects `Recess` and knows `Location04A` is near the playground. Demand spikes to **~$18$ Liters**.
   * `P5` (right after recess): Thirst stays elevated (~$10$ Liters).
4. **It sums up the day**:
   * Total Expected Consumption: **$72.4$ Liters**
   * Total Conservative (Safe) Demand: **$90.5$ Liters**
5. **The Shortage Check (The Alarm Bell)**:
   * Current Water in Tank: **$45$ Liters**
   * Water Needed to Survive the Day: **$90.5$ Liters**
   * **Result**: Projected shortfall of **$-45.5$ Liters** before the day ends!
   * **Alert generated**: 
     > 🚨 **CRITICAL SHORTAGE**: *Tank will run completely dry around Period 5. Immediate refill recommended before Recess.*

---

### Summary Checklist

| Component | What It Is | What It Does In Plain Words |
| :--- | :--- | :--- |
| **Baseline Model** | Historical average with multipliers | Checks what this cooler drank on past Tuesdays at this period. |
| **CatBoost Model** | Smart machine learning regressor | Calculates the 90% safety upper bound (P90) so you don't run dry. |
| **Temperature Controller** | Deterministic rule table (IS 1475) | Acts like a smart thermostat for the chiller hardware. |
| **Shortage Engine** | Sensor level vs Predicted demand | Compares current tank level with predicted demand and sends refill alerts. |