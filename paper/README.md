# PaniPahara Research Paper: IEEE LaTeX Manuscript & Overleaf Bundle

This directory contains the self-contained LaTeX source code and bibliography for the research paper on **PaniPahara**.

---

## Paper Information

- **Title:**  
  *PaniPahara: Smart Drinking Water Demand Prediction and Energy-Efficient Cooler Control for School Campuses*
- **Authors:**  
  - **Abhiraj Adhikary** (Haldia Institute of Technology, Computer Science Engineering - Data Science, `abhirajadhikary06@gmail.com`)
  - **Aritra Karan** (Haldia Institute of Technology, Computer Science Engineering - Data Science, `aritrakaran234@gmail.com`)
  - **Anuj Kumar Yadav** (Kendriya Vidyalaya IIT Kharagpur, PGT Computer Science)
  - **Yashvardhan Yadav** (Kendriya Vidyalaya IIT Kharagpur, Computer Science 2025--27)
- **Format:** Strict IEEE Two-Column Conference/Journal Format (`IEEEtran.cls` v1.8b).
- **Layout Safeguards:**
  - Standard IEEE author block with no column-overlapping.
  - Every table and architecture diagram wrapped in `\resizebox{\columnwidth}{!}{...}` to guarantee zero column-gutter overflow.
  - All equations broken cleanly across multiple lines.
  - Clear, straightforward engineering language without unnecessary jargon.

---

## File Contents

```text
paper/
├── main.tex                       # Core IEEE LaTeX manuscript
├── references.bib                 # Clean BibTeX citations
├── IEEEtran.cls                   # Official IEEE class file (v1.8b)
├── IEEEtran.bst                   # Official IEEE bibliography style file
├── README.md                      # Compilation and Overleaf guide
└── panipahara_research_paper.zip  # 1-Click Overleaf upload package
```

---

## How to Upload to Overleaf

1. Download [`panipahara_research_paper.zip`](file:///c:/Abhiraj/OpenSource/paanipahrapurvanumanapi/paper/panipahara_research_paper.zip).
2. Go to **[Overleaf.com](https://www.overleaf.com/)** and log in.
3. Click **New Project** $\rightarrow$ **Upload Project**.
4. Drag and drop `panipahara_research_paper.zip`.
5. Overleaf will unpack all files automatically and set `main.tex` as the root document.
6. Click **Recompile** (Compiler: `pdfLaTeX` or `XeLaTeX`, TeX Live 2024).

---

## Key Experimental Results

| Metric | Baseline | PaniPahara (CatBoost + IS 1475) | Practical Impact |
|:---|:---:|:---:|:---:|
| **Mean Absolute Error (MAE)** | $1.284\,$L | **$0.913\,$L** | **$28.9\%$ error reduction** |
| **Shortage Prevention Coverage ($P_{90}$)** | $51.2\%$ | **$90.97\%$** | Tank dry-outs avoided in 9 of 10 surges |
| **Humid Heatwave Surge Error** | $1.412\,$L | **$1.042\,$L** | **$26.2\%$ error drop** via feels-like temp |
| **Cooler Electricity Consumption** | $28.4\,$kWh/day | **$18.7\,$kWh/day** | **$34.2\%$ power saved** via IS 1475:2024 |
| **Prediction Latency (Cached)** | --- | **$< 1.5\,$ms** | In-memory LRU prediction cache |
| **8-Cooler Parallel Dashboard Query** | $2,140\,$ms (serial) | **$242\,$ms (async)** | **$8.8\times$ faster response** |
