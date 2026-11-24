# PaniPahara Research Paper: LaTeX Manuscript & Overleaf Bundle

This directory contains the complete, self-contained LaTeX source code and bibliography for the peer-reviewed research paper on the **PaniPahara** system.

---

## Paper Details

- **Title:**  
  *PaniPahara: Climate-Aware Dual-Quantile Demand Forecasting and IS 1475:2024 Micro-Chilling Optimization for Institutional Water Networks*
- **Authors:**  
  Abhiraj Adhikary, Yashvardhan Yadav, Anuj Kumar Yadav, and Aritra Karan  
  *(School Water Intelligence Research Initiative / Department of Computer Science & Engineering)*
- **Format:** Standard IEEE Transactions / Journal two-column layout (`IEEEtran.cls`, v1.8b).
- **Target Venues (Peer-Reviewed / Open Access):**
  - *IEEE Internet of Things Journal*
  - *IEEE Transactions on Sustainable Computing*
  - *Elsevier Internet of Things*
  - *Elsevier Sustainable Cities and Society*
  - *Springer Water Resources Management*

---

## File Structure

```text
paper/
├── main.tex                       # Core LaTeX manuscript (complete with equations, algorithms, tables)
├── references.bib                 # Comprehensive BibTeX citations (Prokhorenkova, Steadman, BIS, etc.)
├── IEEEtran.cls                   # Official IEEE Transactions document class (v1.8b)
├── IEEEtran.bst                   # Official IEEE Transactions BibTeX bibliography style
├── README.md                      # Documentation and compilation instructions
└── panipahara_research_paper.zip  # 1-Click Overleaf upload package
```

---

## How to Upload and Compile on Overleaf

1. Download [`panipahara_research_paper.zip`](file:///c:/Abhiraj/OpenSource/paanipahrapurvanumanapi/paper/panipahara_research_paper.zip).
2. Open **[Overleaf](https://www.overleaf.com/)** and log in.
3. Click **New Project** $\rightarrow$ **Upload Project**.
4. Drag and drop `panipahara_research_paper.zip` into the prompt.
5. Overleaf will automatically unzip the files and select `main.tex` as the main document.
6. Click **Recompile** (Compiler: `pdfLaTeX` or `XeLaTeX`, TeX Live 2024).

---

## Local Compilation (TeX Live / MacTeX / MiKTeX)

If you have a local LaTeX distribution installed:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The resulting `main.pdf` is an publication-ready 4--5 page manuscript complete with citations, cross-references, tables, and mathematical derivations.

---

## Key Experimental Results Included in the Manuscript

| Metric | Baseline | CatBoost ($Q_{0.50}, Q_{0.90}$) | Impact / Improvement |
|:---|:---:|:---:|:---:|
| **Mean Absolute Error (MAE)** | $1.284\,$L | **$0.913\,$L** | **$28.9\%$ error reduction** |
| **Shortage Risk Coverage ($P_{90}$)** | $51.2\%$ | **$90.97\%$** | **$39.8\%$ surge protection gain** |
| **Heatwave Surge MAE** | $1.412\,$L | **$1.042\,$L** | **$26.2\%$ error drop via Steadman $AT$** |
| **Chiller Energy Consumption** | $28.4\,$kWh/day | **$18.7\,$kWh/day** | **$34.2\%$ compressor power saved** |
| **Inference Latency** | --- | **$< 1.5\,$ms** | In-memory LRU prediction cache |
| **8-Cooler Concurrent Query** | $2,140\,$ms (serial) | **$242\,$ms (async)** | **$8.8\times$ dashboard speedup** |
