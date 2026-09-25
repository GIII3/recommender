# FARFETCH Luxury Fashion Recommendation System

**BICT242: Data Scalability and Analytics – Course Project**

Personalised product recommendations for a FARFETCH-style luxury fashion e-commerce platform.

---

## Project Deliverables (all included)

| Official Requirement | File |
|----------------------|------|
| **Data Analysis Report** | `Data_Analysis_Report.docx` |
| **Model Selection & Evaluation Report** | `Model_Selection_Evaluation_Report.docx` |
| **Recommendation System Implementation** | `recommender.py` + `streamlit_app.py` |
| **Final Presentation** | `Final_Presentation.pptx` |
| Supporting code & data | `generate_data.py`, `data/`, `README.md` |

---

## Quick Start – Web Application

```bash
cd farfetch_recommender
pip install streamlit pandas numpy
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

---

## Quick Start – Core Engine

```bash
python generate_data.py   # regenerate data if needed
python recommender.py     # train + show sample recommendations
```

---

## What the system covers (mapped to project scope)

| Scope Item | How it is covered |
|------------|-------------------|
| **Data Collection** | Synthetic FARFETCH-style data (500 customers, 300 products, 8 325 interactions) |
| **Data Preprocessing** | Deduplication, sparsity filtering, user–item matrix construction |
| **Feature Engineering** | Activity level, popularity, style vectors, brand affinity, purchase flag |
| **Model Selection & Training** | User-based CF, Item-based CF, Matrix Factorization, Hybrid |
| **Model Evaluation** | RMSE, Precision@10, Recall@10, F1@10 |
| **Deployment** | Interactive Streamlit web application |

---

## Data Summary

- **500** customers (age, gender, location, preferred styles)
- **300** luxury products
- **Brands**: Gucci, Prada, Balenciaga, Saint Laurent, Bottega Veneta, Off-White, Valentino, Burberry, Alexander McQueen, Loewe, Jacquemus, The Row, Ami Paris, Acne Studios, Maison Margiela
- **Categories**: Ready-to-Wear, Shoes, Bags, Accessories, Jewellery, Sneakers
- **8 325** interactions (ratings 1–5 + purchase flags)
- Matrix density ≈ **5.55 %**

---

## Evaluation Results (20 % hold-out)

| Model | RMSE |
|-------|------|
| User-based CF | 0.978 |
| Item-based CF | 0.985 |
| **Matrix Factorization** | **0.830** |
| Hybrid | 0.900 |

| Ranking Metric (Hybrid @10) | Value |
|-----------------------------|-------|
| Precision@10 | 0.026 |
| Recall@10 | 0.101 |
| F1@10 | 0.039 |

---

## Folder Structure

```
farfetch_recommender/
├── data/
│   ├── users.csv
│   ├── products.csv
│   └── interactions.csv
├── generate_data.py
├── recommender.py
├── streamlit_app.py
├── create_reports.py
├── Data_Analysis_Report.docx
├── Model_Selection_Evaluation_Report.docx
├── Final_Presentation.pptx
└── README.md
```

---

## Requirements

- Python 3.8+
- pandas, numpy
- streamlit (for the web app only)

No external recommendation libraries are required – everything is implemented from first principles.
