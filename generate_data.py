"""
Generate synthetic FARFETCH-style luxury fashion dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# --- Users ---
n_users = 500
user_ids = [f"U{i:04d}" for i in range(1, n_users + 1)]
ages = np.random.randint(22, 60, n_users)
genders = np.random.choice(["Women", "Men", "Other"], n_users, p=[0.55, 0.42, 0.03])
locations = np.random.choice(
    ["London", "New York", "Paris", "Dubai", "Hong Kong", "Milan", "Los Angeles", "Tokyo", "Singapore", "Riyadh"],
    n_users
)
# Preferred styles
styles = ["Contemporary", "Classic", "Streetwear", "Minimalist", "Avant-Garde", "Bohemian"]
user_styles = {
    uid: list(np.random.choice(styles, size=np.random.randint(1, 3), replace=False))
    for uid in user_ids
}

users = pd.DataFrame({
    "user_id": user_ids,
    "age": ages,
    "gender": genders,
    "location": locations,
    "preferred_styles": [",".join(user_styles[u]) for u in user_ids]
})
users.to_csv(OUT_DIR / "users.csv", index=False)

# --- Products (FARFETCH luxury fashion) ---
n_products = 300
categories = ["Ready-to-Wear", "Shoes", "Bags", "Accessories", "Jewellery", "Sneakers"]
brands = [
    "Gucci", "Prada", "Balenciaga", "Saint Laurent", "Bottega Veneta",
    "Off-White", "Valentino", "Burberry", "Alexander McQueen", "Loewe",
    "Jacquemus", "The Row", "Ami Paris", "Acne Studios", "Maison Margiela"
]
product_ids = [f"P{i:04d}" for i in range(1, n_products + 1)]
cat_choices = np.random.choice(categories, n_products)
brand_choices = np.random.choice(brands, n_products)
# Luxury price range
prices = np.round(np.random.uniform(180, 4500, n_products), 0)

products = pd.DataFrame({
    "product_id": product_ids,
    "name": [f"{b} {c} Item {i}" for i, (b, c) in enumerate(zip(brand_choices, cat_choices), 1)],
    "brand": brand_choices,
    "category": cat_choices,
    "price": prices,
    "style": np.random.choice(styles, n_products)
})
products.to_csv(OUT_DIR / "products.csv", index=False)

# --- Interactions (ratings + purchases) ---
n_interactions = 9000
rows = []
for _ in range(n_interactions):
    uid = np.random.choice(user_ids)
    prefs = user_styles[uid]
    if np.random.rand() < 0.7:
        # Prefer matching style
        cand = products[products["style"].isin(prefs)]["product_id"].values
        if len(cand) == 0:
            pid = np.random.choice(product_ids)
        else:
            pid = np.random.choice(cand)
    else:
        pid = np.random.choice(product_ids)

    style = products.loc[products["product_id"] == pid, "style"].values[0]
    base = 4.0 if style in prefs else 2.8
    rating = np.clip(np.random.normal(base, 0.9), 1.0, 5.0)
    rating = round(rating * 2) / 2
    purchased = 1 if rating >= 3.5 and np.random.rand() < 0.45 else 0
    rows.append({
        "user_id": uid,
        "product_id": pid,
        "rating": rating,
        "purchased": purchased,
        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 365))
    })

interactions = pd.DataFrame(rows)
interactions = interactions.drop_duplicates(subset=["user_id", "product_id"], keep="first")
interactions.to_csv(OUT_DIR / "interactions.csv", index=False)

print(f"FARFETCH dataset generated:")
print(f"  {len(users)} users")
print(f"  {len(products)} products")
print(f"  {len(interactions)} interactions")
print(f"  Brands: {', '.join(brands[:5])}...")
print(f"  Categories: {', '.join(categories)}")
print(f"Saved to {OUT_DIR}")
