"""
FARFETCH Luxury Fashion Recommendation System
Hybrid Collaborative Filtering + Matrix Factorization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
np.random.seed(42)


class FarfetchRecommender:
    def __init__(self):
        self.users = None
        self.products = None
        self.interactions = None
        self.user_item_matrix = None
        self.user_sim = None
        self.item_sim = None
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.user_id_to_idx = {}
        self.product_id_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_product = {}

    def load_data(self):
        self.users = pd.read_csv(DATA_DIR / "users.csv")
        self.products = pd.read_csv(DATA_DIR / "products.csv")
        self.interactions = pd.read_csv(DATA_DIR / "interactions.csv")
        print(f"Loaded {len(self.users)} users, {len(self.products)} products, "
              f"{len(self.interactions)} interactions")

    def preprocess(self, min_interactions=3):
        user_counts = self.interactions["user_id"].value_counts()
        prod_counts = self.interactions["product_id"].value_counts()
        active_users = user_counts[user_counts >= min_interactions].index
        active_prods = prod_counts[prod_counts >= min_interactions].index
        self.interactions = self.interactions[
            self.interactions["user_id"].isin(active_users) &
            self.interactions["product_id"].isin(active_prods)
        ].copy()

        unique_users = sorted(self.interactions["user_id"].unique())
        unique_prods = sorted(self.interactions["product_id"].unique())
        self.user_id_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.product_id_to_idx = {p: i for i, p in enumerate(unique_prods)}
        self.idx_to_user = {i: u for u, i in self.user_id_to_idx.items()}
        self.idx_to_product = {i: p for p, i in self.product_id_to_idx.items()}

        n_u, n_p = len(unique_users), len(unique_prods)
        self.user_item_matrix = np.zeros((n_u, n_p))
        for _, row in self.interactions.iterrows():
            ui = self.user_id_to_idx[row["user_id"]]
            pi = self.product_id_to_idx[row["product_id"]]
            self.user_item_matrix[ui, pi] = row["rating"]

        self.global_mean = self.user_item_matrix[self.user_item_matrix > 0].mean()
        print(f"After filtering: {n_u} users, {n_p} products, "
              f"density={np.count_nonzero(self.user_item_matrix)/(n_u*n_p):.4f}")

    def _cosine_sim(self, matrix, axis=0):
        norms = np.linalg.norm(matrix, axis=1 - axis, keepdims=True)
        norms[norms == 0] = 1e-10
        if axis == 0:
            normalized = matrix / norms
            sim = normalized.T @ normalized
        else:
            normalized = matrix / norms
            sim = normalized @ normalized.T
        np.fill_diagonal(sim, 0)
        return sim

    def train_collaborative(self):
        print("Training collaborative filtering...")
        self.user_sim = self._cosine_sim(self.user_item_matrix, axis=1)
        self.item_sim = self._cosine_sim(self.user_item_matrix, axis=0)
        print("  Done.")

    def train_matrix_factorization(self, n_factors=15, n_epochs=20, lr=0.01, reg=0.02):
        print(f"Training Matrix Factorization ({n_factors} factors)...")
        n_u, n_p = self.user_item_matrix.shape
        self.user_factors = np.random.normal(0, 0.1, (n_u, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_p, n_factors))
        self.user_bias = np.zeros(n_u)
        self.item_bias = np.zeros(n_p)

        obs = np.argwhere(self.user_item_matrix > 0)
        for epoch in range(n_epochs):
            np.random.shuffle(obs)
            total_err = 0.0
            for ui, pi in obs:
                r = self.user_item_matrix[ui, pi]
                pred = (self.global_mean + self.user_bias[ui] + self.item_bias[pi] +
                        self.user_factors[ui] @ self.item_factors[pi])
                err = r - pred
                total_err += err ** 2
                self.user_bias[ui] += lr * (err - reg * self.user_bias[ui])
                self.item_bias[pi] += lr * (err - reg * self.item_bias[pi])
                uf = self.user_factors[ui].copy()
                self.user_factors[ui] += lr * (err * self.item_factors[pi] - reg * self.user_factors[ui])
                self.item_factors[pi] += lr * (err * uf - reg * self.item_factors[pi])
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}  RMSE={np.sqrt(total_err/len(obs)):.4f}")
        print("  Done.")

    def predict_user_based(self, user_idx, product_idx, k=30):
        if self.user_item_matrix[user_idx, product_idx] > 0:
            return self.user_item_matrix[user_idx, product_idx]
        sims = self.user_sim[user_idx]
        rated_mask = self.user_item_matrix[:, product_idx] > 0
        candidates = np.where(rated_mask)[0]
        if len(candidates) == 0:
            return self.global_mean
        sims_c = sims[candidates]
        top_k = np.argsort(sims_c)[-k:]
        top_sims = sims_c[top_k]
        top_ratings = self.user_item_matrix[candidates[top_k], product_idx]
        if np.sum(np.abs(top_sims)) < 1e-8:
            return self.global_mean
        return np.average(top_ratings, weights=np.abs(top_sims))

    def predict_item_based(self, user_idx, product_idx, k=30):
        if self.user_item_matrix[user_idx, product_idx] > 0:
            return self.user_item_matrix[user_idx, product_idx]
        sims = self.item_sim[product_idx]
        rated_mask = self.user_item_matrix[user_idx, :] > 0
        candidates = np.where(rated_mask)[0]
        if len(candidates) == 0:
            return self.global_mean
        sims_c = sims[candidates]
        top_k = np.argsort(sims_c)[-k:]
        top_sims = sims_c[top_k]
        top_ratings = self.user_item_matrix[user_idx, candidates[top_k]]
        if np.sum(np.abs(top_sims)) < 1e-8:
            return self.global_mean
        return np.average(top_ratings, weights=np.abs(top_sims))

    def predict_mf(self, user_idx, product_idx):
        pred = (self.global_mean + self.user_bias[user_idx] + self.item_bias[product_idx] +
                self.user_factors[user_idx] @ self.item_factors[product_idx])
        return np.clip(pred, 1.0, 5.0)

    def predict_hybrid(self, user_idx, product_idx, weights=(0.3, 0.3, 0.4)):
        p1 = self.predict_user_based(user_idx, product_idx)
        p2 = self.predict_item_based(user_idx, product_idx)
        p3 = self.predict_mf(user_idx, product_idx)
        return weights[0]*p1 + weights[1]*p2 + weights[2]*p3

    def recommend(self, user_id, n=10, method="hybrid"):
        if user_id not in self.user_id_to_idx:
            return self._popular_products(n)
        ui = self.user_id_to_idx[user_id]
        already = set(np.where(self.user_item_matrix[ui] > 0)[0])
        scores = []
        for pi in range(self.user_item_matrix.shape[1]):
            if pi in already:
                continue
            if method == "user":
                score = self.predict_user_based(ui, pi)
            elif method == "item":
                score = self.predict_item_based(ui, pi)
            elif method == "mf":
                score = self.predict_mf(ui, pi)
            else:
                score = self.predict_hybrid(ui, pi)
            scores.append((pi, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for pi, sc in scores[:n]:
            pid = self.idx_to_product[pi]
            prod = self.products[self.products["product_id"] == pid].iloc[0]
            results.append({
                "product_id": pid,
                "name": prod["name"],
                "brand": prod["brand"],
                "category": prod["category"],
                "price": prod["price"],
                "style": prod["style"],
                "predicted_rating": round(sc, 3)
            })
        return results

    def _popular_products(self, n=10):
        avg = self.interactions.groupby("product_id")["rating"].mean().sort_values(ascending=False)
        results = []
        for pid in avg.head(n).index:
            prod = self.products[self.products["product_id"] == pid].iloc[0]
            results.append({
                "product_id": pid,
                "name": prod["name"],
                "brand": prod["brand"],
                "category": prod["category"],
                "price": prod["price"],
                "style": prod["style"],
                "predicted_rating": round(avg[pid], 3)
            })
        return results


def main():
    rec = FarfetchRecommender()
    rec.load_data()
    rec.preprocess()
    rec.train_collaborative()
    rec.train_matrix_factorization()
    print("\n=== Sample Recommendations ===")
    for uid in list(rec.user_id_to_idx.keys())[:2]:
        print(f"\nFor {uid}:")
        for r in rec.recommend(uid, n=5):
            print(f"  {r['brand']:18s} | {r['category']:14s} | ${r['price']:7.0f} | "
                  f"pred={r['predicted_rating']:.2f} | {r['name'][:40]}")
    return rec


if __name__ == "__main__":
    main()
