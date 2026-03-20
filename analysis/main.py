"""
PageTurn Analytics — main.py
Flask API that serves your real Goodreads books.csv data to the dashboard.

Install dependencies (run once in your project folder):
    pip install flask pandas numpy

Run:
    python main.py

Then open:  http://localhost:5000
"""

import os, numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="../frontend")

# ── Load & clean ──────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "books.csv")

def load_data():
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df = df[["bookID","title","authors","average_rating",
             "ratings_count","num_pages","publication_date","publisher","language_code"]]
    df = df.dropna(subset=["title","authors","average_rating"])
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0)
    df["ratings_count"]  = pd.to_numeric(df["ratings_count"],  errors="coerce").fillna(0).astype(int)
    df["num_pages"]      = pd.to_numeric(df["num_pages"],      errors="coerce").fillna(0).astype(int)
    df = df[df["average_rating"] > 0]
    df["year"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.year.fillna(0).astype(int)

    # Genre classification
    fiction_kw    = ["novel","fantasy","romance","magic","witch","vampire","dragon","love",
                     "heart","princess","wizard","elf","myth","fairy","ghost","haunted",
                     "mystery","detective","spy","adventure","quest","warrior"]
    nonfiction_kw = ["guide","history","science","biography","memoir","philosophy",
                     "how to","psychology","business","economics","politics","religion",
                     "self","success","leadership","finance","health","diet","fitness"]
    def classify(title):
        t = str(title).lower()
        if any(k in t for k in fiction_kw):    return "Fiction"
        if any(k in t for k in nonfiction_kw): return "Non-Fiction"
        return "Fiction"
    df["genre"] = df["title"].apply(classify)

    # Simulated demographics
    np.random.seed(42)
    df["age"]       = np.random.randint(15, 60, size=len(df))
    df["age_group"] = pd.cut(df["age"], bins=[10,20,30,40,50,60],
                              labels=["10-20","20-30","30-40","40-50","50-60"])
    df["state"] = np.random.choice(
        ["Tamil Nadu","Kerala","Karnataka","Delhi","Maharashtra",
         "West Bengal","Gujarat","Rajasthan","Punjab","Telangana"],
        size=len(df))
    return df

print("Loading dataset…")
DF = load_data()
print(f"Loaded {len(DF):,} books.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/api/kpis")
def kpis():
    return jsonify({
        "total_reads":  int(DF["ratings_count"].sum()),
        "total_books":  len(DF),
        "avg_rating":   round(float(DF["average_rating"].mean()), 2),
        "avg_per_book": round(float(DF["ratings_count"].mean()), 1),
    })

@app.route("/api/top_books")
def top_books():
    limit = int(request.args.get("limit", 10))
    top = (DF[DF["ratings_count"] > 1000]
           .sort_values("ratings_count", ascending=False)
           .head(limit)
           [["bookID","title","authors","average_rating","ratings_count","genre","num_pages","year"]]
           .fillna(""))
    return jsonify(top.to_dict(orient="records"))

@app.route("/api/books")
def books():
    page    = int(request.args.get("page", 1))
    per     = int(request.args.get("per",  12))
    q       = request.args.get("q",     "").lower()
    genre   = request.args.get("genre", "")
    year    = request.args.get("year",  "")
    sort_by = request.args.get("sort",  "ratings_count")
    order   = request.args.get("order", "desc")

    df = DF.copy()
    if q:     df = df[df["title"].str.lower().str.contains(q, na=False) | df["authors"].str.lower().str.contains(q, na=False)]
    if genre: df = df[df["genre"] == genre]
    if year:  df = df[df["year"] == int(year)]

    if sort_by not in {"ratings_count","average_rating","title","year","num_pages"}:
        sort_by = "ratings_count"
    df = df.sort_values(sort_by, ascending=(order == "asc"))

    total  = len(df)
    pages  = max(1, (total + per - 1) // per)
    page   = min(page, pages)
    sl     = df.iloc[(page-1)*per : page*per]
    cols   = ["bookID","title","authors","average_rating","ratings_count","genre","num_pages","year","publisher"]
    return jsonify({"total": total, "pages": pages, "page": page,
                    "books": sl[cols].fillna("").to_dict(orient="records")})

@app.route("/api/genres")
def genres():
    g = (DF.groupby("genre")
           .agg(reads=("ratings_count","sum"), avg_rating=("average_rating","mean"), book_count=("bookID","count"))
           .reset_index())
    g["avg_rating"] = g["avg_rating"].round(2)
    g["reads"]      = g["reads"].astype(int)
    return jsonify(g.sort_values("reads", ascending=False).to_dict(orient="records"))

@app.route("/api/age_groups")
def age_groups():
    ag = (DF.groupby("age_group", observed=True)
            .agg(count=("bookID","count"), fiction_count=("genre", lambda x: (x=="Fiction").sum()))
            .reset_index())
    ag["age_group"]       = ag["age_group"].astype(str)
    ag["nonfiction_count"]= ag["count"] - ag["fiction_count"]
    ag["fiction_pct"]     = (ag["fiction_count"]    / ag["count"] * 100).round(1)
    ag["nonfiction_pct"]  = (ag["nonfiction_count"] / ag["count"] * 100).round(1)
    return jsonify(ag.to_dict(orient="records"))

@app.route("/api/states")
def states():
    st = (DF.groupby("state")["ratings_count"].sum()
            .sort_values(ascending=False).reset_index())
    st.columns = ["state","reads"]
    st["reads"] = st["reads"].astype(int)
    return jsonify(st.to_dict(orient="records"))

@app.route("/api/rating_dist")
def rating_dist():
    df = DF[DF["average_rating"] > 0].copy()
    df["bucket"] = pd.cut(df["average_rating"], bins=[0,1,2,3,4,5], labels=["0-1","1-2","2-3","3-4","4-5"])
    dist = df.groupby("bucket", observed=True).size().reset_index(name="count")
    dist["bucket"] = dist["bucket"].astype(str)
    return jsonify(dist.to_dict(orient="records"))

@app.route("/api/year_trend")
def year_trend():
    df = DF[(DF["year"] >= 1990) & (DF["year"] <= 2020)].copy()
    t  = df.groupby(["year","genre"]).agg(reads=("ratings_count","sum"), books=("bookID","count")).reset_index()
    t["year"]  = t["year"].astype(int)
    t["reads"] = t["reads"].astype(int)
    return jsonify(t.to_dict(orient="records"))

@app.route("/api/top_authors")
def top_authors():
    limit = int(request.args.get("limit", 10))
    a = (DF.groupby("authors")
          .agg(books=("bookID","count"), total_reads=("ratings_count","sum"), avg_rating=("average_rating","mean"))
          .reset_index()
          .sort_values("total_reads", ascending=False)
          .head(limit))
    a["total_reads"] = a["total_reads"].astype(int)
    a["avg_rating"]  = a["avg_rating"].round(2)
    return jsonify(a.to_dict(orient="records"))

@app.route("/api/publishers")
def publishers():
    limit = int(request.args.get("limit", 10))
    p = (DF.groupby("publisher")
          .agg(books=("bookID","count"), total_reads=("ratings_count","sum"))
          .reset_index()
          .sort_values("total_reads", ascending=False)
          .head(limit))
    p["total_reads"] = p["total_reads"].astype(int)
    return jsonify(p.to_dict(orient="records"))

@app.route("/api/filters")
def filters():
    years  = sorted([int(y) for y in DF["year"].unique() if y > 1950], reverse=True)
    genres = sorted(DF["genre"].dropna().unique().tolist())
    return jsonify({"years": years, "genres": genres})

if __name__ == "__main__":
    print("\n🚀  PageTurn API →  http://localhost:5000\n")
    app.run(debug=True, port=5000)
