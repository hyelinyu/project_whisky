import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from recommender_model import WhiskyRecommender

# ---------------------------------------------------------
# Flask 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates")
CORS(app)

# ---------------------------------------------------------
# 데이터 로드 + 모델 생성
# ---------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "whisky_recommendation.csv")
df = pd.read_csv(DATA_PATH)

recommender = WhiskyRecommender(df)

# 공통 출력 컬럼
BASIC_COLS = [
    "name", "country", "region", "whisky_type",
    "price(£)", "alcohol(%)",
    "style_body", "style_richness", "style_smoke", "style_sweetness",
]

# ---------------------------------------------------------
# 메인 페이지
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------------------------------------
# 설문 기반 추천 API
# ---------------------------------------------------------
@app.post("/api/recommend/survey")
def api_recommend_survey():
    payload = request.get_json(force=True)

    body_level = payload.get("body_level", "medium")
    rich_level = payload.get("rich_level", "round")
    smoke_level = payload.get("smoke_level", "non")
    sweet_level = payload.get("sweet_level", "balanced")
    selected_families = payload.get("selected_families")

    res = recommender.recommend(
        body_level=body_level,
        rich_level=rich_level,
        smoke_level=smoke_level,
        sweet_level=sweet_level,
        selected_families=selected_families,
        top_k_taste=20,
        top_k_meta=10,
    )

    df_out = res["taste_only_main"].head(5)[BASIC_COLS]

    return jsonify({
        "mode": res["mode"],
        "recommendations": df_out.to_dict(orient="records"),
    })

# ---------------------------------------------------------
# 랜덤 위스키 샘플 제공 API
# ---------------------------------------------------------
@app.get("/api/whiskies/sample")
def api_whisky_sample():
    n = int(request.args.get("n", 9))
    sample_df = df.sample(n)[BASIC_COLS]
    return jsonify(sample_df.to_dict(orient="records"))

# ---------------------------------------------------------
# 제품 기반 추천 API (라운드로빈 적용)
# ---------------------------------------------------------
@app.post("/api/recommend/products")
def api_recommend_products():
    payload = request.get_json(force=True)
    product_list = payload.get("product_list", [])

    res = recommender.recommend(
        product_list=product_list,
        top_k_taste=20,
        top_k_meta=20,
    )

    # taste version 우선
    if "taste_based_main" in res:
        df_out = res["taste_based_main"].head(5)
    else:
        df_out = res["meta_based_main"].head(5)

    df_out = df_out[BASIC_COLS]

    return jsonify({
        "mode": res["mode"],
        "input_products": res.get("input_products", []),
        "recommendations": df_out.to_dict(orient="records"),
    })

# ---------------------------------------------------------
# 로컬 실행용
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
