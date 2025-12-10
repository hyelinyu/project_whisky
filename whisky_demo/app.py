# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from recommender_model import WhiskyRecommender

# templates 폴더에서 index.html 찾도록 설정
app = Flask(__name__, template_folder="templates")
CORS(app)  # 사실 같은 도메인에서 쓸 거면 없어도 되지만 놔둬도 됨

# 1) 데이터 로드 + 모델 생성 -----------------------------
df = pd.read_csv("data/whisky_recommendation.csv")   # 경로/파일명 맞게 유지
recommender = WhiskyRecommender(df)

# 공통: 응답에 포함할 컬럼
BASIC_COLS = [
    "name", "country", "region", "whisky_type",
    "price(£)", "alcohol(%)",
    "style_body", "style_richness", "style_smoke", "style_sweetness"
]

# 0) 메인 페이지 (HTML 렌더링) -----------------------------
@app.route("/")
def index():
    # templates/index.html 을 렌더링
    return render_template("index.html")

# 2) 설문 기반 추천 --------------------------------------
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
        top_k_meta=0,
    )

    df_out = res["taste_only_main"].head(5)
    df_out = df_out[BASIC_COLS].copy()

    return jsonify({
        "mode": res["mode"],
        "recommendations": df_out.to_dict(orient="records"),
    })

# 3) 샘플 위스키 9개 --------------------------------------
@app.get("/api/whiskies/sample")
def api_whisky_sample():
    n = int(request.args.get("n", 9))
    sample_df = df.sample(n)
    sample_df = sample_df[BASIC_COLS].copy()
    return jsonify(sample_df.to_dict(orient="records"))

# 4) 제품 기반 추천 --------------------------------------
@app.post("/api/recommend/products")
def api_recommend_products():
    payload = request.get_json(force=True)
    product_list = payload.get("product_list", [])

    res = recommender.recommend(
        product_list=product_list,
        top_k_taste=20,
        top_k_meta=0,
    )

    if "taste_based_main" in res:
        df_out = res["taste_based_main"].head(5)
    else:
        df_out = res["meta_based_main"].head(5)

    df_out = df_out[BASIC_COLS].copy()

    return jsonify({
        "mode": res["mode"],
        "input_products": res.get("input_products", []),
        "recommendations": df_out.to_dict(orient="records"),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
