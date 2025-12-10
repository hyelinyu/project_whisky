import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from recommender_model import WhiskyRecommender

# ---------------------------------------------------------
# Logging 설정
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Flask 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates")
CORS(app)

# ---------------------------------------------------------
# 데이터 로드 + 모델 생성
# ---------------------------------------------------------
try:
    DATA_PATH = os.path.join(BASE_DIR, "data", "whisky_recommendation.csv")
    logger.info(f"Loading data from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} whiskies from CSV")

    # 필수 컬럼 확인
    required_cols = ["name", "style_missing"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    recommender = WhiskyRecommender(df)
    logger.info("Recommender initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize recommender: {str(e)}")
    raise

# 공통 출력 컬럼 (있으면 쓰고, 없으면 자동으로 제외)
BASIC_COLS = [
    "name", "country", "region", "whisky_type",
    "price(£)", "alcohol(%)",
    "style_body", "style_richness", "style_smoke", "style_sweetness",
    "body_tag", "richness_tag", "smoke_tag", "sweetness_tag",
]


def select_basic_cols(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    BASIC_COLS 중에서 실제로 df_like에 존재하는 컬럼만 선택해서 반환.
    """
    cols = [c for c in BASIC_COLS if c in df_like.columns]
    if not cols:
        return df_like
    return df_like[cols]


def df_to_records(df_like: pd.DataFrame):
    """
    JSON 응답용으로 NaN → None 으로 바꾸고 records 리스트로 변환.
    (NaN 이 그대로 나가면 브라우저 JSON.parse 에서 에러가 나므로 반드시 필요)
    """
    df_clean = df_like.where(pd.notnull(df_like), None)
    return df_clean.to_dict(orient="records")


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
    try:
        payload = request.get_json(force=True)
        logger.info(f"Survey recommendation request: {payload}")

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

        if "taste_only_main" not in res:
            raise ValueError("recommend 결과에 'taste_only_main' 키가 없습니다.")

        base_df = res["taste_only_main"].head(5)
        base_df = select_basic_cols(base_df)
        records = df_to_records(base_df)

        logger.info(f"Survey recommendation returned {len(records)} results")

        return jsonify({
            "mode": res.get("mode", "survey"),
            "recommendations": records,
        })

    except Exception as e:
        logger.exception("Survey recommendation error")
        return jsonify({"error": str(e), "recommendations": []}), 500


# ---------------------------------------------------------
# 랜덤 위스키 샘플 제공 API
# ---------------------------------------------------------
@app.get("/api/whiskies/sample")
def api_whisky_sample():
    try:
        n = int(request.args.get("n", 9))
        n = min(max(n, 1), len(df))
        logger.info(f"Sample request: n={n}")

        sample_df = df.sample(n)
        sample_df = select_basic_cols(sample_df)
        records = df_to_records(sample_df)

        logger.info(f"Returning {len(records)} sample whiskies")

        return jsonify(records)

    except Exception as e:
        logger.exception("Sample API error")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# 제품 기반 추천 API
# ---------------------------------------------------------
@app.post("/api/recommend/products")
def api_recommend_products():
    try:
        payload = request.get_json(force=True)
        product_list = payload.get("product_list", [])
        logger.info(f"Product recommendation request: {product_list}")

        if not product_list:
            logger.warning("Empty product list received")
            return jsonify({
                "mode": "empty",
                "input_products": [],
                "recommendations": []
            })

        res = recommender.recommend(
            product_list=product_list,
            top_k_taste=20,
            top_k_meta=20,
        )

        if "taste_based_main" in res:
            base_df = res["taste_based_main"].head(5)
        else:
            base_df = res["meta_based_main"].head(5)

        base_df = select_basic_cols(base_df)
        records = df_to_records(base_df)

        logger.info(f"Product recommendation returned {len(records)} results")

        return jsonify({
            "mode": res.get("mode", "product"),
            "input_products": res.get("input_products", []),
            "recommendations": records,
        })

    except ValueError as e:
        logger.error(f"Product not found error: {str(e)}")
        return jsonify(
            {"error": f"제품을 찾을 수 없습니다: {str(e)}", "recommendations": []}
        ), 404

    except Exception as e:
        logger.exception("Product recommendation error")
        return jsonify({"error": str(e), "recommendations": []}), 500


# ---------------------------------------------------------
# 헬스체크 (Render용)
# ---------------------------------------------------------
@app.get("/health")
def health():
    return "ok", 200


# ---------------------------------------------------------
# 로컬 실행용
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
