from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

from recommender_model import WhiskyRecommender


def to_float_or_none(x: str) -> Optional[float]:
    """폼에서 받은 값을 float로 변환. 빈 문자열/None이면 None."""
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "none":
        return None
    try:
        return float(x)
    except Exception:
        return None


def chunk_page(items: List[Dict[str, Any]], page: int, per_page: int = 7) -> Tuple[List[Dict[str, Any]], int]:
    """list[dict]를 페이지 단위로 자르기"""
    total = len(items)
    total_pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], total_pages


def df_to_cards(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    카드 표시용으로 필요한 컬럼만 뽑아서 dict 리스트로 변환.
    (데이터셋 컬럼이 조금 달라도 안전하게 동작하도록 get 사용)
    """
    if df is None or df.empty:
        return []

    cols = [
        "name",
        "whisky_type",
        "region",
        "country",
        "age",
        "vintage",
        "bottler_group",
        "cask_group",
        "price(£)",
        "abv",
        "final_score",
        "taste_distance",
        "meta_distance",
        "rare_meta_distance",
        "family_match",
        "style_missing",
    ]

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        item: Dict[str, Any] = {}
        for c in cols:
            item[c] = row[c] if c in df.columns else None
        out.append(item)
    return out


def build_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

    # ----------------------------
    # 1) 데이터 로드 (Render/로컬 공통: repo 상대경로)
    # ----------------------------
    base_dir = Path(__file__).resolve().parent

    # 환경변수로 경로를 주면 그걸 우선 사용 (선택)
    env_path = os.environ.get("WHISKY_DATA_PATH")

    if env_path:
        data_path = Path(env_path)
        if not data_path.is_absolute():
            data_path = (base_dir / data_path).resolve()
    else:
        data_path = base_dir / "data" / "whisky_recommendation.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # ✅ 모델 생성
    recommender = WhiskyRecommender(df)

    # ✅ family 후보(14개) 자동 생성: *_family 컬럼에서 prefix 추출
    family_options = sorted([c.replace("_family", "") for c in recommender.family_cols])

    # ----------------------------
    # Routes
    # ----------------------------
    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            family_options=family_options,
            input_state={
                "style_body": None,
                "style_richness": None,
                "style_smoke": None,
                "style_sweetness": None,
                "selected_families": [],
            },
            main_cards=[],
            rare_cards=[],
            main_page=1,
            main_total_pages=1,
            rare_page=1,
            rare_total_pages=1,
            has_results=False,
        )

    @app.route("/recommend", methods=["POST", "GET"])
    def recommend():
        if request.method == "POST":
            style_body = to_float_or_none(request.form.get("style_body"))
            style_richness = to_float_or_none(request.form.get("style_richness"))
            style_smoke = to_float_or_none(request.form.get("style_smoke"))
            style_sweetness = to_float_or_none(request.form.get("style_sweetness"))

            selected_families = request.form.getlist("families")

            out = recommender.recommend_from_survey(
                style_body=style_body,
                style_richness=style_richness,
                style_smoke=style_smoke,
                style_sweetness=style_sweetness,
                selected_families=selected_families,
                final_k=60,
                top_k_rare=60,
                random_state=None,
            )

            main_df = out.get("final_candidates", pd.DataFrame())
            rare_df = out.get("rare", {}).get("rare_personalized_by_meta", pd.DataFrame())

            session["input_state"] = {
                "style_body": style_body,
                "style_richness": style_richness,
                "style_smoke": style_smoke,
                "style_sweetness": style_sweetness,
                "selected_families": selected_families,
            }
            session["main_cards"] = df_to_cards(main_df)
            session["rare_cards"] = df_to_cards(rare_df)

            return redirect(url_for("recommend", main_page=1, rare_page=1))

        input_state = session.get(
            "input_state",
            {
                "style_body": None,
                "style_richness": None,
                "style_smoke": None,
                "style_sweetness": None,
                "selected_families": [],
            },
        )
        main_cards_all = session.get("main_cards", [])
        rare_cards_all = session.get("rare_cards", [])

        main_page = int(request.args.get("main_page", 1))
        rare_page = int(request.args.get("rare_page", 1))

        main_cards, main_total_pages = chunk_page(main_cards_all, main_page, per_page=7)
        rare_cards, rare_total_pages = chunk_page(rare_cards_all, rare_page, per_page=7)

        return render_template(
            "index.html",
            family_options=family_options,
            input_state=input_state,
            main_cards=main_cards,
            rare_cards=rare_cards,
            main_page=main_page,
            main_total_pages=main_total_pages,
            rare_page=rare_page,
            rare_total_pages=rare_total_pages,
            has_results=(len(main_cards_all) > 0 or len(rare_cards_all) > 0),
        )

    return app


app = build_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
