
import numpy as np
import pandas as pd

from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
class WhiskyRecommender:
    """
    Whisky Recommender (FA taste + Meta KNN 기반)

    시나리오
    --------
    - Scenario A: 스타일 있는 제품(들)을 기준으로 추천
    - Scenario B: 스타일 없는 제품(들)을 기준으로 추천
    - Scenario C: 설문(Body/Rich/Smoke/Sweet + family)만으로 추천

    특징
    ----
    - taste space: style_body, style_richness, style_smoke, style_sweetness → FA(2차원)
    - meta space:
        * categorical: whisky_type, region, bottler_group, cask_group, age_bin, price_bin
        * numeric: vintage_flag (0/1)
        * age, price는 bin + Unknown (NA를 median으로 채우지 않음)
        * whisky_type > region > bottler_group > cask_group 순으로 가중치
    - special item 분리: 빈티지 / 고연산 / 고가 / high rarity 제품을 따로
    - 설명:
        * taste_explanation: “비슷한 맛 기반” 안내문
        * meta_explanation: “어떤 meta 공통점으로 추천했는지” 안내문
    """

    # --------------------------------------------------------- #
    #                          INIT                             #
    # --------------------------------------------------------- #
    def __init__(
        self,
        df: pd.DataFrame,
        style_cols=None,
        family_cols=None,
        n_taste_neighbors: int = 200,
        n_meta_neighbors: int = 200,
    ):
        self.df = df.copy()

        self.style_cols = style_cols or [
            "style_body",
            "style_richness",
            "style_smoke",
            "style_sweetness",
        ]

        # *_family 자동 탐색
        if family_cols is None:
            self.family_cols = [c for c in self.df.columns if c.endswith("_family")]
        else:
            self.family_cols = family_cols

        # taste space
        self._build_taste_space(n_taste_neighbors)

        # meta space
        self._build_meta_space(n_meta_neighbors)

    # --------------------------------------------------------- #
    #                    TASTE SPACE (FA)                       #
    # --------------------------------------------------------- #
    def _build_taste_space(self, n_neighbors: int):
        if "style_missing" not in self.df.columns:
            raise ValueError("df에 'style_missing' 컬럼이 필요합니다. (0=style有, 1=style無)")

        mask = (self.df["style_missing"] == 0)
        df_taste_raw = self.df.loc[mask].dropna(subset=self.style_cols).copy()

        if df_taste_raw.empty:
            self.df_taste = df_taste_raw
            self.scaler_taste = None
            self.fa_model = None
            self.knn_taste = None
            self._taste_index = np.array([], dtype=int)
            return

        # 1) 표준화
        scaler = StandardScaler()
        X_std = scaler.fit_transform(df_taste_raw[self.style_cols])

        # 2) Factor Analysis
        fa = FactorAnalyzer(n_factors=2, rotation="varimax")
        fa.fit(X_std)
        scores = fa.transform(X_std)

        df_taste_raw["FA1"] = scores[:, 0]
        df_taste_raw["FA2"] = scores[:, 1]

        # 전체 df 에 붙이기
        self.df["FA1"] = np.nan
        self.df["FA2"] = np.nan
        self.df.loc[df_taste_raw.index, "FA1"] = df_taste_raw["FA1"]
        self.df.loc[df_taste_raw.index, "FA2"] = df_taste_raw["FA2"]

        self.df_taste = df_taste_raw
        self.scaler_taste = scaler
        self.fa_model = fa

        # 3) KNN fitting
        X_taste = df_taste_raw[["FA1", "FA2"]].values
        n_neighbors = min(n_neighbors, len(df_taste_raw))

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(X_taste)

        self.knn_taste = knn
        self._taste_index = df_taste_raw.index.to_numpy()

    # --------------------------------------------------------- #
    #                 META SPACE (ABV 제거 버전)                 #
    # --------------------------------------------------------- #
    def _build_meta_space(self, n_neighbors: int):

        df_meta = self.df.copy()

        # --------------------
        # 1) age_bin
        # --------------------
        def age_to_bin(x):
            if pd.isna(x):
                return "Unknown"
            x = int(x)
            if x <= 8:
                return "0-8"
            elif x <= 12:
                return "9-12"
            elif x <= 16:
                return "13-16"
            elif x <= 21:
                return "17-21"
            else:
                return "22+"

        if "age" in df_meta.columns:
            df_meta["age_bin"] = df_meta["age"].apply(age_to_bin)
        else:
            df_meta["age_bin"] = "Unknown"

        # --------------------
        # 2) vintage_flag
        # --------------------
        if "is_vintage" in df_meta.columns:
            df_meta["vintage_flag"] = df_meta["is_vintage"].astype(int)
        elif "vintage" in df_meta.columns:
            df_meta["vintage_flag"] = np.where(df_meta["vintage"].notna(), 1, 0)
        else:
            df_meta["vintage_flag"] = 0

        # --------------------
        # 3) price_bin
        # --------------------
        def price_to_bin(x):
            if pd.isna(x):
                return "Unknown"
            x = float(x)
            if x < 40:
                return "low"
            elif x < 80:
                return "mid"
            elif x < 150:
                return "high"
            elif x < 300:
                return "premium"
            else:
                return "luxury"

        if "price(£)" in df_meta.columns:
            df_meta["price_bin"] = df_meta["price(£)"].apply(price_to_bin)
        else:
            df_meta["price_bin"] = "Unknown"

        # 🔴 원본 df에도 같이 붙여둠 (추천 결과에서 바로 보이도록)
        self.df["age_bin"] = df_meta["age_bin"]
        self.df["price_bin"] = df_meta["price_bin"]
        self.df["vintage_flag"] = df_meta["vintage_flag"]

        # --------------------
        # 4) categorical / numeric feature 리스트
        # --------------------
        cat_features = [
            "whisky_type",
            "region",
            "bottler_group",
            "cask_group",
            "age_bin",
            "price_bin",
        ]
        num_features = ["vintage_flag"]

        # 결측값 처리 (카테고리는 Unknown)
        for col in ["whisky_type", "region", "bottler_group", "cask_group"]:
            if col not in df_meta.columns:
                df_meta[col] = "Unknown"
            else:
                df_meta[col] = df_meta[col].fillna("Unknown")

        # 저장 (설명용)
        self.meta_cat_features = cat_features
        self.meta_num_features = num_features

        # --------------------
        # 5) OneHot + Standardize
        # --------------------
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                ("num", StandardScaler(), num_features),
            ]
        )

        X_meta_raw = preprocessor.fit_transform(df_meta[cat_features + num_features])

        # --------------------
        # 6) 가중치 적용 (type > region > bottler > cask)
        # --------------------
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(cat_features)

        n_cat = len(cat_names)
        X_cat = X_meta_raw[:, :n_cat]
        X_num = X_meta_raw[:, n_cat:]

        X_cat = X_cat.toarray() if hasattr(X_cat, "toarray") else X_cat
        X_num = X_num.toarray() if hasattr(X_num, "toarray") else X_num

        def idx(prefix):
            return [i for i, name in enumerate(cat_names) if name.startswith(prefix + "_")]

        idx_type = idx("whisky_type")
        idx_region = idx("region")
        idx_bottler = idx("bottler_group")
        idx_cask = idx("cask_group")

        if idx_type:
            X_cat[:, idx_type] *= 3.0
        if idx_region:
            X_cat[:, idx_region] *= 2.0
        if idx_bottler:
            X_cat[:, idx_bottler] *= 1.5
        if idx_cask:
            X_cat[:, idx_cask] *= 1.0

        X_meta = np.hstack([X_cat, X_num])

        # save
        self.df_meta = df_meta
        self.X_meta = X_meta
        self.preprocessor_meta = preprocessor
        self._meta_cat_names = cat_names
        self._meta_n_cat = n_cat

        n_neighbors = min(n_neighbors, X_meta.shape[0])
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(X_meta)

        self.knn_meta = knn
        self._meta_index = df_meta.index.to_numpy()

    # --------------------------------------------------------- #
    #                       UTILITIES                           #
    # --------------------------------------------------------- #
    def _get_index_by_name(self, name: str):
        matches = self.df.index[self.df["name"] == name]
        if len(matches) == 0:
            raise ValueError(f"name '{name}' not found.")
        return matches[0]

    def _make_style_from_levels(self, body_level, rich_level, smoke_level, sweet_level):
        body_map = {"light": 1, "medium": 2, "full": 3}
        rich_map = {"lean": 1, "round": 2, "rich": 3}
        smoke_map = {"non": 1, "light-smoky": 2, "smoky": 3}
        sweet_map = {"dry": 1, "balanced": 2, "sweet": 3}

        return np.array([
            body_map[body_level],
            rich_map[rich_level],
            smoke_map[smoke_level],
            sweet_map[sweet_level],
        ], dtype=float)

    def _project_style_to_factor(self, style_vec):
        if self.scaler_taste is None or self.fa_model is None:
            raise RuntimeError("taste space가 구성되어 있지 않습니다.")
        v_std = self.scaler_taste.transform(style_vec.reshape(1, -1))
        factor = self.fa_model.transform(v_std)
        return factor[0]  # (FA1, FA2)

    def _aggregate_product_points(self, product_list):
        """
        입력 제품 리스트 기준:
        - taste_center: FA1, FA2 평균 (style_missing=0인 애들만)
        - meta_center : meta embedding 평균
        """
        idx_list = [self._get_index_by_name(n) for n in product_list]

        # taste center
        idx_taste = [i for i in idx_list if i in self.df_taste.index]
        taste_center = None
        if idx_taste:
            taste_center = self.df_taste.loc[idx_taste, ["FA1", "FA2"]].mean().values

        # meta center
        pos = [np.where(self._meta_index == i)[0][0] for i in idx_list]
        meta_center = self.X_meta[pos].mean(axis=0)

        return taste_center, meta_center

    def _knn_taste_point(self, point, top_k):
        if self.knn_taste is None:
            raise RuntimeError("taste KNN이 구성되어 있지 않습니다.")

        n = min(top_k, len(self._taste_index))
        dists, idxs = self.knn_taste.kneighbors(point.reshape(1, -1), n_neighbors=n)
        reals = self._taste_index[idxs[0]]

        df_out = self.df.loc[reals].copy()
        df_out["taste_distance"] = dists[0]
        return df_out

    def _knn_meta_point(self, point, top_k):
        if self.knn_meta is None:
            raise RuntimeError("meta KNN이 구성되어 있지 않습니다.")

        n = min(top_k, self.X_meta.shape[0])
        dists, idxs = self.knn_meta.kneighbors(point.reshape(1, -1), n_neighbors=n)
        reals = self._meta_index[idxs[0]]

        df_out = self.df.loc[reals].copy()
        df_out["meta_distance"] = dists[0]
        return df_out

    def _filter_by_family(self, df_in, selected_families):
        if not selected_families:
            return df_in

        fam_cols = []
        for fam in selected_families:
            col = f"{fam}_family"
            if col in df_in.columns:
                fam_cols.append(col)
        if not fam_cols:
            return df_in

        mask = np.zeros(len(df_in), dtype=bool)
        for col in fam_cols:
            mask |= (df_in[col] == 1)

        return df_in.loc[mask].copy()

    # ---------------- special 분리 ---------------- #
    def _split_special_items(self, df_in):
        """
        빈티지 / 고연산 / 고가 / high rarity 제품을 별도 리스트로 분리
        """
        if df_in is None or df_in.empty:
            return df_in, df_in

        df = df_in.copy()
        idx = df.index

        is_vintage = df.get("is_vintage", pd.Series(0, index=idx))
        age = df.get("age", pd.Series(np.nan, index=idx))
        rarity = df.get("rarity_score", pd.Series(-np.inf, index=idx))
        price_bin = df.get("price_bin", pd.Series("Unknown", index=idx))

        special_mask = (
            (is_vintage == 1)
            | (age >= 21)
            | (rarity >= 3)
            | (price_bin.isin(["premium", "luxury"]))
        )

        return df[~special_mask].copy(), df[special_mask].copy()

    # ---------------- meta 설명 생성 ---------------- #
    def _explain_meta_center(self, meta_center_vec):
        """
        meta_center 벡터를 바탕으로
        '어떤 메타 공통점 때문에 추천되었는지' 설명 문자열 생성
        """
        if meta_center_vec is None:
            return ""

        cat_names = self._meta_cat_names
        n_cat = self._meta_n_cat

        X_cat_center = meta_center_vec[:n_cat]
        X_num_center = meta_center_vec[n_cat:]

        explanation = {}

        for feature in ["whisky_type", "region", "bottler_group",
                        "cask_group", "age_bin", "price_bin"]:
            cols = [
                i for i, name in enumerate(cat_names)
                if name.startswith(feature + "_")
            ]
            if not cols:
                continue
            best_idx = cols[np.argmax(X_cat_center[cols])]
            value = cat_names[best_idx].split(feature + "_", 1)[1]
            explanation[feature] = value

        # vintage_flag (numeric)
        if len(X_num_center) > 0:
            explanation["vintage_flag"] = 1 if X_num_center[0] > 0 else 0
        else:
            explanation["vintage_flag"] = 0

        # 자연어 텍스트
        lines = ["이 메타 추천은 아래 공통 특성을 기준으로 선정되었습니다:"]
        if "whisky_type" in explanation:
            lines.append(f"- Whisky Type: **{explanation['whisky_type']}**")
        if "region" in explanation:
            lines.append(f"- Region: **{explanation['region']}**")
        if "bottler_group" in explanation:
            lines.append(f"- Bottler Group: **{explanation['bottler_group']}**")
        if "cask_group" in explanation:
            lines.append(f"- Cask Group: **{explanation['cask_group']}**")
        if "age_bin" in explanation:
            lines.append(f"- Age Range: **{explanation['age_bin']}**")
        if "price_bin" in explanation:
            lines.append(f"- Price Tier: **{explanation['price_bin']}**")
        if explanation.get("vintage_flag", 0) == 1:
            lines.append(f"- Vintage: **Yes**")

        return "\n".join(lines)

    # --------------------------------------------------------- #
    #                       SCENARIO A                          #
    # --------------------------------------------------------- #
    def recommend_from_style_products(
        self,
        product_list,
        selected_families=None,
        top_k_taste=20,
        top_k_meta=20,
    ):
        if not product_list:
            raise ValueError("product_list가 비어 있습니다.")

        taste_center, meta_center = self._aggregate_product_points(product_list)
        if taste_center is None:
            raise ValueError("선택한 제품 중 style_missing=0 인 제품이 없습니다.")

        taste_df = self._knn_taste_point(taste_center, top_k_taste)
        meta_df = self._knn_meta_point(meta_center, top_k_meta)

        taste_fam_df = self._filter_by_family(taste_df, selected_families)

        # special split
        taste_main, taste_special = self._split_special_items(taste_df)
        meta_main, meta_special = self._split_special_items(meta_df)
        fam_main, fam_special = self._split_special_items(taste_fam_df)

        # 설명 문구
        taste_explanation = (
            "입력하신 제품들의 FA taste 위치(Body·Richness·Smoke·Sweetness 요인 점수)를 평균 내고, "
            "이와 가까운 순서대로 비슷한 맛의 위스키를 추천했습니다."
        )
        meta_explanation = self._explain_meta_center(meta_center)

        return {
            "mode": "product_style_yes",
            "input_products": product_list,
            "taste_explanation": taste_explanation,
            "meta_explanation": meta_explanation,
            "taste_based_main": taste_main.reset_index(drop=True),
            "taste_based_special": taste_special.reset_index(drop=True),
            "meta_based_main": meta_main.reset_index(drop=True),
            "meta_based_special": meta_special.reset_index(drop=True),
            "taste_with_family_main": fam_main.reset_index(drop=True),
            "taste_with_family_special": fam_special.reset_index(drop=True),
        }

    # --------------------------------------------------------- #
    #                       SCENARIO B                          #
    # --------------------------------------------------------- #
    def recommend_from_style_missing_products(
        self,
        product_list,
        top_k_meta=20,
    ):
        if not product_list:
            raise ValueError("product_list가 비어 있습니다.")

        _, meta_center = self._aggregate_product_points(product_list)
        meta_df = self._knn_meta_point(meta_center, top_k_meta)

        meta_main, meta_special = self._split_special_items(meta_df)

        has_style = meta_df[meta_df["style_missing"] == 0]
        taste_summary = {}
        if not has_style.empty:
            taste_summary = {
                "FA1_mean": float(has_style["FA1"].mean()),
                "FA2_mean": float(has_style["FA2"].mean()),
                "count": int(len(has_style)),
            }

        meta_explanation = self._explain_meta_center(meta_center)

        return {
            "mode": "product_style_no",
            "input_products": product_list,
            "meta_explanation": meta_explanation,
            "meta_based_main": meta_main.reset_index(drop=True),
            "meta_based_special": meta_special.reset_index(drop=True),
            "taste_summary": taste_summary,
        }

    # --------------------------------------------------------- #
    #                       SCENARIO C                          #
    # --------------------------------------------------------- #
    def recommend_from_survey(
        self,
        body_level="medium",
        rich_level="round",
        smoke_level="non",
        sweet_level="balanced",
        selected_families=None,
        top_k_taste=20,
        top_k_meta=20,
    ):
        # 스타일 벡터 → FA space
        style_vec = self._make_style_from_levels(
            body_level, rich_level, smoke_level, sweet_level
        )
        user_factor = self._project_style_to_factor(style_vec)

        taste_df = self._knn_taste_point(user_factor, top_k_taste)
        taste_fam_df = self._filter_by_family(taste_df, selected_families)

        # meta expansion (taste neighbor의 meta center)
        meta_df = pd.DataFrame()
        meta_explanation = ""
        if not taste_df.empty:
            idxs = taste_df.index
            pos = [np.where(self._meta_index == i)[0][0] for i in idxs]
            meta_center = self.X_meta[pos].mean(axis=0)
            meta_df = self._knn_meta_point(meta_center, top_k_meta)
            meta_explanation = self._explain_meta_center(meta_center)

        # special split
        taste_main, taste_special = self._split_special_items(taste_df)
        fam_main, fam_special = self._split_special_items(taste_fam_df)

        taste_explanation = (
            "선호 맛(Body: {b}, Rich: {r}, Smoke: {s}, Sweet: {sw})을 FA taste space로 투영하고, "
            "그 위치와 가까운 순서대로 위스키를 추천했습니다."
        ).format(b=body_level, r=rich_level, s=smoke_level, sw=sweet_level)

        return {
            "mode": "survey_only",
            "input_flavour": {
                "body": body_level,
                "rich": rich_level,
                "smoke": smoke_level,
                "sweet": sweet_level,
                "families": selected_families or [],
            },
            "taste_explanation": taste_explanation,
            "meta_explanation": meta_explanation,
            "taste_only_main": taste_main.reset_index(drop=True),
            "taste_only_special": taste_special.reset_index(drop=True),
            "taste_with_family_main": fam_main.reset_index(drop=True),
            "taste_with_family_special": fam_special.reset_index(drop=True),
            "meta_expansion": meta_df.reset_index(drop=True),
        }

    # --------------------------------------------------------- #
    #                      MASTER WRAPPER                       #
    # --------------------------------------------------------- #
    def recommend(
        self,
        product_list=None,
        body_level=None,
        rich_level=None,
        smoke_level=None,
        sweet_level=None,
        selected_families=None,
        top_k_taste=20,
        top_k_meta=20,
    ):
        product_list = product_list or []
        has_products = len(product_list) > 0
        has_survey = any([body_level, rich_level, smoke_level, sweet_level, selected_families])

        # Scenario A/B: 제품만 입력
        if has_products and not has_survey:
            idxs = [self._get_index_by_name(n) for n in product_list]
            if any(self.df.loc[idxs, "style_missing"] == 0):
                return self.recommend_from_style_products(
                    product_list, selected_families, top_k_taste, top_k_meta
                )
            else:
                return self.recommend_from_style_missing_products(
                    product_list, top_k_meta
                )

        # Scenario C: 설문만 입력
        if has_survey and not has_products:
            return self.recommend_from_survey(
                body_level or "medium",
                rich_level or "round",
                smoke_level or "non",
                sweet_level or "balanced",
                selected_families,
                top_k_taste,
                top_k_meta,
            )

        # 제품 + 설문 둘 다 있는 경우
        if has_products and has_survey:
            return {
                "mode": "product_and_survey",
                "product_based": self.recommend_from_style_products(
                    product_list, selected_families, top_k_taste, top_k_meta
                ),
                "survey_based": self.recommend_from_survey(
                    body_level or "medium",
                    rich_level or "round",
                    smoke_level or "non",
                    sweet_level or "balanced",
                    selected_families,
                    top_k_taste,
                    top_k_meta,
                ),
            }

        raise ValueError("product_list 또는 flavour 입력 중 최소 하나는 필요합니다.")
