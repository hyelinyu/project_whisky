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
    """

    # --------------------------------------------------------- #
    # INIT
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

        # taste / meta space 구성
        self._build_taste_space(n_taste_neighbors)
        self._build_meta_space(n_meta_neighbors)

    # --------------------------------------------------------- #
    # TASTE SPACE (FA)
    # --------------------------------------------------------- #
    def _build_taste_space(self, n_neighbors: int):
        if "style_missing" not in self.df.columns:
            raise ValueError("df에 'style_missing' 컬럼이 필요합니다.")

        mask = self.df["style_missing"] == 0
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

        # 2) FA
        fa = FactorAnalyzer(n_factors=2, rotation="varimax")
        fa.fit(X_std)
        scores = fa.transform(X_std)

        df_taste_raw["FA1"] = scores[:, 0]
        df_taste_raw["FA2"] = scores[:, 1]

        # 전체 DF에 반영
        self.df["FA1"] = np.nan
        self.df["FA2"] = np.nan
        self.df.loc[df_taste_raw.index, ["FA1", "FA2"]] = df_taste_raw[["FA1", "FA2"]]

        # 저장
        self.df_taste = df_taste_raw
        self.scaler_taste = scaler
        self.fa_model = fa

        # KNN
        X_taste = df_taste_raw[["FA1", "FA2"]].values
        n_neighbors = min(n_neighbors, len(df_taste_raw))

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(X_taste)

        self.knn_taste = knn
        self._taste_index = df_taste_raw.index.to_numpy()

    # --------------------------------------------------------- #
    # META SPACE
    # --------------------------------------------------------- #
    def _build_meta_space(self, n_neighbors: int):
        df_meta = self.df.copy()

        # ---------------- age_bin ----------------
        def age_to_bin(x):
            if pd.isna(x):
                return "Unknown"
            x = int(x)
            if x <= 8: return "0-8"
            elif x <= 12: return "9-12"
            elif x <= 16: return "13-16"
            elif x <= 21: return "17-21"
            else: return "22+"

        df_meta["age_bin"] = df_meta.get("age", np.nan).apply(age_to_bin)

        # ---------------- vintage_flag ----------------
        if "is_vintage" in df_meta.columns:
            df_meta["vintage_flag"] = df_meta["is_vintage"].astype(int)
        elif "vintage" in df_meta.columns:
            df_meta["vintage_flag"] = np.where(df_meta["vintage"].notna(), 1, 0)
        else:
            df_meta["vintage_flag"] = 0

        # ---------------- price_bin ----------------
        def price_to_bin(x):
            if pd.isna(x): return "Unknown"
            x = float(x)
            if x < 40: return "low"
            elif x < 80: return "mid"
            elif x < 150: return "high"
            elif x < 300: return "premium"
            else: return "luxury"

        df_meta["price_bin"] = df_meta.get("price(£)", np.nan).apply(price_to_bin)

        # 원본 df에도 붙이기
        self.df["age_bin"] = df_meta["age_bin"]
        self.df["price_bin"] = df_meta["price_bin"]
        self.df["vintage_flag"] = df_meta["vintage_flag"]

        # ---------------- features ----------------
        cat_features = [
            "whisky_type", "region", "bottler_group",
            "cask_group", "age_bin", "price_bin",
        ]
        num_features = ["vintage_flag"]

        # 카테고리 결측 Unknown
        for col in ["whisky_type", "region", "bottler_group", "cask_group"]:
            df_meta[col] = df_meta.get(col, "Unknown").fillna("Unknown")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                ("num", StandardScaler(), num_features),
            ]
        )

        X_meta_raw = preprocessor.fit_transform(df_meta[cat_features + num_features])

        # one-hot & numeric split
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(cat_features)

        n_cat = len(cat_names)
        X_cat = X_meta_raw[:, :n_cat]
        X_num = X_meta_raw[:, n_cat:]

        # densify
        X_cat = X_cat.toarray() if hasattr(X_cat, "toarray") else X_cat
        X_num = X_num.toarray() if hasattr(X_num, "toarray") else X_num

        # ---------------- 가중치 ----------------
        def idx(prefix):
            return [i for i, name in enumerate(cat_names) if name.startswith(prefix + "_")]

        for prefix, weight in [
            ("whisky_type", 3.0),
            ("region", 2.0),
            ("bottler_group", 1.5),
            ("cask_group", 1.0),
        ]:
            cols = idx(prefix)
            if cols:
                X_cat[:, cols] *= weight

        X_meta = np.hstack([X_cat, X_num])

        # 저장
        self.df_meta = df_meta
        self.X_meta = X_meta
        self.preprocessor_meta = preprocessor
        self._meta_cat_names = cat_names
        self._meta_n_cat = n_cat

        # KNN
        n_neighbors = min(n_neighbors, X_meta.shape[0])
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(X_meta)
        self.knn_meta = knn
        self._meta_index = df_meta.index.to_numpy()

    # --------------------------------------------------------- #
    # UTILITIES
    # --------------------------------------------------------- #
    def _get_index_by_name(self, name: str):
        matches = self.df.index[self.df["name"] == name]
        if len(matches) == 0:
            raise ValueError(f"name '{name}' not found.")
        return matches[0]

    # ---------------- style level → numeric ----------------
    def _make_style_from_levels(self, body_level, rich_level, smoke_level, sweet_level):

        body_map = {"light": 1.0, "medium": 3.0, "full": 5.0}
        rich_map = {"lean": 1.0, "round": 3.0, "rich": 5.0}
        smoke_map = {"non": 0.0, "light-smoky": 1.5, "smoky": 4.0}
        sweet_map = {"dry": 1.0, "balanced": 3.0, "sweet": 5.0}

        return np.array([
            body_map[body_level],
            rich_map[rich_level],
            smoke_map[smoke_level],
            sweet_map[sweet_level],
        ], dtype=float)

    # ---------------- style → FA factor ----------------
    def _project_style_to_factor(self, style_vec):
        v_std = self.scaler_taste.transform(style_vec.reshape(1, -1))
        factor = self.fa_model.transform(v_std)
        return factor[0]

    # ---------------- per-product KNN ----------------
    def _knn_taste_point(self, point, top_k):
        n = min(top_k, len(self._taste_index))
        dists, idxs = self.knn_taste.kneighbors(point.reshape(1, -1), n_neighbors=n)
        reals = self._taste_index[idxs[0]]
        df_out = self.df.loc[reals].copy()
        df_out["taste_distance"] = dists[0]
        return df_out

    def _knn_meta_point(self, point, top_k):
        n = min(top_k, self.X_meta.shape[0])
        dists, idxs = self.knn_meta.kneighbors(point.reshape(1, -1), n_neighbors=n)
        reals = self._meta_index[idxs[0]]
        df_out = self.df.loc[reals].copy()
        df_out["meta_distance"] = dists[0]
        return df_out

    # ---------------- family filter ----------------
    def _filter_by_family(self, df_in, selected_families):
        if not selected_families:
            return df_in

        fam_cols = [f"{fam}_family" for fam in selected_families if f"{fam}_family" in df_in.columns]
        if not fam_cols:
            return df_in

        mask = np.zeros(len(df_in), dtype=bool)
        for col in fam_cols:
            mask |= df_in[col] == 1

        return df_in.loc[mask]

    # ---------------- special 분리 ----------------
    def _split_special_items(self, df_in):
        if df_in is None or df_in.empty:
            return df_in, df_in

        df = df_in.copy()
        idx = df.index

        is_vintage = df.get("is_vintage", pd.Series(0, index=idx))
        age = df.get("age", pd.Series(np.nan, index=idx))
        rarity = df.get("rarity_score", pd.Series(-np.inf, index=idx))
        price_bin = df.get("price_bin", pd.Series("Unknown", index=idx))

        special_mask = (
            (is_vintage == 1) |
            (age >= 21) |
            (rarity >= 3) |
            (price_bin.isin(["premium", "luxury"]))
        )

        return df[~special_mask], df[special_mask]

    # ---------------- aggregate product points ----------------
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

    # ---------------- meta 설명 생성 ----------------
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
    # NEW: ROUND-ROBIN SELECTION LOGIC
    # --------------------------------------------------------- #
    def _select_topk_round_robin(self, neighbor_lists, k=5):
        """
        neighbor_lists = {
            "제품명1": df_neighbors_sorted,
            "제품명2": df_neighbors_sorted,
            ...
        }
        - 각 df는 distance 기준으로 정렬되어 있어야 함
        - round 1 → 각 제품의 1등
        - round 2 → 각 제품의 2등 …
        - 중복 제거 후 거리순으로 채워 넣기
        """
        result = []
        used = set()
        round_idx = 0

        max_len = max(len(df) for df in neighbor_lists.values())

        while len(result) < k and round_idx < max_len:
            candidates = []

            for prod, df in neighbor_lists.items():
                if round_idx < len(df):
                    row = df.iloc[round_idx]
                    name = row["name"]
                    dist = row.get("taste_distance", row.get("meta_distance", None))

                    if name not in used:
                        candidates.append((name, dist, row))

            candidates.sort(key=lambda x: x[1] if x[1] is not None else 999)

            for name, dist, row in candidates:
                if len(result) >= k:
                    break
                if name not in used:
                    result.append(row)
                    used.add(name)

            round_idx += 1

        if not result:
            return pd.DataFrame()

        return pd.DataFrame(result).reset_index(drop=True)

    # --------------------------------------------------------- #
    # SCENARIO A: 스타일 있는 제품 기준
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

        # taste center (쓰지 않음 — 개별 기준 사용)
        taste_center, meta_center = self._aggregate_product_points(product_list)

        # --------- 개별 product basis neighbor lists 생성 ---------
        neighbor_lists = {}
        for p in product_list:
            idx = self._get_index_by_name(p)
            if idx in self.df_taste.index:
                point = self.df.loc[idx, ["FA1", "FA2"]].values
                df_single = self._knn_taste_point(point, top_k_taste)
                df_single = df_single[df_single["name"] != p].reset_index(drop=True)
                neighbor_lists[p] = df_single

        # Round-robin 방식 추천 5개
        taste_main = self._select_topk_round_robin(neighbor_lists, k=5)
        taste_main, taste_special = self._split_special_items(taste_main)

        # 메타는 기존 평균 center 기반 (변경 없음)
        meta_df = self._knn_meta_point(meta_center, top_k_meta)
        meta_main, meta_special = self._split_special_items(meta_df)

        # family filter
        taste_fam = self._filter_by_family(taste_main, selected_families)
        fam_main, fam_special = self._split_special_items(taste_fam)

        # 설명
        taste_explanation = (
            "여러 제품 각각의 FA taste 위치를 기준으로 KNN을 수행한 뒤, "
            "각 제품에서 가장 가까운 이웃을 먼저 배분하는 라운드로빈 방식으로 추천했습니다."
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
    # SCENARIO B: 스타일 없는 제품 기준 (메타 기반)
    # --------------------------------------------------------- #
    def recommend_from_style_missing_products(
        self,
        product_list,
        top_k_meta=20,
    ):
        if not product_list:
            raise ValueError("product_list가 비어 있습니다.")

        _, meta_center = self._aggregate_product_points(product_list)

        # --------- 개별 meta neighbor lists 생성 ---------
        neighbor_lists = {}
        for p in product_list:
            idx = self._get_index_by_name(p)
            pos = np.where(self._meta_index == idx)[0]
            if len(pos) > 0:
                point = self.X_meta[pos[0]]
                df_single = self._knn_meta_point(point, top_k_meta)
                df_single = df_single[df_single["name"] != p].reset_index(drop=True)
                neighbor_lists[p] = df_single

        # round-robin top5
        meta_main = self._select_topk_round_robin(neighbor_lists, k=5)
        meta_main, meta_special = self._split_special_items(meta_main)

        # taste summary
        has_style = meta_main[meta_main["style_missing"] == 0]
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
    # SCENARIO C (설문)
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
        style_vec = self._make_style_from_levels(
            body_level, rich_level, smoke_level, sweet_level
        )
        user_factor = self._project_style_to_factor(style_vec)

        taste_df = self._knn_taste_point(user_factor, top_k_taste)
        taste_fam_df = self._filter_by_family(taste_df, selected_families)

        # meta expansion
        meta_df = pd.DataFrame()
        meta_explanation = ""
        if not taste_df.empty:
            idxs = taste_df.index
            pos = [np.where(self._meta_index == i)[0][0] for i in idxs]
            meta_center = self.X_meta[pos].mean(axis=0)
            meta_df = self._knn_meta_point(meta_center, top_k_meta)
            meta_explanation = self._explain_meta_center(meta_center)

        taste_main, taste_special = self._split_special_items(taste_df)
        fam_main, fam_special = self._split_special_items(taste_fam_df)

        taste_explanation = (
            f"선호 맛(Body: {body_level}, Rich: {rich_level}, "
            f"Smoke: {smoke_level}, Sweet: {sweet_level})을 기반으로 "
            "FA taste space에서 가까운 위스키를 추천했습니다."
        )

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
    # MASTER WRAPPER
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

        # 제품 기반
        if has_products and not has_survey:
            idxs = [self._get_index_by_name(n) for n in product_list]
            # 스타일 있는 제품이 하나라도 있으면 Scenario A
            if any(self.df.loc[idxs, "style_missing"] == 0):
                return self.recommend_from_style_products(
                    product_list, selected_families, top_k_taste, top_k_meta
                )
            # 전부 style missing → Scenario B
            else:
                return self.recommend_from_style_missing_products(
                    product_list, top_k_meta
                )

        # 설문만 입력된 경우
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

        # 제품 + 설문 모두 있는 경우 (두 결과 병렬 반환)
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
