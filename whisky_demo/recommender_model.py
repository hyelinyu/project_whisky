
import numpy as np
import pandas as pd

from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors


class WhiskyRecommender:
    """
    Whisky Recommender (Taste: FA + Partial-Style, Meta: OneHot KNN)

    + Rare Pool (희귀 제품군) 기능
    ---------------------------
    - 희귀 제품군은 고정된 규칙으로 정의한다. (rare pool)
    - 고객의 meta 취향(meta_center)에 따라 rare pool 안에서만 개인화 추천을 별도로 제공한다.
    - 메인 추천 결과와는 "별개의 작업"으로 결과 dict에 따로 붙인다.

  
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

        # ✅ 항상 컬럼 존재 보장
        if "FA1" not in self.df.columns:
            self.df["FA1"] = np.nan
        if "FA2" not in self.df.columns:
            self.df["FA2"] = np.nan

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

            self._Z_style = None
            self._style_means_ = None
            self._style_scales_ = None
            return

        scaler = StandardScaler()
        X_std = scaler.fit_transform(df_taste_raw[self.style_cols])

        fa = FactorAnalyzer(n_factors=2, rotation="varimax")
        fa.fit(X_std)
        scores = fa.transform(X_std)

        # ✅ 방어 체크 (혹시라도 scores shape 이상하면 바로 터뜨리기)
        if scores is None or scores.shape[1] < 2:
            raise ValueError(
                f"FA transform 결과가 이상합니다. scores shape={None if scores is None else scores.shape}"
            )

        df_taste_raw["FA1"] = scores[:, 0]
        df_taste_raw["FA2"] = scores[:, 1]

        # self.df에 반영
        self.df.loc[df_taste_raw.index, ["FA1", "FA2"]] = df_taste_raw[["FA1", "FA2"]]

        self.df_taste = df_taste_raw
        self.scaler_taste = scaler
        self.fa_model = fa

        # ✅ 여기서도 혹시 없으면 명확히 에러
        if not set(["FA1", "FA2"]).issubset(self.df_taste.columns):
            raise ValueError("df_taste_raw에 FA1/FA2 생성이 되지 않았습니다.")

        X_taste = self.df_taste[["FA1", "FA2"]].values
        n_neighbors = min(n_neighbors, len(self.df_taste))

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(X_taste)

        self.knn_taste = knn
        self._taste_index = self.df_taste.index.to_numpy()

        self._Z_style = X_std
        self._style_means_ = scaler.mean_.copy()
        self._style_scales_ = scaler.scale_.copy()

    # --------------------------------------------------------- #
    # META SPACE (카테고리컬로 통합함)
    # --------------------------------------------------------- #
    def _build_meta_space(self, n_neighbors: int):
        df_meta = self.df.copy()

        def age_to_bin(x):
            if pd.isna(x):
                return "Unknown"
            try:
                x = int(float(x))
            except Exception:
                return "Unknown"
            if x <= 3:
                return "0-3"
            elif x <= 13:
                return "4-13"
            elif x <= 16:
                return "14-16"
            elif x <= 20:
                return "17-20"
            else:
                return "21+"

        df_meta["age_bin"] = df_meta.get("age", np.nan).apply(age_to_bin)

        def price_to_bin(x):
            if pd.isna(x):
                return "Unknown"
            try:
                x = float(x)
            except Exception:
                return "Unknown"
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

        df_meta["price_bin"] = df_meta.get("price(£)", np.nan).apply(price_to_bin)

        self.df["age_bin"] = df_meta["age_bin"]
        self.df["price_bin"] = df_meta["price_bin"]

        cat_features = [
            "whisky_type",
            "region",
            "bottler_group",
            "age_bin",
            "cask_group",
            "price_bin",
        ]

        for col in cat_features:
            df_meta[col] = df_meta.get(col, "Unknown").fillna("Unknown")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
            remainder="drop",
        )

        X_meta_raw = preprocessor.fit_transform(df_meta[cat_features])
        X_cat = X_meta_raw.toarray() if hasattr(X_meta_raw, "toarray") else X_meta_raw

        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(cat_features)

        def idx(prefix):
            return [i for i, name in enumerate(cat_names) if name.startswith(prefix + "_")]

        weights = {
            "whisky_type": 3.0,
            "region": 2.5,
            "bottler_group": 2.0,
            "age_bin": 1.5,
            "cask_group": 1.2,
            "price_bin": 1.0,
        }

        for prefix, weight in weights.items():
            cols = idx(prefix)
            if cols:
                X_cat[:, cols] *= weight

        self.df_meta = df_meta
        self.X_meta = X_cat
        self.preprocessor_meta = preprocessor
        self._meta_cat_names = cat_names
        self._meta_n_cat = len(cat_names)

        n_neighbors = min(n_neighbors, self.X_meta.shape[0])
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(self.X_meta)
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

    def _project_style_to_factor(self, style_vec):
        if self.scaler_taste is None or self.fa_model is None:
            raise ValueError("FA taste space가 준비되지 않았습니다.")
        v_std = self.scaler_taste.transform(style_vec.reshape(1, -1))
        factor = self.fa_model.transform(v_std)
        return factor[0]

    def _knn_taste_point(self, point, top_k):
        if self.knn_taste is None or len(self._taste_index) == 0:
            return pd.DataFrame()
        n = min(top_k, len(self._taste_index))
        dists, idxs = self.knn_taste.kneighbors(point.reshape(1, -1), n_neighbors=n)
        reals = self._taste_index[idxs[0]]
        df_out = self.df.loc[reals].copy()
        df_out["taste_distance"] = dists[0]
        return df_out

    def _knn_meta_point(self, point, top_k):
        if self.knn_meta is None or self.X_meta is None:
            return pd.DataFrame()
        n = min(top_k, self.X_meta.shape[0])
        dists, idxs = self.knn_meta.kneighbors(point.reshape(1, -1), n_neighbors=n)
        reals = self._meta_index[idxs[0]]
        df_out = self.df.loc[reals].copy()
        df_out["meta_distance"] = dists[0]
        return df_out

    def _add_family_match_score(self, df_in, selected_families):
        if df_in is None or df_in.empty:
            return df_in
        df = df_in.copy()
        if not selected_families:
            df["family_match"] = 0
            return df
        fam_cols = [f"{fam}_family" for fam in selected_families if f"{fam}_family" in df.columns]
        if not fam_cols:
            df["family_match"] = 0
            return df
        df["family_match"] = df[fam_cols].fillna(0).sum(axis=1).astype(int)
        return df

    def _taste_candidates_from_partial_style(
        self,
        style_body=None,
        style_richness=None,
        style_smoke=None,
        style_sweetness=None,
        top_k=80,
    ):
        if (
            self.df_taste is None
            or self.df_taste.empty
            or self._Z_style is None
            or self._style_means_ is None
            or self._style_scales_ is None
        ):
            return pd.DataFrame()

        vals = [style_body, style_richness, style_smoke, style_sweetness]
        provided = np.array([v is not None for v in vals], dtype=bool)
        if provided.sum() == 0:
            return pd.DataFrame()

        v = np.array(
            [
                style_body if style_body is not None else self._style_means_[0],
                style_richness if style_richness is not None else self._style_means_[1],
                style_smoke if style_smoke is not None else self._style_means_[2],
                style_sweetness if style_sweetness is not None else self._style_means_[3],
            ],
            dtype=float,
        )

        z_u = (v - self._style_means_) / self._style_scales_
        Z = self._Z_style

        diff = Z[:, provided] - z_u[provided]
        dists = np.sqrt((diff**2).sum(axis=1))

        n = min(top_k, len(self.df_taste))
        top_idx = np.argpartition(dists, n - 1)[:n]
        top_idx = top_idx[np.argsort(dists[top_idx])]

        real_index = self._taste_index[top_idx]
        df_out = self.df.loc[real_index].copy()
        df_out["taste_distance"] = dists[top_idx]
        df_out = df_out.reset_index(drop=False)
        return df_out

    # ---------------- special 분리 ----------------
    def _split_special_items(self, df_in):
        if df_in is None or df_in.empty:
            return df_in, df_in

        df = df_in.copy()
        idx = df.index

        is_vintage = df.get("is_vintage", pd.Series(0, index=idx)).fillna(0)
        vintage = df.get("vintage", pd.Series(np.nan, index=idx))
        age = df.get("age", pd.Series(np.nan, index=idx))
        has_bdec = df.get("has_bottling_decade", pd.Series(0, index=idx)).fillna(0)
        bdec = df.get("bottling_decade", pd.Series(np.nan, index=idx))

        special_mask = (((is_vintage == 1) & (vintage <= 1990)) | (age >= 20) | ((has_bdec == 1) & (bdec <= 1980)))

        return df[~special_mask], df[special_mask]

    # --------------------------------------------------------- #
    # RARE POOL (희귀 제품군)
    # --------------------------------------------------------- #
    def _rare_mask(self, df_in: pd.DataFrame) -> pd.Series:
        """
        희귀 제품군 정의 (rarity_score 사용 안 함)

        Rare 조건 (OR):
        - (is_vintage == 1) & (vintage <= 1990)
        - (age >= 20)
        - (has_bottling_decade == 1) & (bottling_decade <= 1980)
        """
        df = df_in

        is_vintage = df.get("is_vintage", 0).fillna(0)
        vintage = df.get("vintage", np.nan)
        age = df.get("age", np.nan)
        has_bdec = df.get("has_bottling_decade", 0).fillna(0)
        bdec = df.get("bottling_decade", np.nan)

        rare_mask = (((is_vintage == 1) & (vintage <= 1990)) | (age >= 20) | ((has_bdec == 1) & (bdec <= 1980)))

        return rare_mask.fillna(False)

    def get_rare_pool(self, top_n: int = 200) -> pd.DataFrame:
        """
        희귀 제품군 자체 (정적 진열용)
        """
        rare = self.df[self._rare_mask(self.df)].copy()
        if rare.empty:
            return rare

        # 정렬 우선순위
        sort_cols = []
        if "age" in rare.columns:
            sort_cols.append("age")
        if "vintage" in rare.columns:
            sort_cols.append("vintage")
        if "bottling_decade" in rare.columns:
            sort_cols.append("bottling_decade")

        if sort_cols:
            rare = rare.sort_values(sort_cols, ascending=False, na_position="last")

        return rare.head(top_n).reset_index(drop=True)

    def _rare_knn_by_meta(self, meta_center: np.ndarray, top_k: int = 20) -> pd.DataFrame:
        """
        '희귀 제품군 안에서만' meta 거리를 계산해서 Top-K 추천.
        - 메인 추천과 독립적인 "희귀 개인화" 결과.
        """
        if meta_center is None or len(meta_center) == 0:
            return pd.DataFrame()

        rare_mask = self._rare_mask(self.df_meta)
        if rare_mask.sum() == 0:
            return pd.DataFrame()

        # meta distance를 "전체 벡터 거리"로 직접 계산 (희귀만 필터)
        X = self.X_meta
        d = np.linalg.norm(X - meta_center.reshape(1, -1), axis=1)

        d_rare = d[rare_mask.values]
        idx_rare = self._meta_index[rare_mask.values]

        if len(idx_rare) == 0:
            return pd.DataFrame()

        k = min(top_k, len(idx_rare))
        top_pos = np.argpartition(d_rare, k - 1)[:k]
        top_pos = top_pos[np.argsort(d_rare[top_pos])]

        picked_idx = idx_rare[top_pos]
        out = self.df.loc[picked_idx].copy()
        out["rare_meta_distance"] = d_rare[top_pos]
        return out.reset_index(drop=True)

    # --------------------------------------------------------- #
    # OUTPUT SPLIT/ORDER
    # --------------------------------------------------------- #
    def _postprocess_final_outputs(self, df_final):
        df = df_final.copy()

        if "style_missing" not in df.columns:
            df["style_missing"] = 0
        if "strong_smoke" not in df.columns:
            df["strong_smoke"] = False

        df_na = df[df["style_missing"] == 1].copy().reset_index(drop=True)
        df_ok = df[df["style_missing"] == 0].copy()

        if "style_richness" in df_ok.columns:
            df_ok = df_ok.sort_values("style_richness", ascending=True, na_position="last")

        less_smoky = df_ok[df_ok["strong_smoke"] == False].copy().reset_index(drop=True)
        smoky = df_ok[df_ok["strong_smoke"] == True].copy().reset_index(drop=True)

        return {
            "style_ok_sorted": df_ok.reset_index(drop=True),
            "less_smoky": less_smoky,
            "smoky": smoky,
            "style_missing": df_na,
        }

    # --------------------------------------------------------- #
    # DIVERSITY HELPERS
    # --------------------------------------------------------- #
    def _sample_axis_value(self, col: str, rng: np.random.Generator):
        if col not in self.df.columns:
            return None
        s = self.df[col].dropna()
        if s.empty:
            return None
        return float(rng.choice(s.values))

    def _explore_sample(self, df_cand: pd.DataFrame, k: int, rng: np.random.Generator):
        if df_cand.empty or k <= 0:
            return df_cand.iloc[0:0].copy()

        df = df_cand.copy()

        if "final_score" not in df.columns:
            return df.sample(n=min(k, len(df)), random_state=int(rng.integers(0, 1_000_000_000)))

        scores = df["final_score"].to_numpy(dtype=float)
        scores = scores - np.nanmax(scores)
        probs = np.exp(scores)
        probs_sum = probs.sum()
        probs = (probs / probs_sum) if probs_sum > 0 else (np.ones_like(probs) / len(probs))

        idx = rng.choice(len(df), size=min(k, len(df)), replace=False, p=probs)
        return df.iloc[idx].copy()

    # --------------------------------------------------------- #
    # MAIN: TAG-BASED RECOMMENDATION + RARE PERSONALIZED
    # --------------------------------------------------------- #
    def recommend_from_survey(
        self,
        style_body=None,
        style_richness=None,
        style_smoke=None,
        style_sweetness=None,
        selected_families=None,
        top_k_taste=80,
        seed_k=15,
        top_k_meta=120,
        final_k=60,
        # 다양성
        n_draws=5,
        explore_k=15,
        candidate_pool_k=300,
        random_state=None,
        # 희귀 개인화
        top_k_rare=20,
        show_rare_pool_top_n=100,
    ):
        selected_families = selected_families or []
        rng = np.random.default_rng(random_state)

        axes = {
            "style_body": style_body,
            "style_richness": style_richness,
            "style_smoke": style_smoke,
            "style_sweetness": style_sweetness,
        }
        none_axes = [k for k, v in axes.items() if v is None]
        draws = n_draws if len(none_axes) > 0 else 1

        all_cands = []
        all_seeds = []
        meta_center_last = None

        for _ in range(draws):
            b = axes["style_body"]
            r = axes["style_richness"]
            s = axes["style_smoke"]
            sw = axes["style_sweetness"]

            if b is None:
                b = self._sample_axis_value("style_body", rng)
            if r is None:
                r = self._sample_axis_value("style_richness", rng)
            if s is None:
                s = self._sample_axis_value("style_smoke", rng)
            if sw is None:
                sw = self._sample_axis_value("style_sweetness", rng)

            have_all_axes = all(v is not None for v in [b, r, s, sw])

            # 1) taste 후보
            if have_all_axes and (self.fa_model is not None) and (self.scaler_taste is not None) and (self.knn_taste is not None):
                style_vec = np.array([b, r, s, sw], dtype=float)
                user_factor = self._project_style_to_factor(style_vec)
                taste_df = self._knn_taste_point(user_factor, top_k_taste).reset_index(drop=False)
            else:
                taste_df = self._taste_candidates_from_partial_style(
                    style_body=b,
                    style_richness=r,
                    style_smoke=s,
                    style_sweetness=sw,
                    top_k=top_k_taste,
                )

            if taste_df is None or taste_df.empty:
                continue

            # family_match 계산 (가중치용)  
            taste_df = self._add_family_match_score(taste_df, selected_families)

            # 2) seed 선택 (전체 taste_df에서 뽑되 family_match 우선)
            seed_df = taste_df.sort_values(
                by=["family_match", "taste_distance"],
                ascending=[False, True],
                na_position="last",
            ).head(seed_k).copy()

            all_seeds.append(seed_df)

            # 3) meta 확장 (seed -> meta_center)
            if "index" not in seed_df.columns:
                seed_df = seed_df.reset_index(drop=False)

            seed_idxs = seed_df["index"].values
            positions = []
            for i in seed_idxs:
                pos = np.where(self._meta_index == i)[0]
                if len(pos) > 0:
                    positions.append(pos[0])

            meta_df = pd.DataFrame()
            if positions:
                meta_center = self.X_meta[positions].mean(axis=0)
                meta_center_last = meta_center  # ✅ 마지막 meta_center 저장 (희귀 개인화에 사용)
                meta_df = self._knn_meta_point(meta_center, top_k_meta).reset_index(drop=False)
                meta_df = self._add_family_match_score(meta_df, selected_families)

            # 4) 후보 만들기: ✅ meta_df 중심 (최종은 meta에서 뽑히도록)
            if meta_df is not None and not meta_df.empty:
                taste_back = taste_df.head(max(10, seed_k)).copy()  # 보조로 조금만
                cand = pd.concat([meta_df, taste_back], axis=0, ignore_index=True)
            else:
                cand = taste_df.copy()

            if cand.empty:
                continue

            for col in ["taste_distance", "meta_distance"]:
                if col not in cand.columns:
                    cand[col] = np.nan
            if "family_match" not in cand.columns:
                cand["family_match"] = 0

            cand = cand.sort_values(
                by=["family_match", "taste_distance", "meta_distance"],
                ascending=[False, True, True],
                na_position="last",
            ).drop_duplicates(subset=["name"], keep="first")

            # 5) 스코어
            provided_axes = sum(v is not None for v in [style_body, style_richness, style_smoke, style_sweetness])
            w_taste = provided_axes / 4.0
            w_meta = 1.0 - w_taste

            taste_d = cand["taste_distance"]
            meta_d = cand["meta_distance"]
            taste_d_filled = taste_d.fillna(taste_d.max() if taste_d.notna().any() else 999.0)
            meta_d_filled = meta_d.fillna(meta_d.max() if meta_d.notna().any() else 999.0)

            cand["final_score"] = (
                cand["family_match"] * 1.0
                - w_taste * taste_d_filled
                - w_meta * meta_d_filled
            )

            all_cands.append(cand)

        if not all_cands:
            return {
                "mode": "survey_only",
                "message": "추천 후보를 만들 수 없습니다.",
                "results": {},
                "rare": {},
            }

        cand = pd.concat(all_cands, axis=0, ignore_index=True)
        cand = cand.sort_values("final_score", ascending=False).drop_duplicates(subset=["name"], keep="first")

        # exploit + explore
        exploit_k = max(final_k - explore_k, 0)
        exploit = cand.head(exploit_k).copy()

        pool = cand.head(candidate_pool_k).copy()
        pool = pool[~pool["name"].isin(exploit["name"])]

        explore = self._explore_sample(pool, k=explore_k, rng=rng)
        final_df = pd.concat([exploit, explore], axis=0, ignore_index=True)
        final_df = final_df.sort_values("final_score", ascending=False).head(final_k).copy()

        results = self._postprocess_final_outputs(final_df)

        seed_products = pd.concat(all_seeds, ignore_index=True) if all_seeds else pd.DataFrame()
        if not seed_products.empty and "name" in seed_products.columns:
            seed_products = seed_products.drop_duplicates(subset=["name"], keep="first")

        #  희귀 제품군 (정적) + 희귀 개인화(meta 기반)
        rare_pool = self.get_rare_pool(top_n=show_rare_pool_top_n)
        if meta_center_last is not None:
            rare_personalized = self._rare_knn_by_meta(meta_center_last, top_k=top_k_rare)
        else:
            rare_personalized = rare_pool.head(top_k_rare).copy()

        return {
            "mode": "survey_only",
            "input": {
                "style_body": style_body,
                "style_richness": style_richness,
                "style_smoke": style_smoke,
                "style_sweetness": style_sweetness,
                "families": selected_families,
            },
            "seed_products": seed_products.reset_index(drop=True),
            "final_candidates": final_df.reset_index(drop=True),
            "results": results,
            "diversity_settings": {
                "n_draws": draws,
                "explore_k": explore_k,
                "candidate_pool_k": candidate_pool_k,
                "random_state": random_state,
            },
            "rare": {
                "rare_pool_top": rare_pool.reset_index(drop=True),
                "rare_personalized_by_meta": rare_personalized.reset_index(drop=True),
                "rare_rule_description": (
                    "Rare pool은 고정 규칙으로 정의됨. "
                    "고객 meta_center 기준으로 rare pool 내부에서만 거리 계산하여 개인화 추천."
                ),
            },
        }
