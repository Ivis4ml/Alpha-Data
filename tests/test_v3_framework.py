"""v3 框架单元测试（全部离线，不依赖本地数据）。"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from alpha_data.polymarket.v3 import (  # noqa: E402,E501
    coherence,
    families,
    gates,
    geometry,
    impact,
    latent,
    story,
)


class TestFamilies:
    def test_价格阈值族解析(self):
        p = families.parse_slug("will-crude-oil-cl-hit-high-100-by-end-of-march-658-396-769-971")
        assert p.family_type == "price_ladder"
        assert p.strike == 100.0
        assert p.barrier == "high"
        assert p.deadline == date(2026, 3, 31)
        assert p.family_key == "crude-oil-cl:high:2026-03-31"

    def test_日期阶梯族解析(self):
        p = families.parse_slug("us-strikes-iran-by-february-28-2026-227-967")
        assert p.family_type == "date_ladder"
        assert p.family_key == "us-strikes-iran"
        assert p.deadline == date(2026, 2, 28)

    def test_fed_meeting形态(self):
        p = families.parse_slug("fed-rate-cut-by-march-2026-meeting")
        assert p.family_type == "date_ladder"
        assert p.family_key == "fed-rate-cut"
        assert p.deadline == date(2026, 3, 31)

    def test_单日事件为date_point(self):
        p = families.parse_slug("will-iran-strike-israel-on-march-10")
        assert p.family_type == "date_point"
        assert p.deadline == date(2026, 3, 10)

    def test_年末形态(self):
        p = families.parse_slug("will-the-iranian-regime-fall-by-the-end-of-2026")
        assert p.family_type == "date_ladder"
        assert p.deadline == date(2026, 12, 31)

    def test_无模式为single(self):
        p = families.parse_slug("permanent-peace-deal-with-iran")
        assert p.family_type == "single"

    def test_价格阈值禁入Layer4(self):
        reg = pd.DataFrame(
            {
                "condition_id": ["a", "b"],
                "slug": [
                    "will-gold-gc-hit-high-3500-by-end-of-june",
                    "us-strikes-iran-by-june-30",
                ],
                "theme": ["metal_price", "mideast_conflict"],
            }
        )
        fam = families.build_families(reg)
        assert not fam.loc[fam["condition_id"] == "a", "layer4_eligible"].iloc[0]
        assert fam.loc[fam["condition_id"] == "b", "layer4_eligible"].iloc[0]


class TestCoherence:
    def test_pava单调性与守恒(self):
        y = np.array([0.1, 0.3, 0.2, 0.5])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        q = coherence.pava(y, w)
        assert (np.diff(q) >= -1e-12).all()
        assert abs(q.sum() - y.sum()) < 1e-9  # 等权时均值守恒

    def test_pava加权(self):
        # 大权重点几乎不动：违反对 (0.4, 0.1)，权重 100:1。
        q = coherence.pava(np.array([0.4, 0.1]), np.array([100.0, 1.0]))
        assert abs(q[0] - q[1]) < 1e-9
        assert abs(q[0] - (0.4 * 100 + 0.1) / 101) < 1e-9

    def test_日期族投影方向(self):
        snap = pd.DataFrame(
            {
                "condition_id": ["m1", "m2"],
                "order_key": [1.0, 2.0],   # 截止先后
                "p": [0.6, 0.4],           # 违反：早截止概率更高
                "w": [1.0, 1.0],
            }
        )
        out = coherence.project_family(snap, family_type="date_ladder")
        assert out["p_proj"].iloc[0] <= out["p_proj"].iloc[1] + 1e-12
        assert out.attrs["tension"] > 0

    def test_hit_high族沿行权价不增(self):
        snap = pd.DataFrame(
            {
                "condition_id": ["m1", "m2", "m3"],
                "order_key": [100.0, 110.0, 120.0],
                "p": [0.5, 0.6, 0.2],      # 中间违反
                "w": [1.0, 1.0, 1.0],
            }
        )
        out = coherence.project_family(snap, family_type="price_ladder", barrier="high")
        q = out["p_proj"].to_numpy()
        assert (np.diff(q) <= 1e-12).all()


class TestGeometry:
    def test_累计强度(self):
        lam = geometry.cumulative_intensity(np.array([0.0, 0.5, 0.9]))
        assert lam[0] == 0.0
        assert abs(lam[1] - np.log(2.0)) < 1e-12
        assert lam[2] > lam[1]

    def test_价格分布特征(self):
        # 生存函数 S(K)：P(max >= 100)=0.9, >=110)=0.5, >=120)=0.1。
        f = geometry.price_features_snapshot(
            np.array([100.0, 110.0, 120.0]), np.array([0.9, 0.5, 0.1]),
            barrier="high",
        )
        assert abs(f["implied_med"] - 110.0) < 1e-9
        assert abs(f["tail_mass"] - 0.1) < 1e-12
        assert f["entropy"] > 0


class TestLatent:
    def test_滤波收敛与创新缩减(self):
        rng = np.random.default_rng(7)
        T = 400
        true_z = np.cumsum(rng.normal(0, 0.05, T))
        noise = rng.normal(0, 0.5, T)
        y = true_z + noise
        obs = pd.DataFrame(
            {
                "condition_id": "m",
                "bucket_end": np.arange(T) * 900 + 900,
                "y": y,
                "n_trades": 10.0,
                "usdc": 1000.0,
                "spread": 0.1,
                "last_ts": np.arange(T) * 900 + 890,
            }
        )
        fit = latent.fit_filter(obs)
        st = fit.states
        # 滤波状态比原始观测更接近真值。
        mse_filter = np.nanmean((st["z"].to_numpy() - true_z) ** 2)
        mse_raw = np.mean((y - true_z) ** 2)
        assert mse_filter < mse_raw * 0.5
        # 标准化创新方差接近 1（模型自洽）。
        v = np.nanvar(st["nu_std"].to_numpy())
        assert 0.5 < v < 2.0

    def test_边界LOCF与时效(self):
        st = pd.DataFrame(
            {
                "condition_id": ["m"] * 2,
                "bucket_end": [900, 1800],
                "z": [0.1, 0.2],
                "P": [0.01, 0.01],
                "nu": [np.nan, 0.1],
                "S": [np.nan, 0.05],
                "nu_std": [np.nan, 0.45],
                "last_ts": [890, 1790],
            }
        )
        b = pd.DataFrame(
            {"condition_id": ["m", "m", "m"], "boundary_ts": [1000, 1800, 1790 + 121 * 60]}
        )
        out = latent.state_at_boundaries(st, b, p_age_max_minutes=120.0)
        assert abs(out["z"].iloc[0] - 0.1) < 1e-12   # 900 桶 LOCF
        assert abs(out["z"].iloc[1] - 0.2) < 1e-12   # 恰在边界（允许相等）
        assert np.isnan(out["z"].iloc[2])            # 超时效


class TestStory:
    def test_正交化无前视(self):
        dates = [f"2026-01-{d:02d}" for d in range(1, 31)]
        rng = np.random.default_rng(1)
        sig = pd.DataFrame(
            {
                "theme": "t", "product": "P", "trade_date": dates,
                "s_pre": rng.normal(0, 1, 30),
            }
        )
        f = pd.DataFrame(
            {"trade_date": dates, "window": "pre", "f_pm": rng.normal(0, 1, 30)}
        )
        out1 = story.orthogonalize(sig, f, cols=("s_pre",), min_obs=5)
        # 修改最后一天的因子值，不应影响此前任何残差。
        f2 = f.copy()
        f2.loc[f2.index[-1], "f_pm"] = 99.0
        out2 = story.orthogonalize(sig, f2, cols=("s_pre",), min_obs=5)
        a = out1["s_pre_orth"].to_numpy()[:-1]
        b = out2["s_pre_orth"].to_numpy()[:-1]
        assert np.allclose(a, b, equal_nan=True)


class TestGates:
    def test_MDE公式(self):
        surp = pd.DataFrame(
            {
                "condition_id": [f"m{i}" for i in range(10)],
                "theme": "t", "product": "P",
                "family_key": [f"f{i}" for i in range(10)],
                "resolved_at": pd.to_datetime(
                    [f"2026-03-{d:02d}" for d in range(1, 11)], utc=True),
                "q_pre": 0.5, "y": [1, 0] * 5,
                "surprise": [0.5, -0.5] * 5,
                "cluster": [f"c{i}" for i in range(10)],
            }
        )
        power = gates.power_table(surp, pd.Series({"P": 0.01}), rho=0.5)
        row = power.iloc[0]
        # 每簇一事件 -> 设计效应 1；sum_s2 = 10*0.25 = 2.5。
        assert abs(row["design_effect"] - 1.0) < 1e-9
        assert abs(row["sum_s2"] - 2.5) < 1e-9
        expect_mde = (1.959964 + 0.841621) * 0.01 / np.sqrt(2.5)
        assert abs(row["mde"] - expect_mde) < 1e-6
        assert row["decision"] == "category_beta"  # mde ~ 0.0177 < 0.02


class TestImpact:
    def test_下一可交易日对齐(self):
        rets = pd.DataFrame(
            {
                "product": "P",
                "trade_date": ["2026-03-02", "2026-03-03", "2026-03-04"],
                "r_cc": [0.01, 0.02, 0.03],
            }
        )
        # 北京 3/3 08:00（09:00 前）-> 当日；北京 3/3 10:00 -> 次日。
        ts = pd.Series(pd.to_datetime(
            ["2026-03-03 00:00:00", "2026-03-03 02:00:00"], utc=True))
        out = impact.next_trading_return(ts, rets, product="P")
        assert abs(out.iloc[0] - 0.02) < 1e-12
        assert abs(out.iloc[1] - 0.03) < 1e-12


class TestEpoch:
    def test_微秒精度时间列不失真(self):
        from v3_measurement import beijing_to_epoch

        ts = pd.Series(pd.to_datetime(["2026-01-05 00:00:00"])).astype("datetime64[us]")
        out = beijing_to_epoch(ts)
        expect = int(pd.Timestamp("2026-01-04 16:00:00", tz="UTC").timestamp())
        assert out[0] == expect
