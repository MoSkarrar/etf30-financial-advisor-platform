from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _ensure_package(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType(name)
    pkg.__path__ = []  # type: ignore[attr-defined]
    _install_module(name, pkg)
    return pkg


def _stub_if_missing(name: str, factory):
    if name in sys.modules:
        return sys.modules[name]
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        module = factory()
        _install_module(name, module)
        return module


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -----------------------------
# asgiref / channels stubs
# -----------------------------
def _make_asgiref_sync():
    m = types.ModuleType("asgiref.sync")

    def async_to_sync(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    m.async_to_sync = async_to_sync
    return m


_stub_if_missing("asgiref", lambda: _ensure_package("asgiref"))
_stub_if_missing("asgiref.sync", _make_asgiref_sync)


def _make_channels_ws():
    m = types.ModuleType("channels.generic.websocket")

    class WebsocketConsumer:
        def __init__(self, *args, **kwargs):
            self.scope = kwargs.get("scope", {})
            self.sent = []
            self.accepted = False
            self.channel_name = "test-channel"
            self.channel_layer = types.SimpleNamespace(send=lambda *a, **k: None)

        def accept(self):
            self.accepted = True

        def send(self, text_data=None, bytes_data=None):
            self.sent.append({"text_data": text_data, "bytes_data": bytes_data})

    m.WebsocketConsumer = WebsocketConsumer
    return m


_stub_if_missing("channels", lambda: _ensure_package("channels"))
_stub_if_missing("channels.generic", lambda: _ensure_package("channels.generic"))
_stub_if_missing("channels.generic.websocket", _make_channels_ws)


# -----------------------------
# django stubs
# -----------------------------
def _make_django_messages():
    m = types.ModuleType("django.contrib.messages")
    m.error = lambda *a, **k: None
    m.warning = lambda *a, **k: None
    return m


def _make_django_auth():
    m = types.ModuleType("django.contrib.auth")
    m.authenticate = lambda *a, **k: None
    m.login = lambda *a, **k: None
    m.logout = lambda *a, **k: None
    return m


def _make_django_auth_models():
    m = types.ModuleType("django.contrib.auth.models")

    class _FilterResult:
        def __init__(self, exists=False):
            self._exists = exists

        def exists(self):
            return self._exists

    class _UserManager:
        def filter(self, **kwargs):
            return _FilterResult(False)

        def create_user(self, **kwargs):
            user = User(**kwargs)
            return user

    class User:
        objects = _UserManager()

        def __init__(self, username="", email="", password="", first_name="", last_name=""):
            self.username = username
            self.email = email
            self.password = password
            self.first_name = first_name
            self.last_name = last_name

        def save(self):
            return None

    m.User = User
    return m


def _make_django_http():
    m = types.ModuleType("django.http")

    class HttpResponseRedirect:
        def __init__(self, target):
            self.target = target

    m.HttpResponseRedirect = HttpResponseRedirect
    return m


def _make_django_shortcuts():
    m = types.ModuleType("django.shortcuts")
    m.redirect = lambda target: {"redirect": target}
    m.render = lambda request, template, context=None: {"template": template, "context": context or {}}
    return m


def _make_django_urls():
    m = types.ModuleType("django.urls")
    m.reverse = lambda name: f"/{name}/"
    return m


def _make_django_views():
    m = types.ModuleType("django.views")

    class View:
        pass

    m.View = View
    return m


_stub_if_missing("django", lambda: _ensure_package("django"))
_stub_if_missing("django.contrib", lambda: _ensure_package("django.contrib"))
_stub_if_missing("django.contrib.messages", _make_django_messages)
_stub_if_missing("django.contrib.auth", _make_django_auth)
_stub_if_missing("django.contrib.auth.models", _make_django_auth_models)
_stub_if_missing("django.http", _make_django_http)
_stub_if_missing("django.shortcuts", _make_django_shortcuts)
_stub_if_missing("django.urls", _make_django_urls)
_stub_if_missing("django.views", _make_django_views)


# -----------------------------
# gym / stable_baselines3 stubs
# -----------------------------
def _make_gym():
    m = types.ModuleType("gym")

    class Wrapper:
        def __init__(self, env):
            self.env = env

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

    m.Wrapper = Wrapper
    m.Env = object
    return m


_stub_if_missing("gym", _make_gym)


def _make_sb3():
    m = types.ModuleType("stable_baselines3")

    class _Algo:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def learn(self, total_timesteps):
            self.total_timesteps = total_timesteps
            return self

        def save(self, path):
            self.saved_path = path

        def predict(self, obs, deterministic=True):
            import numpy as np

            if hasattr(obs, "shape"):
                size = obs.shape[-1] if len(obs.shape) else 1
            else:
                size = 1
            return np.zeros(size, dtype=float), None

    m.A2C = _Algo
    m.DDPG = _Algo
    m.PPO = _Algo
    return m


def _make_sb3_noise():
    m = types.ModuleType("stable_baselines3.common.noise")

    class NormalActionNoise:
        def __init__(self, mean, sigma):
            self.mean = mean
            self.sigma = sigma

    m.NormalActionNoise = NormalActionNoise
    return m


def _make_sb3_vec_env():
    m = types.ModuleType("stable_baselines3.common.vec_env")

    class DummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]
            env = self.envs[0]
            self.observation_space = getattr(env, "observation_space", types.SimpleNamespace(shape=(1,)))
            self.action_space = getattr(env, "action_space", types.SimpleNamespace(shape=(1,)))

    m.DummyVecEnv = DummyVecEnv
    return m


_stub_if_missing("stable_baselines3", _make_sb3)
_stub_if_missing("stable_baselines3.common", lambda: _ensure_package("stable_baselines3.common"))
_stub_if_missing("stable_baselines3.common.noise", _make_sb3_noise)
_stub_if_missing("stable_baselines3.common.vec_env", _make_sb3_vec_env)


# -----------------------------
# internal module stubs used by imports
# -----------------------------
def _make_preprocess():
    import pandas as pd

    m = types.ModuleType("trader.drl_stock_trader.preprocess")

    def clean_etf_frame(df):
        return df.copy() if df is not None else pd.DataFrame()

    def data_split(df, start_date=None, end_date=None):
        out = df.copy()
        if start_date is not None:
            out = out[out["datadate"] >= int(start_date)]
        if end_date is not None:
            out = out[out["datadate"] <= int(end_date)]
        return out.copy()

    def infer_feature_columns(df):
        reserved = {
            "datadate",
            "tic",
            "adjcp",
            "close",
            "open",
            "high",
            "low",
            "volume",
        }
        return [c for c in df.columns if c not in reserved and not c.startswith("benchmark_")]

    def pivot_prices(df):
        price_col = "adjcp" if "adjcp" in df.columns else "close"
        return df.pivot(index="datadate", columns="tic", values=price_col).sort_index().ffill().fillna(0.0)

    def pivot_features(df, feature_columns):
        return {
            feature: df.pivot(index="datadate", columns="tic", values=feature).sort_index().ffill().fillna(0.0)
            for feature in feature_columns
        }

    m.clean_etf_frame = clean_etf_frame
    m.data_split = data_split
    m.infer_feature_columns = infer_feature_columns
    m.pivot_features = pivot_features
    m.pivot_prices = pivot_prices
    return m


_stub_if_missing("trader", lambda: _ensure_package("trader"))
_stub_if_missing("trader.drl_stock_trader", lambda: _ensure_package("trader.drl_stock_trader"))
_stub_if_missing("trader.drl_stock_trader.preprocess", _make_preprocess)


def _make_env_module(class_name):
    m = types.ModuleType(class_name)

    class _Env:
        def __init__(self, *args, **kwargs):
            self.investor_profile = kwargs.get("investor_profile")
            self.policy = kwargs.get("policy")
            self.observation_space = types.SimpleNamespace(shape=(3,))
            self.action_space = types.SimpleNamespace(shape=(3,))

        def reset(self):
            return [0.0, 100.0, 1.0]

        def step(self, action):
            return [0.0, 100.0, 1.0], 0.0, True, {"portfolio_value": 100.0}

        def get_history_frame(self):
            import pandas as pd

            return pd.DataFrame(
                {
                    "datadate": [20200101, 20200102],
                    "account_value": [100.0, 101.0],
                    "portfolio_return": [0.0, 0.01],
                    "benchmark_return": [0.0, 0.005],
                    "active_return": [0.0, 0.005],
                    "turnover": [0.0, 0.1],
                    "concentration_hhi": [0.1, 0.12],
                }
            )

        def get_current_weights(self):
            return {"AAA": 0.5, "BBB": 0.4, "CASH": 0.1}

    setattr(m, class_name.split(".")[-1], _Env)
    return m


_stub_if_missing(
    "trader.drl_stock_trader.RL_envs.EnvMultipleStock_Validation",
    lambda: _make_env_module("StockEnvValidation"),
)
_stub_if_missing(
    "trader.drl_stock_trader.RL_envs.EnvMultipleStocks_Train",
    lambda: _make_env_module("StockEnvTrain"),
)


def _make_dataset_builder():
    import pandas as pd

    m = types.ModuleType("trader.drl_stock_trader.data.make_etf_dataset_yf")
    m.DEFAULT_ETF30 = ["AAA", "BBB", "CCC"]

    def build_etf_dataset(*args, **kwargs):
        return pd.DataFrame(
            {
                "datadate": [20200101, 20200101, 20200102, 20200102],
                "tic": ["AAA", "BBB", "AAA", "BBB"],
                "adjcp": [100.0, 200.0, 101.0, 202.0],
                "volume": [10.0, 20.0, 11.0, 21.0],
                "macd": [0.1, 0.2, 0.15, 0.25],
            }
        )

    m.build_etf_dataset = build_etf_dataset
    return m


_stub_if_missing("trader.drl_stock_trader.data", lambda: _ensure_package("trader.drl_stock_trader.data"))
_stub_if_missing("trader.drl_stock_trader.data.make_etf_dataset_yf", _make_dataset_builder)
