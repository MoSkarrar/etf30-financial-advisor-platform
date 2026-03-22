from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View

from trader.drl_stock_trader.config.app_config import APP_CONFIG
from trader.services import artifact_store


class SignUp(View):
    def get(self, request):
        return render(request, "trader/register.html")

    def post(self, request):
        email = request.POST.get("email").strip()
        first_name = request.POST.get("first_name").strip()
        last_name = request.POST.get("last_name").strip()
        password = request.POST.get("password").strip()
        password_repeat = request.POST.get("password_repeat").strip()

        if password != password_repeat:
            messages.error(request, "Passwords do not match")
            return HttpResponseRedirect(request.path_info)

        user_already_exists = User.objects.filter(username=email).exists()
        if user_already_exists:
            messages.error(request, "User already exists")
            return HttpResponseRedirect(request.path_info)

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
        )
        user.save()

        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse("trader-home"))


class Login(View):
    def get(self, request):
        return render(request, "trader/login.html")

    def post(self, request):
        username = request.POST.get("email").strip()
        password = request.POST.get("password").strip()
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse("trader-home"))
        messages.error(request, "Email or Password is incorrect")
        return HttpResponseRedirect(request.path_info)


class Logout(View):
    def get(self, request):
        if request.user.is_authenticated:
            logout(request)
        return redirect(reverse("login"))


def home(request):
    if not request.user.is_authenticated:
        messages.warning(request, "Login First!")
        return redirect(reverse("login"))

    investor_defaults = APP_CONFIG.investor_defaults
    policy_defaults = APP_CONFIG.policy
    benchmark_labels = {
        "equal_weight": "Equal-weight ETF30",
        "spy": "SPY proxy",
        "60_40": "60/40 proxy",
        "cap_weight_proxy": "Liquidity-weighted cap proxy",
    }

    context = {
        "app_title": APP_CONFIG.identity.app_title,
        "app_subtitle": "ETF30 portfolio advisor intake",
        "default_initial_amount": int(APP_CONFIG.rl.default_initial_cash),
        "default_train_start": APP_CONFIG.dataset.build_start,
        "default_trade_start": "2015-01-01",
        "default_trade_end": "2025-12-24",
        "benchmark_choices": list(benchmark_labels.items()),
        "benchmark_default": APP_CONFIG.benchmarks.default_primary,
        "benchmark_labels": benchmark_labels,
        "investor_profile_defaults": {
            "profile_name": investor_defaults.profile_name,
            "target_style": "balanced",
            "risk_tolerance": investor_defaults.risk_tolerance,
            "target_volatility": investor_defaults.target_volatility,
            "max_drawdown_preference": investor_defaults.max_drawdown_preference,
            "turnover_aversion": investor_defaults.turnover_aversion,
            "min_cash_preference": investor_defaults.min_cash_preference,
            "benchmark_preference": investor_defaults.benchmark_preference,
            "advisor_mode": "advisor",
        },
        "policy_defaults": {
            "max_single_position_cap": policy_defaults.max_single_position_cap,
            "min_cash_weight": policy_defaults.target_cash_floor,
            "turnover_budget": policy_defaults.turnover_budget,
            "rebalance_cadence_days": policy_defaults.rebalance_cadence_days,
            "allow_cash_sleeve": policy_defaults.allow_cash_sleeve,
            "long_only": policy_defaults.long_only,
        },
        "risk_profiles": [
            ("conservative", "Conservative"),
            ("balanced", "Balanced"),
            ("growth", "Growth"),
        ],
        "advisor_modes": [("advisor", "Advisor mode"), ("research", "Research mode")],
    }
    return render(request, "trader/home.html", context)


def narration_session(request, session_id):
    if not request.user.is_authenticated:
        messages.warning(request, "Login First!")
        return redirect(reverse("login"))

    session_record = artifact_store.load_session_manifest(session_id) or {}
    runs = list(session_record.get("runs") or [])
    market = session_record.get("market", "etf30")
    latest_run_id = session_record.get("latest_run_id", "")
    err = "" if session_record else "Session manifest not found."

    latest_bundle = artifact_store.load_advisory_bundle(latest_run_id) if latest_run_id else None
    latest_summary = ""
    latest_benchmark = {}
    latest_risk = {}
    latest_allocation = {}
    latest_policy = {}
    if latest_bundle:
        latest_summary = latest_bundle.get("advisory_summary_text") or ""
        latest_benchmark = latest_bundle.get("benchmark_comparison") or {}
        latest_risk = latest_bundle.get("risk_snapshot") or {}
        latest_allocation = latest_bundle.get("allocation_recommendation") or {}
        latest_policy = (latest_bundle.get("manifest") or {}).get("portfolio_policy") or {}

    return render(
        request,
        "trader/narration_session.html",
        {
            "session_id": session_id,
            "runs": runs,
            "market": market,
            "latest_run_id": latest_run_id,
            "latest_summary": latest_summary,
            "latest_benchmark": latest_benchmark,
            "latest_risk": latest_risk,
            "latest_allocation": latest_allocation,
            "latest_policy": latest_policy,
            "benchmark_labels": {
                "equal_weight": "Equal-weight ETF30",
                "spy": "SPY proxy",
                "60_40": "60/40 proxy",
                "cap_weight_proxy": "Liquidity-weighted cap proxy",
            },
            "error": err,
        },
    )