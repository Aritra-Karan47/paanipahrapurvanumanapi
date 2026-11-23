"""
Unit tests for FastAPI async lifespan startup data loading (Issue #23).
"""
import sys
from pathlib import Path
import pytest
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import app as app_module
from main import lifespan


def test_bootstrap_idempotent():
    """Verify bootstrap can be called safely multiple times."""
    res1 = app_module.bootstrap()
    assert res1.get("status") in ("success", "already_bootstrapped")
    assert app_module._is_bootstrapped is True

    res2 = app_module.bootstrap(force=False)
    assert res2.get("status") == "already_bootstrapped"


def test_fastapi_lifespan_execution():
    """Verify that entering the FastAPI lifespan context pre-warms models and loads data."""
    app_module._is_bootstrapped = False
    test_app = FastAPI(lifespan=lifespan)

    assert app_module._is_bootstrapped is False

    # Drive the lifespan async context manager
    ctx = lifespan(test_app)
    enter_coro = ctx.__aenter__()
    try:
        enter_coro.send(None)
    except StopIteration:
        pass

    # Verify bootstrap was triggered and completed by lifespan startup
    assert app_module._is_bootstrapped is True

    # Shutdown lifespan
    exit_coro = ctx.__aexit__(None, None, None)
    try:
        exit_coro.send(None)
    except StopIteration:
        pass

    assert app_module._is_bootstrapped is True
