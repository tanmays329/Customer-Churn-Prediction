"""Reusable Streamlit UI components."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import APP_NAME, APP_TAGLINE, STYLE_PATH, VERSION


def load_css() -> None:
    """Load custom CSS when the stylesheet is available."""

    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def configure_page(title: str) -> None:
    """Apply standard Streamlit page configuration and styling."""

    st.set_page_config(page_title=f"{title} | {APP_NAME}", layout="wide")
    load_css()


def render_header(title: str, subtitle: str) -> None:
    """Render a consistent page header."""

    st.markdown(
        f"""
        <section class="page-header">
            <div>
                <p class="eyebrow">{APP_TAGLINE}</p>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, helper: str | None = None) -> None:
    """Render a compact metric card."""

    helper_html = f"<span>{helper}</span>" if helper else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <p>{label}</p>
            <strong>{value}</strong>
            {helper_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk: str) -> str:
    """Return HTML for a color-coded risk badge."""

    return f'<span class="risk-badge {risk.lower()}">{risk} Risk</span>'


def render_footer() -> None:
    """Render standard footer text."""

    st.markdown(
        f'<footer class="app-footer">{APP_NAME} v{VERSION}</footer>',
        unsafe_allow_html=True,
    )


def render_error(message: str) -> None:
    """Render a professional error message."""

    st.error(message)
