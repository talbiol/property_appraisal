"""HTTP routes for the web app."""

from __future__ import annotations

import logging
from functools import cache

import pandas as pd
from flask import Blueprint, render_template, request

from property_appraisal.config import INFERENCE_DATA_DIR, TRAINING_DATA_DIR
from property_appraisal.features import listed_zip_codes, load_dataset
from property_appraisal.valuation import Appraisal, appraise_zip_code

logger = logging.getLogger(__name__)

bp = Blueprint("appraisal", __name__)


@cache
def available_zip_codes() -> list[int]:
    return sorted(listed_zip_codes(INFERENCE_DATA_DIR))


@cache
def _datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_dataset(TRAINING_DATA_DIR), load_dataset(INFERENCE_DATA_DIR)


@cache
def appraisals_for(zip_code: int) -> tuple[Appraisal, ...]:
    """Appraise a zip code once per process; the bundled data does not change at runtime."""
    logger.info("Appraising listings in %s", zip_code)
    training, listings = _datasets()
    return tuple(appraise_zip_code(training, listings, zip_code))


@bp.get("/")
def index():
    return render_template("index.html", zip_codes=available_zip_codes())


@bp.post("/")
def appraise():
    zip_codes = available_zip_codes()
    selected = request.form.get("zip_code", "").strip()
    if not selected.isdigit() or int(selected) not in zip_codes:
        return render_template(
            "index.html", zip_codes=zip_codes, error="Choose one of the available zip codes."
        ), 400

    zip_code = int(selected)
    return render_template(
        "index.html", zip_codes=zip_codes, zip_code=zip_code, results=appraisals_for(zip_code)
    )
