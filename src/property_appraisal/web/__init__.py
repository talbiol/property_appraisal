"""Flask web app: enter a zip code, see listings priced below their predicted value."""

from __future__ import annotations

import logging

from flask import Flask


def create_app() -> Flask:
    """Application factory used by ``flask --app property_appraisal.web run``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__)

    from property_appraisal.web.routes import bp

    app.register_blueprint(bp)
    return app
