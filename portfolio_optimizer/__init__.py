"""Portfolio Optimization Application package.

Exposes an app factory (`create_app`) instead of a module-level Flask
instance, so tests can spin up isolated app instances (e.g. pointed at an
in-memory database) without touching the real portfolios.db.
"""
import os

# Matplotlib's backend must be set before pyplot is imported anywhere in the
# package (visualizer.py imports it at module load time).
import matplotlib
matplotlib.use('Agg')

from flask import Flask
from flask_migrate import Migrate

from .models import db

migrate = Migrate()


def create_app(config_overrides=None):
    """Application factory.

    Args:
        config_overrides (dict): Optional Flask config overrides, e.g.
            {'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:', 'TESTING': True}
            for tests.
    """
    app = Flask(__name__, template_folder='../templates', static_folder='../static')

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portfolios.db'
    # Required for session cookies (login state) and flash messages. In
    # production this MUST be set via the SECRET_KEY env var to a long
    # random value - the fallback here is only for local/dev use, since a
    # known secret lets an attacker forge session cookies.
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-insecure-secret-key')
    if config_overrides:
        app.config.update(config_overrides)

    db.init_app(app)
    migrate.init_app(app, db)

    from .auth import login_manager, auth_bp
    login_manager.init_app(app)
    app.register_blueprint(auth_bp)

    from .routes import bp
    app.register_blueprint(bp)

    with app.app_context():
        # Safe on an already-up-to-date database: only creates tables that
        # don't exist yet. Schema *changes* to existing tables (like adding
        # a column) go through Flask-Migrate instead - see migrations/.
        db.create_all()

    return app
