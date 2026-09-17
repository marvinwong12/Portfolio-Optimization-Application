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
from flask_wtf import CSRFProtect

from .models import db

migrate = Migrate()
csrf = CSRFProtect()


def _resolve_secret_key(config_overrides):
    """Determine the Flask session secret key.

    A missing SECRET_KEY is a real vulnerability, not just a footgun: it
    lets an attacker who can read/guess a predictable fallback forge signed
    session cookies (and, with CSRFProtect below, CSRF tokens too, since
    those are derived from the same secret). So outside of debug/testing
    runs, this refuses to start rather than silently falling back to a
    constant. Tests and local `FLASK_DEBUG=1` runs get a fixed dev value so
    nobody has to set an env var just to run `pytest`.
    """
    secret_key = os.environ.get('SECRET_KEY')
    if secret_key:
        return secret_key

    is_testing = bool(config_overrides and config_overrides.get('TESTING'))
    is_debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    if is_testing or is_debug:
        return 'dev-only-insecure-secret-key'

    raise RuntimeError(
        "SECRET_KEY environment variable is not set. Refusing to start outside "
        "of debug/testing mode, since a missing SECRET_KEY lets session cookies "
        "and CSRF tokens be forged. Set it, e.g.:\n"
        "  export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')"
    )


def create_app(config_overrides=None):
    """Application factory.

    Args:
        config_overrides (dict): Optional Flask config overrides, e.g.
            {'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:', 'TESTING': True}
            for tests.
    """
    app = Flask(__name__, template_folder='../templates', static_folder='../static')

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portfolios.db'
    app.config['SECRET_KEY'] = _resolve_secret_key(config_overrides)
    if config_overrides:
        app.config.update(config_overrides)

    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)

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

    @app.cli.command('seed-demo-user')
    def seed_demo_user_command():
        """Create (or update) the shared 'Try Demo' account and its sample
        portfolios. Safe to run repeatedly."""
        from .auth import ensure_demo_user_seeded, DEMO_USERNAME
        ensure_demo_user_seeded()
        print(f"Demo user ready (username: '{DEMO_USERNAME}'). Log in via the "
              f"'Try Demo' link on the login page, or GET /demo-login directly.")

    return app
