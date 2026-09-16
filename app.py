"""Entry point for running the Flask app directly or via gunicorn
(`gunicorn app:app`). All actual application code lives in the
`portfolio_optimizer` package."""
import os

from portfolio_optimizer import create_app

app = create_app()

if __name__ == "__main__":
    # Debug mode (which exposes the interactive Werkzeug debugger, a remote
    # code execution risk) is opt-in via env var rather than hardcoded on,
    # so it can't accidentally ship to production.
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode)
