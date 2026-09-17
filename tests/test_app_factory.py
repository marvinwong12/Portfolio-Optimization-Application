"""Tests for create_app()'s own config-resolution logic - specifically
SECRET_KEY hardening, which runs before any blueprint/route code, so it
doesn't fit naturally in test_routes.py or test_auth.py.
"""
import pytest

from portfolio_optimizer import create_app


class TestSecretKeyResolution:
    def test_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv('SECRET_KEY', 'a-real-secret-from-env')
        monkeypatch.delenv('FLASK_DEBUG', raising=False)

        app = create_app({'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
        assert app.config['SECRET_KEY'] == 'a-real-secret-from-env'

    def test_falls_back_to_dev_key_when_testing(self, monkeypatch):
        monkeypatch.delenv('SECRET_KEY', raising=False)
        monkeypatch.delenv('FLASK_DEBUG', raising=False)

        app = create_app({'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:', 'TESTING': True})
        assert app.config['SECRET_KEY'] == 'dev-only-insecure-secret-key'

    def test_falls_back_to_dev_key_when_flask_debug_env_set(self, monkeypatch):
        monkeypatch.delenv('SECRET_KEY', raising=False)
        monkeypatch.setenv('FLASK_DEBUG', '1')

        app = create_app({'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
        assert app.config['SECRET_KEY'] == 'dev-only-insecure-secret-key'

    def test_raises_without_secret_key_outside_debug_or_testing(self, monkeypatch):
        """Regression test: a missing SECRET_KEY used to silently fall back
        to a hardcoded constant in every environment, which would let an
        attacker who reads the source forge session cookies and (now) CSRF
        tokens in a real deployment."""
        monkeypatch.delenv('SECRET_KEY', raising=False)
        monkeypatch.delenv('FLASK_DEBUG', raising=False)

        with pytest.raises(RuntimeError, match='SECRET_KEY'):
            create_app({'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})

    def test_config_override_can_still_set_secret_key_directly(self, monkeypatch):
        """config_overrides is applied after the resolved key, so callers
        (e.g. tests) can still force a specific value if they need to."""
        monkeypatch.delenv('SECRET_KEY', raising=False)
        monkeypatch.setenv('FLASK_DEBUG', '1')

        app = create_app({
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SECRET_KEY': 'explicitly-overridden',
        })
        assert app.config['SECRET_KEY'] == 'explicitly-overridden'
