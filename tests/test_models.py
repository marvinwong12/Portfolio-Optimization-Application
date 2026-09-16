import pytest
from sqlalchemy.exc import IntegrityError

from portfolio_optimizer.models import db, Portfolios, User


def _make_user(username='owner', email='owner@example.com', password='password123'):
    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


class TestUser:
    def test_password_is_hashed_not_stored_in_plaintext(self, app):
        with app.app_context():
            user = _make_user(password='correct horse battery staple')
            assert user.password_hash != 'correct horse battery staple'

    def test_check_password_accepts_correct_password(self, app):
        with app.app_context():
            user = _make_user(password='correct horse battery staple')
            assert user.check_password('correct horse battery staple') is True

    def test_check_password_rejects_wrong_password(self, app):
        with app.app_context():
            user = _make_user(password='correct horse battery staple')
            assert user.check_password('wrong password') is False

    def test_username_must_be_unique(self, app):
        with app.app_context():
            _make_user(username='dup', email='a@example.com')
            db.session.add(User(username='dup', email='b@example.com', password_hash='x'))
            with pytest.raises(IntegrityError):
                db.session.commit()

    def test_email_must_be_unique(self, app):
        with app.app_context():
            _make_user(username='a', email='dup@example.com')
            db.session.add(User(username='b', email='dup@example.com', password_hash='x'))
            with pytest.raises(IntegrityError):
                db.session.commit()

    def test_repr(self, app):
        with app.app_context():
            user = _make_user(username='ReprUser')
            assert repr(user) == '<User ReprUser>'


class TestPortfolios:
    def test_create_and_query_portfolio(self, app):
        with app.app_context():
            user = _make_user()
            portfolio = Portfolios(user_id=user.id, name='Tech', stocks='AAPL,MSFT,GOOG', long_only=True)
            db.session.add(portfolio)
            db.session.commit()

            fetched = Portfolios.query.filter_by(name='Tech').first()
            assert fetched is not None
            assert fetched.stocks == 'AAPL,MSFT,GOOG'
            assert fetched.long_only is True
            assert fetched.weights is None  # not populated until analyzed
            assert fetched.user_id == user.id
            assert fetched.owner.username == user.username

    def test_default_long_only_is_true(self, app):
        with app.app_context():
            user = _make_user()
            portfolio = Portfolios(user_id=user.id, name='Default', stocks='AAPL')
            db.session.add(portfolio)
            db.session.commit()
            assert portfolio.long_only is True

    def test_portfolio_requires_a_user(self, app):
        with app.app_context():
            portfolio = Portfolios(name='Orphan', stocks='AAPL')
            db.session.add(portfolio)
            with pytest.raises(IntegrityError):
                db.session.commit()

    def test_deleting_user_deletes_their_portfolios(self, app):
        with app.app_context():
            user = _make_user()
            db.session.add(Portfolios(user_id=user.id, name='ToCascade', stocks='AAPL'))
            db.session.commit()

            db.session.delete(user)
            db.session.commit()

            assert Portfolios.query.filter_by(name='ToCascade').first() is None

    def test_repr(self, app):
        with app.app_context():
            user = _make_user()
            portfolio = Portfolios(user_id=user.id, name='ReprTest', stocks='AAPL')
            assert repr(portfolio) == '<Portfolio ReprTest>'
