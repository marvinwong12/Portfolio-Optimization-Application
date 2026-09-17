"""SQLAlchemy models. `db` is unbound here (initialized via db.init_app in
the app factory) so this module has no dependency on a live Flask app,
which keeps it importable from tests without spinning up the whole app."""
from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model, UserMixin):
    """
    A registered user. Owns zero or more Portfolios.

    Attributes:
        id (int): Primary key
        username (str): Unique login name
        email (str): Unique email address
        password_hash (str): Salted hash of the user's password (never the
            plaintext password itself)
        date_created (datetime): Registration timestamp
    """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    portfolios = db.relationship('Portfolios', backref='owner', lazy=True,
                                  cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Portfolios(db.Model):
    """
    Database model for storing portfolio information.

    Attributes:
        id (int): Primary key
        user_id (int): Owning user's id - a portfolio is only ever visible
            to, or modifiable by, the user who owns it
        name (str): Portfolio name
        stocks (str): Comma-separated stock symbols
        description (str): Portfolio description
        weights (str): Tangency (max Sharpe) portfolio weights as a JSON
            string, keyed by symbol - populated after each analysis run
            via the /access/<id> route
        long_only (bool): Whether portfolio is long-only or long-short
        date_created (datetime): Creation timestamp
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    stocks = db.Column(db.String(500), nullable=False)
    description = db.Column(db.String(200))
    weights = db.Column(db.String(500))
    long_only = db.Column(db.Boolean, default=True, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    snapshots = db.relationship('PortfolioSnapshot', backref='portfolio', lazy=True,
                                 cascade='all, delete-orphan',
                                 order_by='PortfolioSnapshot.created_at')

    def __repr__(self):
        return f'<Portfolio {self.name}>'


class PortfolioSnapshot(db.Model):
    """
    A single strategy's computed weights at one point in time, recorded
    every time /access re-runs the analysis. Where Portfolios.weights holds
    only the latest Tangency weights (for backward compatibility), this
    table accumulates one row per strategy per analysis run, so weight
    drift over time can actually be shown rather than only the most recent
    snapshot.

    Attributes:
        id (int): Primary key
        portfolio_id (int): The portfolio this snapshot belongs to
        strategy (str): Strategy name, e.g. "Tangency", "Minimum Variance",
            "Equal Weight", "Monte Carlo Optimal"
        weights (str): JSON string, {symbol: weight}
        portfolio_return (float): Annualized expected return at the time
            of this snapshot
        volatility (float): Annualized volatility at the time of this snapshot
        sharpe_ratio (float): Sharpe ratio at the time of this snapshot
        created_at (datetime): When this snapshot was recorded
    """

    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False, index=True)
    strategy = db.Column(db.String(50), nullable=False)
    weights = db.Column(db.String(1000), nullable=False)
    portfolio_return = db.Column(db.Float)
    volatility = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<PortfolioSnapshot {self.strategy} @ {self.created_at}>'
