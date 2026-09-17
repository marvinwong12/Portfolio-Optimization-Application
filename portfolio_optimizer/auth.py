"""Registration, login, and logout. Ownership enforcement for portfolio
routes lives in routes.py (each route checks portfolio.user_id against
current_user.id) rather than here."""
import secrets

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from .models import db, User, Portfolios

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

auth_bp = Blueprint('auth', __name__)

# Username of the shared, one-click "Try Demo" account. Seeded via the
# `flask seed-demo-user` CLI command (see portfolio_optimizer/__init__.py).
DEMO_USERNAME = 'demo'

# Sample portfolios the demo account starts with, so a visitor sees real
# analysis output immediately instead of an empty portfolio list.
DEMO_PORTFOLIOS = [
    {
        'name': 'Tech Giants',
        'stocks': 'AAPL,MSFT,GOOGL,AMZN,NVDA',
        'description': 'A concentrated portfolio of large-cap tech growth stocks.',
        'long_only': True,
    },
    {
        'name': 'Diversified Core',
        'stocks': 'SPY,BND,GLD,VNQ',
        'description': 'A simple multi-asset-class portfolio: stocks, bonds, gold, REITs.',
        'long_only': True,
    },
]


def ensure_demo_user_seeded():
    """Create the shared demo account and its sample portfolios if they
    don't already exist. Idempotent, so it's safe to run on every deploy
    via `flask seed-demo-user` without duplicating data.

    Returns:
        User: the demo user (existing or newly created)
    """
    demo_user = User.query.filter_by(username=DEMO_USERNAME).first()
    if demo_user is None:
        demo_user = User(username=DEMO_USERNAME, email='demo@example.invalid')
        # Nobody types a password for this account - it's only reachable
        # via the one-click /demo-login route - so a random, never-shown
        # password is fine.
        demo_user.set_password(secrets.token_urlsafe(32))
        db.session.add(demo_user)
        db.session.flush()  # assigns demo_user.id, needed below

    existing_names = {p.name for p in demo_user.portfolios}
    for sample in DEMO_PORTFOLIOS:
        if sample['name'] not in existing_names:
            db.session.add(Portfolios(user_id=demo_user.id, **sample))

    db.session.commit()
    return demo_user


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash('Username, email, and password are all required.')
            return render_template('register.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash('That username is already taken.')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('An account with that email already exists.')
            return render_template('register.html')

        user = User(username=username, email=email)
        user.set_password(password)

        try:
            db.session.add(user)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            flash(f'There was a problem creating your account: {str(e)}')
            return render_template('register.html')

        login_user(user)
        return redirect(url_for('main.index'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            flash('Invalid username or password.')
            return render_template('login.html')

        login_user(user)
        next_page = request.args.get('next')
        return redirect(next_page or url_for('main.index'))

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth_bp.route('/demo-login')
def demo_login():
    """One-click login as a shared demo account, so a visitor can explore
    the app without registering first. Gated by DEMO_LOGIN_ENABLED (on by
    default) so it can be turned off with a config change if this is ever
    deployed somewhere that shouldn't offer open, no-password access."""
    if not current_app.config.get('DEMO_LOGIN_ENABLED', True):
        flash('The demo account is not available right now.')
        return redirect(url_for('auth.login'))

    demo_user = User.query.filter_by(username=DEMO_USERNAME).first()
    if demo_user is None:
        flash("The demo account hasn't been set up yet.")
        return redirect(url_for('auth.login'))

    login_user(demo_user)
    return redirect(url_for('main.index'))
