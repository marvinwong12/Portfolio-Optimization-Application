"""Registration, login, and logout. Ownership enforcement for portfolio
routes lives in routes.py (each route checks portfolio.user_id against
current_user.id) rather than here."""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from .models import db, User

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

auth_bp = Blueprint('auth', __name__)


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
