from portfolio_optimizer.models import db, User, Portfolios
from portfolio_optimizer.auth import ensure_demo_user_seeded, DEMO_USERNAME, DEMO_PORTFOLIOS


class TestRegister:
    def test_get_renders_form(self, client):
        response = client.get('/register')
        assert response.status_code == 200
        assert b'Create an Account' in response.data

    def test_successful_registration_creates_user_and_logs_in(self, client, app):
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'password123',
            'confirm_password': 'password123',
        })
        assert response.status_code == 302

        with app.app_context():
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            assert user.check_password('password123')

        # Registering logs the user in - protected route should now work.
        index_response = client.get('/')
        assert index_response.status_code == 200

    def test_password_too_short_is_rejected(self, client, app):
        response = client.post('/register', data={
            'username': 'shortpw',
            'email': 'shortpw@example.com',
            'password': 'short',
            'confirm_password': 'short',
        })
        assert response.status_code == 200
        assert b'at least 8 characters' in response.data
        with app.app_context():
            assert User.query.filter_by(username='shortpw').first() is None

    def test_mismatched_passwords_are_rejected(self, client):
        response = client.post('/register', data={
            'username': 'mismatch',
            'email': 'mismatch@example.com',
            'password': 'password123',
            'confirm_password': 'differentpassword',
        })
        assert response.status_code == 200
        assert b'do not match' in response.data

    def test_duplicate_username_is_rejected(self, client, create_user):
        create_user(username='taken', email='taken@example.com')
        response = client.post('/register', data={
            'username': 'taken',
            'email': 'different@example.com',
            'password': 'password123',
            'confirm_password': 'password123',
        })
        assert response.status_code == 200
        assert b'already taken' in response.data

    def test_duplicate_email_is_rejected(self, client, create_user):
        create_user(username='original', email='dup@example.com')
        response = client.post('/register', data={
            'username': 'different',
            'email': 'dup@example.com',
            'password': 'password123',
            'confirm_password': 'password123',
        })
        assert response.status_code == 200
        assert b'already exists' in response.data


class TestLogin:
    def test_get_renders_form(self, client):
        response = client.get('/login')
        assert response.status_code == 200
        assert b'Log In' in response.data

    def test_correct_credentials_log_in(self, client, create_user):
        create_user(username='alice', password='correcthorse')
        response = client.post('/login', data={'username': 'alice', 'password': 'correcthorse'})
        assert response.status_code == 302
        assert response.headers['Location'] == '/'

    def test_wrong_password_is_rejected(self, client, create_user):
        create_user(username='alice', password='correcthorse')
        response = client.post('/login', data={'username': 'alice', 'password': 'wrongpassword'})
        assert response.status_code == 200
        assert b'Invalid username or password' in response.data

    def test_unknown_username_is_rejected(self, client):
        response = client.post('/login', data={'username': 'nobody', 'password': 'whatever'})
        assert response.status_code == 200
        assert b'Invalid username or password' in response.data

    def test_next_param_redirects_after_login(self, client, create_user):
        create_user(username='alice', password='correcthorse')
        response = client.post('/login?next=/register', data={'username': 'alice', 'password': 'correcthorse'})
        assert response.status_code == 302
        assert response.headers['Location'] == '/register'


class TestLogout:
    def test_logout_requires_login(self, client):
        response = client.get('/logout')
        assert response.status_code == 302  # redirected to login

    def test_logout_ends_session(self, logged_in_client):
        client, _ = logged_in_client
        client.get('/logout')
        response = client.get('/')
        assert response.status_code == 302  # bounced back to login


class TestDemoLogin:
    def test_login_page_shows_demo_button(self, client):
        response = client.get('/login')
        assert b'Try Demo' in response.data

    def test_demo_login_before_seeding_flashes_and_redirects_to_login(self, client):
        response = client.get('/demo-login')
        assert response.status_code == 302
        assert response.headers['Location'] == '/login'

    def test_demo_login_after_seeding_logs_in_and_redirects_to_index(self, client, app):
        with app.app_context():
            ensure_demo_user_seeded()

        response = client.get('/demo-login')
        assert response.status_code == 302
        assert response.headers['Location'] == '/'

        index_response = client.get('/')
        assert b'demo' in index_response.data  # "Signed in as demo"

    def test_seeding_creates_sample_portfolios(self, app):
        with app.app_context():
            demo_user = ensure_demo_user_seeded()
            portfolios = Portfolios.query.filter_by(user_id=demo_user.id).all()
            assert {p.name for p in portfolios} == {p['name'] for p in DEMO_PORTFOLIOS}

    def test_seeding_is_idempotent(self, app):
        with app.app_context():
            ensure_demo_user_seeded()
            ensure_demo_user_seeded()  # calling it again shouldn't duplicate anything

            assert User.query.filter_by(username=DEMO_USERNAME).count() == 1
            demo_user = User.query.filter_by(username=DEMO_USERNAME).first()
            assert Portfolios.query.filter_by(user_id=demo_user.id).count() == len(DEMO_PORTFOLIOS)

    def test_cli_command_seeds_demo_user(self, app):
        runner = app.test_cli_runner()
        result = runner.invoke(args=['seed-demo-user'])
        assert result.exit_code == 0
        assert 'Demo user ready' in result.output

        with app.app_context():
            assert User.query.filter_by(username=DEMO_USERNAME).first() is not None

    def test_demo_login_disabled_via_config(self, app):
        app.config['DEMO_LOGIN_ENABLED'] = False
        client = app.test_client()
        with app.app_context():
            ensure_demo_user_seeded()

        response = client.get('/demo-login')
        assert response.status_code == 302
        assert response.headers['Location'] == '/login'

        # Confirm it actually stayed logged out, not just redirected.
        index_response = client.get('/')
        assert index_response.status_code == 302


class TestProtectedRoutesRequireLogin:
    def test_index_redirects_anonymous_users_to_login(self, client):
        response = client.get('/')
        assert response.status_code == 302
        assert '/login' in response.headers['Location']

    def test_analyze_stock_requires_login(self, client):
        response = client.post('/analyze_stock', data={'ticker_symbol': 'AAPL'})
        assert response.status_code == 302
        assert '/login' in response.headers['Location']
