from portfolio_optimizer.models import db, User


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


class TestProtectedRoutesRequireLogin:
    def test_index_redirects_anonymous_users_to_login(self, client):
        response = client.get('/')
        assert response.status_code == 302
        assert '/login' in response.headers['Location']

    def test_analyze_stock_requires_login(self, client):
        response = client.post('/analyze_stock', data={'ticker_symbol': 'AAPL'})
        assert response.status_code == 302
        assert '/login' in response.headers['Location']
