"""add user table and portfolio ownership

Handles three cases so this works both for a brand-new install and for an
existing pre-auth database:
  1. Fresh clone, no tables at all: create `user` and `portfolios` from
     scratch, with `user_id` already NOT NULL - there's no legacy data to
     backfill.
  2. Existing database from before multi-user support (`portfolios` exists,
     no `user_id` column): add the column nullable, backfill every existing
     row to a new "legacy_owner" account, then tighten it to NOT NULL.
  3. Already up to date: each step below checks the live schema first and
     is a no-op if there's nothing to do.

Revision ID: 061d6a446160
Revises:
Create Date: 2026-09-08 23:44:21.774722

"""
import secrets

from alembic import op
import sqlalchemy as sa
from werkzeug.security import generate_password_hash

# revision identifiers, used by Alembic.
revision = '061d6a446160'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if 'user' not in existing_tables:
        op.create_table(
            'user',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('username', sa.String(length=80), nullable=False),
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('password_hash', sa.String(length=255), nullable=False),
            sa.Column('date_created', sa.DateTime(), nullable=True),
        )
        op.create_index(op.f('ix_user_username'), 'user', ['username'], unique=True)
        op.create_index(op.f('ix_user_email'), 'user', ['email'], unique=True)

    if 'portfolios' not in existing_tables:
        # Brand-new install: create the final schema directly, user_id
        # NOT NULL from the start since there are no pre-existing rows.
        op.create_table(
            'portfolios',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(length=100), nullable=False),
            sa.Column('stocks', sa.String(length=500), nullable=False),
            sa.Column('description', sa.String(length=200), nullable=True),
            sa.Column('weights', sa.String(length=500), nullable=True),
            sa.Column('long_only', sa.Boolean(), nullable=False),
            sa.Column('date_created', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='fk_portfolios_user_id_user'),
        )
        op.create_index(op.f('ix_portfolios_user_id'), 'portfolios', ['user_id'], unique=False)
        return

    portfolio_columns = {c['name'] for c in inspector.get_columns('portfolios')}
    if 'user_id' in portfolio_columns:
        return  # already migrated

    # Existing pre-auth database: add the column nullable first, since
    # SQLite (via batch mode) rebuilds the table by copying existing rows
    # through it, and a NOT NULL column with no default would reject every
    # pre-existing row before we get a chance to backfill them below.
    with op.batch_alter_table('portfolios', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_portfolios_user_id'), ['user_id'], unique=False)
        batch_op.create_foreign_key('fk_portfolios_user_id_user', 'user', ['user_id'], ['id'])

    # Assign any pre-existing, ownerless portfolios (from before multi-user
    # support existed) to a one-off "legacy" account instead of deleting
    # them.
    orphaned = bind.execute(sa.text('SELECT COUNT(*) FROM portfolios WHERE user_id IS NULL')).scalar()
    if orphaned:
        password = secrets.token_urlsafe(24)
        password_hash = generate_password_hash(password)
        result = bind.execute(
            sa.text(
                'INSERT INTO user (username, email, password_hash, date_created) '
                'VALUES (:username, :email, :password_hash, CURRENT_TIMESTAMP)'
            ),
            {
                'username': 'legacy_owner',
                'email': 'legacy_owner@example.invalid',
                'password_hash': password_hash,
            },
        )
        legacy_user_id = result.lastrowid
        bind.execute(
            sa.text('UPDATE portfolios SET user_id = :user_id WHERE user_id IS NULL'),
            {'user_id': legacy_user_id},
        )
        print(
            f"\n[migration] {orphaned} pre-existing portfolio(s) assigned to a new "
            f"'legacy_owner' account.\n"
            f"[migration] Username: legacy_owner\n"
            f"[migration] Password: {password}\n"
            f"[migration] Log in with these credentials once, then either change the "
            f"password or create a real account and re-point these portfolios at it.\n"
        )

    with op.batch_alter_table('portfolios', schema=None) as batch_op:
        batch_op.alter_column('user_id', existing_type=sa.Integer(), nullable=False)


def downgrade():
    with op.batch_alter_table('portfolios', schema=None) as batch_op:
        batch_op.drop_constraint('fk_portfolios_user_id_user', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_portfolios_user_id'))
        batch_op.drop_column('user_id')
    op.drop_index(op.f('ix_user_email'), table_name='user')
    op.drop_index(op.f('ix_user_username'), table_name='user')
    op.drop_table('user')
