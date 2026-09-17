"""add portfolio snapshot history

Written by hand rather than via `flask db migrate --autogenerate`: since
db.create_all() runs on every app boot (including when the `flask db ...`
CLI itself loads the app), a brand-new table like this one is typically
already created by the time autogenerate runs, so it detects no schema
diff to generate. Checks table existence first so this is still correct
and idempotent for a database where db.create_all() got here first, and
for a genuinely fresh database where it hasn't.

Revision ID: a0adad2a2214
Revises: 061d6a446160
Create Date: 2026-09-16 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a0adad2a2214'
down_revision = '061d6a446160'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if 'portfolio_snapshot' not in inspector.get_table_names():
        op.create_table(
            'portfolio_snapshot',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('portfolio_id', sa.Integer(), nullable=False),
            sa.Column('strategy', sa.String(length=50), nullable=False),
            sa.Column('weights', sa.String(length=1000), nullable=False),
            sa.Column('portfolio_return', sa.Float(), nullable=True),
            sa.Column('volatility', sa.Float(), nullable=True),
            sa.Column('sharpe_ratio', sa.Float(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id'],
                                     name='fk_portfolio_snapshot_portfolio_id_portfolios'),
        )
        op.create_index(op.f('ix_portfolio_snapshot_portfolio_id'), 'portfolio_snapshot',
                         ['portfolio_id'], unique=False)
        op.create_index(op.f('ix_portfolio_snapshot_created_at'), 'portfolio_snapshot',
                         ['created_at'], unique=False)


def downgrade():
    op.drop_table('portfolio_snapshot')
