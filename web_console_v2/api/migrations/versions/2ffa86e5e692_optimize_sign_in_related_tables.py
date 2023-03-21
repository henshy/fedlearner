"""optimize sign-in related tables

Revision ID: 2ffa86e5e692
Revises: ec68faa511cc
Create Date: 2021-10-22 12:15:10.710910

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2ffa86e5e692'
down_revision = 'ec68faa511cc'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('session_v2', sa.Column('user_id', sa.Integer(), nullable=False, comment='for whom the session is created'))
    op.add_column('users_v2', sa.Column('last_sign_in_at', sa.DateTime(timezone=True), nullable=True, comment='the last time when user tries to sign in'))
    op.add_column('users_v2', sa.Column('failed_sign_in_attempts', sa.Integer(), nullable=False, comment='failed sign in attempts since last successful sign in'))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('users_v2', 'failed_sign_in_attempts')
    op.drop_column('users_v2', 'last_sign_in_at')
    op.drop_column('session_v2', 'user_id')
    # ### end Alembic commands ###