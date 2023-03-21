"""add_metric_is_public_to_model_job

Revision ID: 0f24b50d4077
Revises: 88f6dd8bcb23
Create Date: 2022-05-27 18:24:06.742262

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0f24b50d4077'
down_revision = '88f6dd8bcb23'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('model_jobs_v2', sa.Column('metric_is_public', sa.Boolean(), nullable=True, comment='is metric public'))
    op.execute('UPDATE model_jobs_v2 SET metric_is_public = true')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('model_jobs_v2', 'metric_is_public')
    # ### end Alembic commands ###