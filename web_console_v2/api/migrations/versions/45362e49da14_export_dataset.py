"""export dataset

Revision ID: 45362e49da14
Revises: c3e83aed516c
Create Date: 2022-11-09 16:55:43.572595

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '45362e49da14'
down_revision = 'c3e83aed516c'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('trusted_jobs_v2', sa.Column('export_dataset_id', sa.Integer(), nullable=True, comment='export dataset id'))
    op.drop_column('trusted_jobs_v2', 'dataset_job_id')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('trusted_jobs_v2', sa.Column('dataset_job_id', mysql.INTEGER(), autoincrement=False, nullable=True, comment='dataset job id'))
    op.drop_column('trusted_jobs_v2', 'export_dataset_id')
    # ### end Alembic commands ###