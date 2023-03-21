"""add_crd_info2job

Revision ID: e4c8d7a2cf34
Revises: bf5e0cdc3e49
Create Date: 2022-01-13 14:34:54.149554

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e4c8d7a2cf34'
down_revision = 'bf5e0cdc3e49'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('job_v2', sa.Column('crd_kind', sa.String(length=255), nullable=True, comment='kind'))
    op.add_column('job_v2', sa.Column('crd_meta', sa.Text(), nullable=True, comment='metadata'))
    op.add_column('job_v2', sa.Column('snapshot', sa.Text(length=16777215), nullable=True, comment='snapshot'))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('job_v2', 'snapshot')
    op.drop_column('job_v2', 'crd_meta')
    op.drop_column('job_v2', 'crd_kind')
    # ### end Alembic commands ###