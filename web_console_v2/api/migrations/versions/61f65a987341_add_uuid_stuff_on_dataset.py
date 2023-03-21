"""add_uuid_stuff_on_dataset

Revision ID: 61f65a987341
Revises: e3166ab65528
Create Date: 2022-02-10 20:17:11.998292

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '61f65a987341'
down_revision = 'e3166ab65528'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('datasets_v2', sa.Column('uuid', sa.String(length=255), nullable=True, comment='dataset uuid'))
    op.add_column('datasets_v2',
                  sa.Column('is_published', sa.Boolean(), nullable=True, comment='dataset is published or not'))
    op.add_column(
        'datasets_v2',
        sa.Column('dataset_kind',
                  sa.Enum('RAW', 'PROCESSED', 'SOURCE', 'EXPORTED', name='datasetkindv2', native_enum=False, length=32, create_constraint=False),
                  nullable=True,
                  comment='new version of dataset kind, choices [raw, processed, ...]'))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('datasets_v2', 'dataset_kind')
    op.drop_column('datasets_v2', 'is_published')
    op.drop_column('datasets_v2', 'uuid')
    # ### end Alembic commands ###