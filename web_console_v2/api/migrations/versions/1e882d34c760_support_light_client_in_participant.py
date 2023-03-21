"""add_type_and_last_connected_at_to_participant

Revision ID: 1e882d34c760
Revises: 93d756004237
Create Date: 2022-01-04 16:33:03.565990

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1e882d34c760'
down_revision = '93d756004237'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('participants_v2', sa.Column('last_connected_at', sa.DateTime(timezone=True), nullable=True, comment='last connected at'))
    op.add_column('participants_v2', sa.Column('participant_type', sa.Enum('PLATFORM', 'LIGHT_CLIENT', name='participanttype', native_enum=False, create_constraint=False, length=32), default='PLATFORM', nullable=True, comment='participant type'))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('participants_v2', 'participant_type')
    op.drop_column('participants_v2', 'last_connected_at')
    # ### end Alembic commands ###