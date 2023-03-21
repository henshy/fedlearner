"""create_models_table

Revision ID: 9a9b20f3804e
Revises: e5d91f0f59a7
Create Date: 2021-07-20 22:41:51.294181

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '9a9b20f3804e'
down_revision = 'e5d91f0f59a7'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('models_v2',
    sa.Column('id', sa.Integer(), nullable=False, comment='id'),
    sa.Column('name', sa.String(length=255), nullable=True, comment='name'),
    sa.Column('version', sa.Integer(), nullable=True, default=0, comment='version'),
    sa.Column('type', sa.Integer(), nullable=True, comment='type'),
    sa.Column('state', sa.Integer(), nullable=True, comment='state'),
    sa.Column('job_name', sa.String(length=255), nullable=True, comment='job_name'),
    sa.Column('parent_id', sa.Integer(), nullable=True, comment='parent_id'),
    sa.Column('params', sa.Text(), nullable=True, comment='params'),
    sa.Column('metrics', sa.Text(), nullable=True, comment='metrics'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True, comment='created_at'),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True, comment='updated_at'),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True, comment='deleted_at'),
    sa.Column('group_id', sa.Integer(), nullable=True, comment='group_id'),
    sa.Column('extra', sa.Text(), nullable=True, comment='extra'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('job_name', name='uniq_job_name'),
    comment='model',
    mysql_charset='utf8mb4',
    mysql_engine='innodb'
    )
    op.create_index('idx_job_name', 'models_v2', ['job_name'], unique=False)
    op.alter_column('workflow_v2', 'extra',
               existing_type=mysql.TEXT(),
               comment='json string that will be send to peer',
               existing_comment='extra',
               existing_nullable=True)
    op.alter_column('workflow_v2', 'local_extra',
               existing_type=mysql.TEXT(),
               comment='json string that will only be store locally',
               existing_comment='local_extra',
               existing_nullable=True)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('workflow_v2', 'local_extra',
               existing_type=mysql.TEXT(),
               comment='local_extra',
               existing_comment='json string that will only be store locally',
               existing_nullable=True)
    op.alter_column('workflow_v2', 'extra',
               existing_type=mysql.TEXT(),
               comment='extra',
               existing_comment='json string that will be send to peer',
               existing_nullable=True)
    op.drop_index('idx_job_name', table_name='models_v2')
    op.drop_table('models_v2')
    # ### end Alembic commands ###