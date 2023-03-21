"""add group id to serving models

Revision ID: 0d6eb48014d8
Revises: 5f322c9d67ea
Create Date: 2022-08-03 20:12:41.173325

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0d6eb48014d8'
down_revision = '5f322c9d67ea'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('model_groups_v2', sa.Column('ticket_uuid', sa.String(length=255), nullable=True, comment='review ticket uuid, empty if review function is disable'))
    op.add_column('model_groups_v2', sa.Column('ticket_status', sa.Enum('PENDING', 'APPROVED', 'DECLINED', name='ticketstatus', native_enum=False, length=64, create_constraint=False), nullable=True, comment='review ticket status'))
    op.add_column('model_groups_v2', sa.Column('cron_job_global_config', sa.Text(length=16777215), nullable=True, comment='global config for cron job'))
    op.add_column('model_groups_v2', sa.Column('algorithm_uuid_list', sa.Text(length=16777215), nullable=True, comment='algorithm project uuid for all participants'))
    op.add_column('model_groups_v2', sa.Column('status', sa.Enum('PENDING', 'FAILED', 'SUCCEEDED', name='groupcreatestatus', native_enum=False, length=64, create_constraint=False), nullable=True, comment='create status'))
    op.add_column('model_jobs_v2', sa.Column('global_config', sa.Text(length=16777215), nullable=True, comment='global_config'))
    op.add_column('model_jobs_v2', sa.Column('status', sa.Enum('PENDING', 'READY', 'ERROR', 'RUNNING', 'STOPPED', 'SUCCEEDED', 'FAILED', name='modeljobstatus', native_enum=False, length=64, create_constraint=False), nullable=True, comment='model job status'))
    op.add_column('model_jobs_v2', sa.Column('auth_status', sa.Enum('PENDING', 'AUTHORIZED', name='authstatus', native_enum=False, length=64, create_constraint=False), nullable=True, comment='authorization status'))
    op.add_column('model_jobs_v2', sa.Column('error_message', sa.Text(), nullable=True, comment='error message'))
    op.add_column('serving_deployments_v2', sa.Column('deploy_platform', sa.String(length=255), nullable=True, comment='deploy platform. None means inside this platform'))
    op.add_column('serving_models_v2', sa.Column('model_group_id', sa.Integer(), nullable=True, comment='model group id for auto update scenario'))
    op.add_column('serving_models_v2', sa.Column('pending_model_id', sa.Integer(), nullable=True, comment="model id when waiting for participants' config"))
    op.add_column('serving_models_v2', sa.Column('pending_model_group_id', sa.Integer(), nullable=True, comment="model group id when waiting for participants' config"))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('serving_models_v2', 'pending_model_group_id')
    op.drop_column('serving_models_v2', 'pending_model_id')
    op.drop_column('serving_models_v2', 'model_group_id')
    op.drop_column('serving_deployments_v2', 'deploy_platform')
    op.drop_column('model_jobs_v2', 'error_message')
    op.drop_column('model_jobs_v2', 'auth_status')
    op.drop_column('model_jobs_v2', 'status')
    op.drop_column('model_jobs_v2', 'global_config')
    op.drop_column('model_groups_v2', 'status')
    op.drop_column('model_groups_v2', 'algorithm_uuid_list')
    op.drop_column('model_groups_v2', 'cron_job_global_config')
    op.drop_column('model_groups_v2', 'ticket_status')
    op.drop_column('model_groups_v2', 'ticket_uuid')
    # ### end Alembic commands ###