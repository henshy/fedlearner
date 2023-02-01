import { WorkflowTemplate } from 'typings/workflow';
import { JobType } from 'typings/job';
import { VariableAccessMode, VariableComponent, VariableValueType } from 'typings/variable';
import { DeepPartial } from 'utility-types';
import { gloabalVariables, stringInput, objectInput, listInput } from '../variables/examples';

export const normalTemplate: DeepPartial<WorkflowTemplate> = {
  id: 1,
  name: 'Test template',
  group_alias: 'foo group',
  config: {
    group_alias: 'foo group',
    variables: gloabalVariables,
    job_definitions: [
      {
        name: 'Initiative',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [],
        variables: [
          {
            name: 'job_name',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
              type: 'string',
              required: true,
            },
          },
          {
            name: 'participant',
            value: '',
            access_mode: VariableAccessMode.PEER_READABLE,
            widget_schema: {
              component: VariableComponent.Input,
              type: 'string',
              required: true,
            },
          },
          {
            name: 'job_type',
            value: '1',
            access_mode: VariableAccessMode.PEER_READABLE,
            widget_schema: {
              component: VariableComponent.Select,
              options: {
                type: 'static',
                source: [1, 2],
              },
              multiple: true,
              type: 'number',
              required: true,
            },
          },
          {
            name: 'is_pair',
            value: '',
            access_mode: VariableAccessMode.PRIVATE,
            widget_schema: {
              component: VariableComponent.Switch,
              type: 'boolean',
            },
          },
          {
            name: 'comment',
            value: '',
            access_mode: VariableAccessMode.PEER_READABLE,
            widget_schema: {
              component: VariableComponent.TextArea,
              rows: 4,
              type: 'string',
            },
          },
          {
            name: 'cpu_limit',
            value: '10',
            access_mode: VariableAccessMode.PRIVATE,
            widget_schema: {
              component: VariableComponent.NumberPicker,
              min: 1,
              max: 80,
              type: 'number',
            },
          },
        ],
      },
      {
        name: 'Raw data upload',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [{ source: 'Initiative' }],
        variables: [
          {
            name: 'job_name',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
              type: 'string',
            },
          },
          {
            name: 'comment2',
            value: '',
            access_mode: VariableAccessMode.PRIVATE,
            widget_schema: {
              component: VariableComponent.TextArea,
              rows: 4,
              type: 'string',
              required: true,
            },
          },
        ],
      },
      {
        name: 'Raw data process',
        job_type: JobType.NN_MODEL_TRANINING,
        is_federated: true,
        dependencies: [{ source: 'Initiative' }],
        variables: [
          {
            name: 'job_name',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
              type: 'string',
              required: true,
            },
          },
        ],
      },
      {
        name: 'Raw data save',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [{ source: 'Initiative' }],
        variables: [
          {
            name: 'job_name',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
              type: 'string',
            },
          },
        ],
      },
      {
        name: 'Training',
        job_type: JobType.NN_MODEL_TRANINING,
        is_federated: true,
        dependencies: [
          { source: 'Raw data upload' },
          { source: 'Raw data process' },
          { source: 'Raw data save' },
        ],
        variables: [
          {
            name: 'job_name',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
              type: 'string',
            },
          },
        ],
      },
      {
        name: 'Finish/clear',
        job_type: JobType.TREE_MODEL_EVALUATION,
        is_federated: true,
        dependencies: [{ source: 'Training' }],
        variables: [
          {
            name: 'job_name',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
              type: 'string',
            },
          },
        ],
      },
    ],
  },
};

export const complexDepsTemplate: DeepPartial<WorkflowTemplate> = {
  id: 2,
  name: 'Complex deps template',
  group_alias: 'c-group',
  config: {
    group_alias: 'c-group',
    variables: [],
    job_definitions: [
      {
        name: 'Initiative',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [],
        variables: [
          {
            name: 'job_name',
            value: '',
            value_type: VariableValueType.STRING,
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
              required: true,
            },
          },
        ],
      },
      {
        name: 'Raw data upload',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [{ source: 'Initiative' }],
        variables: [
          {
            name: 'input_dir',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            value_type: VariableValueType.STRING,
            widget_schema: {
              component: VariableComponent.Dataset,
            },
          },
        ],
      },

      {
        name: 'Raw data process',
        job_type: JobType.NN_MODEL_TRANINING,
        is_federated: true,
        dependencies: [{ source: 'Initiative' }],
        variables: [
          {
            name: 'codes',
            value: JSON.stringify({
              'foo.py': 'int a = 1',
              'folder/bar.py': 'bool b = True',
            }),
            access_mode: VariableAccessMode.PEER_WRITABLE,
            value_type: VariableValueType.CODE,
            widget_schema: {
              component: VariableComponent.Code,
              required: true,
            },
          },
        ],
      },
      {
        name: 'Raw data save',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [{ source: 'Initiative' }],
        variables: [
          {
            name: 'worker_number',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            value_type: VariableValueType.STRING,
            widget_schema: {
              component: VariableComponent.Select,
              enum: [1, 2, 3, 4, 5],
            },
          },
        ],
      },
      {
        name: 'Training',
        job_type: JobType.NN_MODEL_TRANINING,
        is_federated: true,
        dependencies: [{ source: 'Raw data upload' }, { source: 'Raw data process' }],
        variables: [
          {
            name: 'job_name',
            value: '',
            access_mode: VariableAccessMode.PEER_WRITABLE,
            value_type: VariableValueType.STRING,
            widget_schema: {
              component: VariableComponent.Input,
            },
          },
        ],
      },
      {
        name: 'Finish/clear',
        job_type: JobType.TREE_MODEL_EVALUATION,
        is_federated: true,
        dependencies: [{ source: 'Training' }, { source: 'Raw data save' }],
        variables: [
          {
            name: 'job_name',
            value: '',
            value_type: VariableValueType.STRING,
            access_mode: VariableAccessMode.PEER_WRITABLE,
            widget_schema: {
              component: VariableComponent.Input,
            },
          },
        ],
      },
    ],
  },
};

export const xShapeTemplate: DeepPartial<WorkflowTemplate> = {
  id: 3,
  name: 'X Shape template',
  group_alias: 'x-group',
  is_local: true,
  config: {
    group_alias: 'x-group',
    variables: [],
    job_definitions: [
      {
        name: '1-1',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        variables: [],
      },
      {
        name: '1-2',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        variables: [],
      },
      {
        name: '2-1',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [{ source: '1-2' }],
        variables: [],
      },
      {
        name: '2-2',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [{ source: '1-1' }],
        variables: [],
      },
    ],
  },
};

export const localTpl = {
  id: 4,
  name: 'local template',
  comment: 'Comment here',
  is_local: true,
  group_alias: 'test-2',
  config: {
    group_alias: 'test-2',
    job_definitions: [
      {
        name: 'Initiative',
        type: 'RAW_DATA',
        is_federated: true,
        variables: [
          {
            name: 'job_name',
            access_mode: 'PEER_WRITABLE',
            widget_schema: '{"component":"Input","type":"string","required":true}',
            value: '',
          },
        ],
        dependencies: [],
        yaml_template: '',
      },
      {
        name: 'Raw data upload',
        type: 'RAW_DATA',
        is_federated: true,
        variables: [
          {
            name: 'job_name2',
            access_mode: 'PEER_WRITABLE',
            widget_schema: '{"component":"Input","type":"string"}',
            value: '',
          },
          {
            name: 'comment2',
            access_mode: 'PRIVATE',
            widget_schema: '{"component":"TextArea","rows":4,"type":"string","required":true}',
            value: '',
          },
        ],
        dependencies: [
          {
            source: 'Initiative',
            type: 3,
          },
        ],
        yaml_template: '',
      },
      {
        name: 'Training',
        type: 'RAW_DATA',
        is_federated: true,
        variables: [
          {
            name: 'job_name2',
            access_mode: 'PEER_READABLE',
            widget_schema: '{"component":"Input","type":"string"}',
            value: '',
          },
        ],
        dependencies: [
          {
            source: 'Raw data upload',
            type: 'ON_COMPLETE',
          },
        ],
        yaml_template: '',
      },
    ],
  },
};

export const withTypedValueTemplate: DeepPartial<WorkflowTemplate> = {
  id: 5,
  name: 'with typed value template',
  group_alias: 'typed value group',
  config: {
    group_alias: 'typed value group',
    variables: gloabalVariables,
    job_definitions: [
      {
        name: 'Initiative',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [],
        variables: [stringInput, objectInput, listInput],
      },
    ],
  },
};

export const noTypedValueTemplate: DeepPartial<WorkflowTemplate> = {
  id: 6,
  name: 'no typed value template',
  group_alias: 'typed value group',
  config: {
    group_alias: 'typed value group',
    variables: [],
    job_definitions: [
      {
        name: 'Initiative',
        job_type: JobType.RAW_DATA,
        is_federated: true,
        dependencies: [],
        variables: [
          { ...stringInput, typed_value: undefined },
          { ...objectInput, typed_value: undefined },
          { ...listInput, typed_value: undefined },
        ],
      },
    ],
  },
};
