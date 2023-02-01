import { separateLng } from 'i18n/helpers';

const project = {
  no_result: { zh: '暂无工作区', en: 'No workspace yet' },
  create: { zh: '创建工作区', en: 'Create workspace' },
  describe: {
    zh:
      '提供工作区的新增和管理功能，支持对工作区进行新增、编辑、查询、删除功能，可查看一个工作区下的联邦工作流任务列表、模型列表、API列表，一个工作区下可创建多个联邦工作流任务。',
    en:
      'Provide workspace addition and management functions, support adding, editing, querying, and deleting workspaces. You can view the federal workflow task list, model list, and API list under a workspace. Multiple federal tasks can be created under a workspace Stream tasks.',
  },
  search_placeholder: {
    zh: '输入工作区名称关键词搜索',
    en: 'Enter the workspace name or keyword to search',
  },
  display_card: { zh: '卡片视图', en: 'Card view' },
  display_list: { zh: '表格视图', en: 'Table view' },
  connection_status_success: { zh: '成功', en: 'Success' },
  connection_status_waiting: { zh: '待检查', en: 'To be checked' },
  connection_status_checking: { zh: '检查中', en: 'Checking' },
  connection_status_failed: { zh: '失败', en: 'Failed' },
  connection_status_check_failed: { zh: '请重新检查' },
  action_edit: { zh: '编辑', en: 'Edit' },
  action_detail: { zh: '详情', en: 'Detail' },
  check_connection: { zh: '检查连接', en: 'Check connection' },
  create_work_flow: { zh: '创建工作流', en: 'Create a workflow' },
  connection_status: { zh: '连接状态', en: 'Connection status' },
  workflow_number: { zh: '工作流任务数量', en: 'Total workflows' },
  name: { zh: '工作区名称', en: 'workspace name' },
  participant_name: { zh: '合作伙伴名称', en: 'Participant name' },
  participant_url: { zh: '合作伙伴节点地址', en: 'Participant node address' },
  participant_domain: { zh: '合作伙伴泛域名', en: "Participant participant's domain" },
  selft_domain: { zh: '本侧泛域名', en: 'Self domain name' },
  remarks: { zh: '说明描述', en: 'Remarks' },
  name_placeholder: { zh: '请填写工作区名称', en: 'Please enter name' },
  participant_name_placeholder: { zh: '请输入合作伙伴名称', en: 'Please enter participant name' },

  participant_domain_placeholder: { zh: '请填写泛域名', en: 'Please enter domain' },
  remarks_placeholder: { zh: '请填写说明描述', en: 'Please enter remarks' },
  name_message: { zh: '请填写工作区名称', en: 'Please enter name' },
  participant_name_message: { zh: '请填写合作伙伴名称', en: 'Please enter participant name' },
  participant_url_message: {
    zh: '请填写合作伙伴节点地址',
    en: 'Please enter participant node address',
  },
  edit: { zh: '编辑工作区', en: 'Edit workspace' },
  workflow: { zh: '工作流任务', en: 'Workflow task' },
  mix_dataset: { zh: '融合数据集', en: 'Fusion data set' },
  model: { zh: '模型', en: 'Model' },
  creator: { zh: '创建者', en: 'Creator' },
  creat_time: { zh: '创建时间', en: 'Creation time' },
  add_parameters: { zh: '添加参数', en: 'Add parameters' },
  env_path_config: { zh: '环境变量参数配置', en: 'Expand environment variable configuration' },
  show_env_path_config: { zh: '展开环境变量配置', en: 'Environment variable configuration' },
  hide_env_path_config: {
    zh: '收起环境变量配置',
    en: 'Collapse environment variable configuration',
  },
  basic_information: { zh: '基本信息', en: 'Basic Information' },
  participant_information: { zh: '合作伙伴信息', en: 'Participant information' },
  upload_certificate: { zh: '上传证书', en: 'Upload certificate' },
  backend_config_certificate: { zh: '后台手动配置', en: 'Manually configure' },
  upload_certificate_placeholder: {
    zh: '请上传gz格式文件，大小不超过20MB',
    en: 'Please upload a file in gz format, no more than 20MB in size',
  },
  upload_certificate_message: { zh: '请上传证书', en: 'Please upload the certificate' },
  drag_to_upload: { zh: '拖拽到这里进行上传', en: 'Drag and drop here to upload' },
  create_success: { zh: '创建工作区成功', en: 'Create workspace succeed!' },
  label_token: { zh: '联邦密码' },

  edit_success: { zh: '编辑工作区成功', en: 'Edit workspace succeed!' },
  msg_var_name: { zh: '请输入变量名' },
  msg_var_value: { zh: '请输入变量值' },
  msg_sure_2_cancel: { zh: '确认取消？' },
  msg_effect_of_cancel: { zh: '取消后，已填写内容将不再保留' },
  msg_domian_required: { zh: '请补全泛域名' },
  msg_domian_invalid: { zh: '只允许小写英文字母/中划线/数字，请检查' },
  msg_ip_addr_invalid: { zh: 'IP 地址不合法，请检查' },
  msg_no_var_yet: { zh: '当前没有环境变量参数，请添加' },
  msg_token_required: { zh: '联邦密码为必填项' },
  msg_token_invalid: { zh: '只允许英文、数字的组合' },
  placeholder_global_project_filter: { zh: '选择特定工作区筛选资源' },
  placeholder_no_project: { zh: '暂无工作区' },
  placeholder_domain_name: { zh: '泛域名间值' },
  placeholder_token: { zh: '请输入联邦密码' },
  placeholder_participant_url: {
    zh: 'IPv4/v6 地址（包含端口）',
    en: 'IP(v4 or v6) address with Port',
  },

  label_type_light_client: { zh: '轻量级', en: 'Light client' },
  label_type_platform: { zh: '标准', en: 'Platform' },
};

export default separateLng(project);
