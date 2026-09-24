# Azure 日本东部资源准备记录

> **最新实施状态（2026-09-23）**：CPU VM、ACR、项目双容器和私有数据库已部署，HTTPS、Stripe Sandbox 回调、真实 Blob 上传及 T4 模型就绪已验证。最新镜像、验收结果和未解决项以 [部署验收记录](AZURE_JAPAN_DEPLOYMENT_VALIDATION.md) 为准。下文保留初次门户创建时的记录，其中“尚未创建 CPU / 尚未上传模型”等描述为历史状态。

更新日期：2026-09-23。通过外部 Chrome 的 Azure Portal 操作。

## 预算约束

用户要求整体费用控制在 US$20 以内，GPU 按量计费、GPU 总额也在 US$20 以内。本轮按更保守口径处理：全部资源的部署和验证费用合计不超过 US$20，GPU 包含在其中；不采用 US$100/月常驻方案。

这不是 Azure 已设置的硬性消费封顶。预算告警不等于自动停止计费，费用数据也可能延迟。未配置持续运行的 CPU 主机；已启动 GPU 测试容器进行基础验证，费用尚待账单数据更新，不能据此宣称实际费用为零。

已创建资源组范围的预算 `videotranslator-validation-20`。订阅实际预算货币是 **CAD**，已保留更保守的 **CAD 20**，没有上调为等值美元。周期为 Annually，2026-09-01 至 2027-08-31；实际费用达到 50% / 80% / 100%（CAD 10 / 16 / 20）时，发送中文邮件至 `c43892@gmail.com`。没有自动停机动作，资源组之外费用不纳入此预算。

## 已完成

- 订阅：Pay-As-You-Go。
- 区域：Japan East / japaneast。
- 已创建资源组：`videotranslator-jpe-rg`，门户返回创建成功。
- 门户在该区域列出 `Consumption-GPU-NC8as-T4`：1 张 GPU、8 vCPU、56 GiB RAM。
- 原始 Container App 创建向导的预验证通过；这不能替代实际 GPU 分配和模型运行测试。

## 基础部署已成功

用户已在门户点击 Create，部署名 `CustomDeployment-20260923125725`，环境创建操作时间为 2026-09-23 12:57:59（America/Toronto）。13:15 后门户返回 “Your deployment is complete”。环境初始化曾等待约 17 分钟，随后成功；VNet、子网、日志工作区和 GPU 验证应用均已创建。不要重复提交同名部署。

- Container Apps 环境：`videotranslator-jpe-env`。
- 工作负载配置：`gpu-t4`，类型 `Consumption-GPU-NC8as-T4`；保留默认 Consumption 配置。
- 容器应用：`videotranslator-gpu`。
- 验证镜像：`mcr.microsoft.com/k8se/gpu-quickstart:latest`，不是项目镜像。
- 副本数：`minReplicas: 0`、`maxReplicas: 1`，Single revision。
- VNet：`videotranslator-jpe-vnet`，`10.0.0.0/16`。
- ACA 子网：`aca-infrastructure`，`10.0.0.0/23`，委派给 `Microsoft.App/environments`。
- 环境 `internal: true`、`publicNetworkAccess: Disabled`。应用入口仅可从 VNet 访问，目标端口 80。
- 日志工作区：`workspacevideotranslatorjperg84b7`，按量计费、保留 30 天。
- 已在环境 Quota 页面核实：Managed Environment Consumption T4 Gpus，已用 0，上限 5。无需为此环境先申请配额；配额不代表已成功分配 GPU 或通过模型测试。
- 环境 Volume mounts 已成功保存 `data`、`models` 两个 SMB Read/Write 条目，门户返回 “Successfully updated azure files configuration”。首次保存因环境初始化未完成被拒绝，环境就绪后重试成功。它们是环境级存储声明，还需在项目容器修订中配置 `/data`、`/models` 挂载路径。
- 环境默认域名 `ashyflower-ffc9e717.japaneast.azurecontainerapps.io`，静态内网 IP `10.0.0.198`。
- 托管基础设施资源组 `ME_videotranslator-jpe-env_videotranslator-jpe-rg_japaneast`：其潜在费用不包含在当前项目资源组范围预算中，需要单独检查。

## GPU 基础运行验证

- 修订 `videotranslator-gpu--x3sln88`，创建时间 13:15:11（America/Toronto）。
- 系统日志显示官方验证镜像成功拉取，大小 7,530,872,832 bytes，用时 94.70 秒。
- 初始 Startup probe 失败出现在模型下载与加载期间；17:17:56 UTC 应用日志显示 `Model loaded successfully.`，随后 Flask 在端口 80 启动，修订状态转为 Running。
- 已在 Scale 页面核实：min 0 / max 1，cooldown 300 秒，polling 30 秒，HTTP scaling。当前配置还没有接入项目任务队列。
- 2026-09-23 13:22（America/Toronto）门户修订列表显示 **Scaled to 0**、**0 replicas**，已验证首次启动后自动缩回零副本。测试应用仅内网可访问，未对外产生业务请求。
- 以上只验证基础部署及测试镜像启动，不代表已测试 Demucs、IndexTTS2、YouTube 下载或业务任务。

## 尚未完成

- 配置项目容器的实际存储挂载及应用连接；容器镜像仓库和 CPU 主机的资源创建与连接。
- 项目镜像适配、推送和应用部署。
- 私有 DNS、跨容器文件、存储权限、数据库连接和任务触发缩容配置。
- T4 上 Demucs / IndexTTS2 验收，以及上传到结果下载的完整任务验证。

本轮 GPU 验证基础设施及存储准备已完成，项目完整部署和网站上线尚未完成。架构及后续改造见 `AZURE_JAPAN_T4_MIGRATION_PLAN.md`。

## 存储创建进展

已在预算授权后创建 `vtranslatorjpe43892` 存储账户，部署名 `vtranslatorjpe43892_1790180317533`。门户显示部署完成，资源 Provisioning state 为 Succeeded。

- 资源组 `videotranslator-jpe-rg`，Japan East。
- Standard / StorageV2，LRS，Hot；支持 Blob 和 Azure Files。
- 公网网络入口允许经认证的上传下载，Blob 匿名访问禁用；必须 HTTPS、最低 TLS 1.2；SMB 要求传输加密。
- Blob、容器和文件共享软删除保留 7 天。
- 未启用 SFTP、跨租户复制、版本控制或 Defender 付费附加项。
- 已创建私有 Blob 容器 `uploads`、`results`，门户均返回创建成功。
- 已创建 SMB 文件共享 `data`、`models`，Transaction optimized，未开通付费备份服务。
- `data`、`models` 容量上限均为 32 GiB，已在共享列表中核实；容量配额不等于消费金额硬限额。
- 未上传项目数据或模型。环境级存储连接已配置，尚未配置项目容器或 CPU 主机内的实际挂载。
