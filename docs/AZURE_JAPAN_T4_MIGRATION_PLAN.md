# Azure 日本东部 T4 按需 GPU 改造方案

日期：2026-09-23。状态：实施方案草案，基于当前代码、镜像元数据和官方文档；尚未验证订阅配额、云端容量、端到端运行或实际费用。本次只编写文档。

## 1. 已确定的目标与首期边界

- 使用 Azure 全球版 Japan East，日本东部；处理服务能够访问 YouTube。
- 网页、登录、上传、历史记录、余额、下载始终由独立 CPU 服务提供，GPU 归零不影响这些功能。
- GPU 使用 Azure Container Apps（ACA）Consumption GPU T4，minReplicas=0、maxReplicas=1，单任务串行。
- 保留当前视频处理算法、字幕、音频输出、时长对齐、警告和退款语义；优先减少迁移改动。
- 初期所有计算和存储放同一区域，面向大陆、其他东亚及北美分别做网络实测；不承诺所有网络下均可顺畅访问或下载 YouTube。
- “持续在线”在首期指不随 GPU 关闭，不代表单台 CPU 主机具备故障双活能力。若需要高可用 SLA，需另做数据库和多副本改造。

**首期以一个完整引擎任务为 GPU 占用单位，而非每一次 CUDA 运算。** 一个任务中的转写 API 等待、翻译 API 等待和引擎内 CPU 处理也会占用 GPU 副本时间。无任务并经过缩容冷却后归零。此取舍保留现有处理流程，第二阶段再根据实测拆分 GPU 阶段。

## 2. 代码核查结果及关键前置风险

| 发现 | 当前位置 | 对方案的影响 |
| --- | --- | --- |
| Studio 通过 docker exec/cp 提交、查询、收集引擎结果 | packages/videotranslator/videotranslator/adapters/docker_engine.py、engine_bridge.py | 新增私有引擎接口和适配器，云端网页容器不挂 Docker socket |
| Studio 默认使用 SQLite；本地检查任务也有 SQLite 队列 | bootstrap.py、docstore.py、adapters/local_jobs.py | 首期放 CPU VM 的持久化块磁盘，避免同时改写事务存储 |
| 引擎已有 PostgreSQL、Redis、Celery、全局执行锁 | vt-data/engine-source/compose.yaml、backend/videotranslator/tasks.py | 复用任务队列和串行锁；GPU 伸缩以 PostgreSQL 持久任务为依据 |
| GPU 只有 tts 容器使用，包含 Demucs 和 IndexTTS2；引擎 worker 为另一 Python 环境 | services/tts/、containers/app.Dockerfile、containers/tts.Dockerfile | 两镜像同副本部署，保留 Python 3.12 与 3.10 的隔离 |
| 启动及接单检查依赖 GPU /health | bootstrap.py、engine_bridge.py、tasks.py | 必须区分“控制服务可用”和“GPU 正在冷启动”，否则归零后无法接单 |
| Demucs 子任务字典只在内存中 | services/tts/demucs_service.py | 重启恢复不能仅依赖这个字典；要校验持久结果并安全重跑未完成阶段 |
| Blob 适配器已有上传下载，但签名实现仍需修正 | adapters/azure_blob.py | 托管身份需获取 user delegation key 再签 SAS；下载改为流式 |
| 旧 azure profile 实际指向 Azure ML 和旧 worker 入口 | bootstrap.py、adapters/azure_ml.py、deploy/README.md | 新增独立 profile，不把旧 Azure 代码当作当前引擎已上线 |
| 当前源码转写走外部 OpenAI 接口 | backend/videotranslator/transcription.py、tasks.py 及 synthesis-limits.patch | 首期保留实际行为，不能把旧镜像的本地 Whisper 方案或旧文档当作现状 |

代码基线以 `vt-data/engine-source` 当前恢复的引擎源码，加 `deploy/Dockerfile.engine-patched`、`deploy/Dockerfile.tts-patched` 的补丁链为准。正式构建前记录源码提交、未提交补丁、模型版本和镜像 digest；不能直接改 ignored worktree 后就认为构建可复现。

### 必须先解决：镜像体积

本机只读检查 `docker image inspect` 得到：

- `videotranslator-app:studio`：1,569,210,524 bytes，约 1.46 GiB。
- `videotranslator-indextts2:studio`：14,356,984,301 bytes，约 13.37 GiB。
- 两镜像 Size 简单相加约 14.83 GiB；它不是注册表压缩下载大小，也未扣除共享层。

ACA 通用容器文档列出 Consumption 每副本镜像总大小 8 GB 的限制，而 GPU 指南又讨论 5–15 GB GPU 镜像，文档适用范围存在需要确认之处。[容器限制](https://learn.microsoft.com/en-us/azure/container-apps/containers#limitations)、[GPU 镜像指南](https://learn.microsoft.com/en-us/azure/container-apps/functions-gpu-container-apps)。

**不能承诺当前镜像原样可部署。** 阶段 0 应确认日本 T4 对镜像大小的实际限制、计量口径和拉取行为；优先用多阶段构建去除编译工具、重复依赖层、缓存和测试文件，模型单独存储。在目标环境完成真实拉取与启动验证。若仍不满足限制，暂停主体云改造并调整打包方案；不默认购买 Dedicated GPU 来绕过限制。

## 3. 推荐部署拓扑

| 组件 | 首期部署 | 生命周期及职责 |
| --- | --- | --- |
| HTTPS 入口、Studio、后台协调与媒体检查 | Japan East 普通 CPU VM 上的 Compose | 持续运行，负责网页、认证、计费、YouTube 导入和结果发布 |
| 私有 engine-control、PostgreSQL、Redis、Celery beat | 同一 CPU VM，私有容器网络 | 持续运行，保存任务、派发、恢复、取消与状态查询 |
| TTS/Demucs 容器（第一个容器） | ACA T4 GPU app 的主容器 | 取得 GPU，挂载 /data 和 /models |
| Celery engine worker（第二个容器） | 同一个 ACA GPU app 的伴随容器 | 无 GPU 分配，驱动现有 pipeline，通过 localhost:8001 调用 GPU 服务 |
| 原始上传、已验证最终结果和字幕 | Japan East 私有 Azure Blob | 独立于 GPU，供浏览器直传、签名播放及下载 |
| 引擎工作文件及模型缓存 | Japan East Azure Files，分别设共享目录 | /data 在 CPU 与 GPU 端一致；/models 保留模型，归零后仍存在 |
| SQLite、PostgreSQL、Redis 持久数据 | CPU VM 数据盘及独立备份 | 不放容器临时层，不放 Azure Files/SMB 上的 SQLite 数据文件 |

选择 CPU VM 的原因是当前 SQLite、子进程媒体检查和 Compose 已可复用；避免在首次上云时再增加 Studio 数据库迁移。该 VM 不带 GPU，独立持续计费。参考起点为 4 vCPU / 8–16 GiB，需按导入、导出及数据库实测确定，不是已核实的 SKU 或报价。CPU 重型工作单独限并发，保证网页有资源。

ACA 支持同一 app 中多个紧耦合容器；它们共享网络和生命周期。GPU 文档规定只有第一个容器获得 GPU，因此顺序必须明确。TTS 与 worker 的 CPU/内存分配及平台限制需在阶段 0 验证。[多容器](https://learn.microsoft.com/en-us/azure/container-apps/containers#multiple-containers)、[GPU 约束](https://learn.microsoft.com/en-us/azure/container-apps/gpu-serverless-overview)。

此方案使用长期存活、自动伸缩的 **Container App**，不是预先假定可用的 Container Apps GPU Job。worker 与 TTS 在同一个副本中启停，避免每个合成接口都改成跨服务的异步任务协议。

## 4. 端到端执行流程

1. 用户登录；浏览器直传 Blob，或 CPU 导入器从 YouTube 下载后上传 Blob。此时不启动 GPU。
2. CPU 媒体检查确认格式、时长、大小；冻结输入对象，生成不可变任务规格和摘要。
3. Studio 通过新的 `PrivateEngineBackend` 提交至私有 engine-control；输入从 Blob 流式下载至 /data 的临时文件，校验后原子发布。纯音频包装仍在 CPU 执行。
4. 输入准备完成后才把引擎任务设为可执行 queued。幂等键继续映射到确定性 UUID；同键不同规格拒绝，避免重复扣费或生成两个任务。
5. KEDA 查询持久任务，发现可执行任务后从 0 拉起 GPU app；CPU beat 保持现有 Celery 派发/恢复机制。
6. worker 取得现有 PostgreSQL 全局锁，登记心跳并等待本副本 GPU readiness，再执行 pipeline。等待 GPU 时展示“启动处理中”，不标成服务配置永久错误。
7. TTS/Demucs 通过 localhost 通信，共享 /data，沿用现有路径键与 GPU 互斥逻辑；长时间处理不经过 ACA 对外 HTTP ingress。
8. 引擎完成后保存输出清单，CPU 发布器执行兼容性导出、音视频完整解码检查、时长检查和字幕上传；原有 MP4/MP3 格式及失败警告继续保留。
9. 最终输出以每次 attempt 唯一的 Blob 对象键上传，全部校验完成后才在数据库中发布结果引用并标记成功；不能用“某个对象已存在”代替结果完整性验证。
10. 引擎无待执行或执行中任务后，GPU app 经过冷却缩容到零。用户读取历史、播放和下载只依赖 CPU 与 Blob，不唤醒 GPU。

最终导出在 CPU 进行，因此引擎的所有中间结果一旦可靠写入 Azure Files，GPU 可先释放；Studio 的 publishing 状态继续由 CPU 跟踪。不要让 CPU 发布步骤的状态误保持 GPU 常驻。

## 5. 自动伸缩、重启和费用上限

### 5.1 伸缩设计

- 设置单 active revision、minReplicas=0、maxReplicas=1；worker concurrency=1，保留数据库全局执行锁。
- 优先使用 KEDA PostgreSQL scaler 查询一个受限视图，例如 `gpu_runnable_work`。这是拟新增的视图名称。
- 视图计入：输入已就绪的 queued；provisioning/running；执行尚未停止的 cancel_requested；到达重试时刻且次数未超限的恢复任务。
- 明确排除：上传中、导入中、永久缺配置、超过启动/执行截止时间、失败、完成、已取消、仅剩 CPU 发布的任务。
- **running 必须一直计入**，不能因为 Celery 已取走消息、Redis 队列为空，就把长任务缩容。取消请求直到执行者确认停止前也计入。
- scaler 使用只读数据库账号，只能读所需视图；执行端用独立最小权限账号。私有网络下验证数据库可达性与 TLS。
- 参考官方默认先用 30 秒轮询、300 秒冷却。它们是初始配置基线，不是“任务结束立即停费”的承诺；验证目标区域/API 实际行为后再调小。
- 日常网页健康探测、结果查询不探测 GPU HTTP 接口；GPU 的 liveness 与模型 readiness 分离，慢加载不得造成无限重启。

ACA 官方提供自定义 KEDA 伸缩，KEDA 提供 PostgreSQL 查询触发器；其在本项目目标环境的组合仍需实测，包括 scaler 的版本、凭据映射和网络行为。[ACA scaling](https://learn.microsoft.com/en-us/azure/container-apps/scale-app)、[PostgreSQL scaler](https://keda.sh/docs/2.18/scalers/postgresql/)。

### 5.2 任务安全

- CPU 侧独立 watchdog 处理 GPU 无容量、启动失败、心跳丢失、执行超时、取消不响应；不能依赖出故障的 GPU 自行结束。
- 现有心跳、全局锁和幂等机制继续使用。领取任务时分配 attempt/执行令牌，写结果时验证令牌仍有效；失去数据库连接、心跳续约失败或锁所有权不确定时，执行端停止计算并禁止发布。
- maxReplicas=1 不是绝对单实例锁：平台维护、修订切换仍可能短时出现额外副本。必须依靠数据库锁防止重复执行，并为额外启动费用留余量。
- Demucs 子进程启动、取消和退出受 worker 生命周期管理。进程重启后根据持久 manifest/输出校验恢复；没有可信完成标记的阶段使用独立 attempt 目录重跑。
- 合成缓存按输入指纹、文本、参考音频、模型和配置版本匹配；旧 attempt 不得覆盖新输出，失败的 .partial 文件不得被当作完成结果。
- SIGTERM 时停止领取新任务，尽力结束子进程并保存恢复点；不能把平台优雅退出窗口当作保证长视频一定跑完。
- 自动重试仅针对临时失败且受次数/时间/预算约束；初始上限沿用配置中的 3 次语义并核实总尝试数。确定性坏输入、配置缺失和持续 OOM 不循环重试。

### 5.3 成本规则

总成本 = CPU VM 与磁盘/备份固定成本 + GPU app 实际副本运行成本 + Blob/Files/镜像/日志成本 + 网络出站 + 已有模型 API 成本。

GPU app 成本按实际 SKU 的 GPU、CPU、内存计费项合计；含启动、加载、API 等待、计算和缩容冷却，不能只统计 CUDA 活跃秒。副本为零后计算资源不计费，持久存储继续计费。[计费说明](https://learn.microsoft.com/en-us/azure/container-apps/billing)。

- 改造前核对 Japan East T4 的完整费率，替换 `CostPolicy` 的 local-dev 单价、runtime_ratio 和 provisioning_allowance；现有默认值不是 Azure 报价。
- 接单前校验预算；已有 queued 任务的启动资格也必须检查 `gpu_starts_enabled` 和预算，不能只关闭新接单却让 scaler 启动旧积压任务。
- 记录每个任务等待、启动、处理、发布时长及副本存活区间；共用冷却成本按明确口径分摊，对账以 Azure 实际账单为准。
- 成本告警不是硬断电；同时实施启动上限、任务截止时间和熔断。超预算时先禁新启动，运行中任务按预定取消/收尾策略处理。
- 不先启用额外常驻 GPU、ACR Premium 或双区域部署。是否购买镜像流式拉取等功能以冷启动收益和新增费用决定。

## 6. 存储与网页适配

### Blob：输入和最终结果

- 沿用 ObjectStorage 抽象，完成 AzureBlobStorage；使用托管身份及 user delegation SAS，不把长期存储密钥交给浏览器。[SAS 官方实现](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-user-delegation-sas-create-python)。
- 修复当前直接向签名函数传入 `credential` 的用法；从服务端获取并缓存有时限的 delegation key，再签发对象级 SAS。
- 云端 `local_path()` 返回 None，所有输入检查、冻结、导入、导出路径都必须支持下载到受控临时目录；不只替换 storage 初始化。
- 下载使用分块流，避免当前 readall 把 2 GiB 视频一次读入内存；上传设置正确 Content-Type、Content-Disposition 和校验信息。
- 现有前端已有 Blob PUT 的 `x-ms-blob-type` 分支。首期必须补齐大文件分块上传/重试：建议大于 100 MiB 使用 Azure Block Blob，16 MiB 块、最多 3 并发；最后由服务端校验/提交块列表。
- SAS 建议继续 15 分钟有效并可续签；只允许目标对象操作。SAS 不限制真实文件总大小，服务端必须校验提交后的实际大小与 2 GiB 上限。
- 提交后由服务端冻结到浏览器没有写权限的新对象键，记录 ETag/版本和长度，避免尚未过期的上传 SAS 修改正在处理的输入。
- 配置实际前端域名的 CORS 与所需方法/请求头；私有播放使用短期 GET SAS，验证 Range/206、拖动、过期续签及 WebVTT。
- SAS 不写入持久 JobSpec，不记录在日志；存储对象键随任务固定，长任务不依赖早先签发的 URL。
- 保留当前输出 7 天、等待输入 72 小时的默认语义；清理必须检查执行/重试中的引用。用户可见过期与后台生命周期删除一致，模型缓存不套用用户媒体规则。

### Azure Files：工作目录和模型

- /data 与 /models 分开；CPU engine-control、发布器和两个 GPU app 容器使用一致的 /data 路径。
- 首期优先验证 Azure Files SMB 挂载和权限；只有性能实测需要时再评估其他协议及其成本。不要把数据库数据文件放共享文件系统上。
- 确认非 root 用户能读写挂载点、缓存、临时文件及原子 rename；模型下载一次、固定版本并校验，重启不重复完整下载。
- 当前 GPU 镜像在 root 下运行，需在镜像构建时装好依赖，支持平台非 root 运行；修正 /models、HOME、HF_HOME、工作目录和子进程环境。[非 root 要求](https://learn.microsoft.com/en-us/azure/container-apps/functions-gpu-container-apps)。
- ACA 临时磁盘不能默认承载长视频解码后的音频和模型；按原始时长、采样率、声道及并行工作文件测算容量，磁盘不足时提前拒绝/等待。
- Azure Files 支持跨容器持久共享；Blob 不直接作为 ACA 的该类文件挂载。[挂载文档](https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts)。

## 7. 代码与部署改动清单

以下新名称均为拟新增文件或接口，不表示已经实现。

| 范围 | 修改内容 | 优先级 |
| --- | --- | --- |
| config.py、bootstrap.py | 新增 `azure-jp-t4` profile，显式组合 Blob、PrivateEngineBackend、CPU 检查；保留 local-full 回归路径，启动不要求 GPU 常驻 | P1 |
| adapters/private_engine.py（新） | 实现现有 JobBackend submit/get_status/cancel；确定性 UUID、规格摘要、超时及临时错误映射 | P1 |
| 私有 engine-control 接口（新） | 从 engine_bridge 抽取提交/状态/取消能力；输入物化、鉴权、参数/路径约束、输出清单；不接受任意命令或任意远程下载 URL | P1 |
| 结果发布模块（新，从 docker_engine 抽取） | 用 Files 读取替换 docker cp，复用导出和 validate_output；幂等发布 Blob、字幕、警告与结果引用 | P1 |
| azure_blob.py、inspection_worker.py、local_jobs.py | 正确签名、流式 I/O；CPU inspection 选择云存储但保留本地子进程队列 | P1 |
| web/app.js、上传完成 API | 分块上传、续签、冻结输入、错误恢复；GPU 归零不影响播放器 | P2 |
| 引擎 tasks.py、数据库迁移、伸缩视图 | 冷启动等待与心跳顺序、可执行状态、attempt 保护、超时和取消恢复 | P2 |
| services/tts/ 与云端镜像构建 | 非 root、体积优化、探针、Demucs 重启恢复、持久模型缓存 | P0/P2 |
| deploy/azure-jp/（新） | CPU Compose、ACA/Bicep 参数、VNet/权限、挂载、scaler、备份、监测、只读配额检查与受控验证脚本 | P0/P3 |
| 原计费/reconcile 流程 | 保留一次扣款和一次退款，区分启动等待与永久失败，按真实计算口径校准预算 | P2/P3 |

公网仅暴露 HTTPS 网页入口。CPU 上 engine-control 只允许受信服务访问并验证服务身份；PostgreSQL/Redis 只允许指定私网来源，应用、数据库与伸缩器分权。GPU app 不设公网入口。镜像可保持私有并配置拉取身份，不必公开 GHCR 包。

## 8. 实施顺序与阶段退出条件

### 阶段 0：可行性验证，先于主体改造

1. 记录实际源码、补丁和镜像基线；确认两个运行环境的依赖及 GPU 镜像体积。
2. 查看/准备 Japan East ACA 环境的 Consumption T4 配额；不足则申请，配额批准不等于容量预留。旧 East US Azure ML 配额不适用。[配额说明](https://learn.microsoft.com/en-us/azure/container-apps/quotas)。
3. 查目标配置完整费率；明确验证资源的预算、结束条件和清理列表，再运行收费测试。
4. 验证一个 T4 副本中的两容器启动、GPU 只给主容器、localhost 通信、共享文件权限、镜像拉取限制和 CUDA/FP16。
5. 用一条持久测试任务验证 PostgreSQL scaler 的 0→1→0，正在执行时队列可为空但副本不能因正常缩容被停掉。

退出条件：配额/容量、镜像、双容器、共享存储和伸缩全部有实际验证记录。任一不通过，先调整该项，不开始完整数据迁移。

### 阶段 1：本地可回归的接口与存储适配

实现 PrivateEngineBackend、私有 engine-control、CPU 发布器和 Blob 适配；先在本地 Compose 中验证接口语义，维持当前 GPU 部署作为基线。补齐上传和媒体检查，不引入模型替换。

退出条件：同一真实短视频与纯音频在新接口下完成，输出、警告、计费与原路径一致；重复提交无重复任务。

### 阶段 2：缩容与异常恢复

实现冷启动状态、伸缩视图、watchdog、attempt 校验、Demucs 恢复和非 root 镜像；验证本地重启/取消后，再验证日本 T4。GPU 从零启动可接受排队，前端明确显示阶段，不承诺固定秒数冷启动。

退出条件：长任务期间不因队列取空被缩容；失败不产生无限重启/扣费，任务重试不重复发布或退款。

### 阶段 3：小规模云端全流程

部署 CPU 服务、私有网络、Blob/Files；导入测试账号及沙箱数据，不直接迁移正式余额。分别用文件上传与 YouTube 输入跑 MP4、MP3、字幕、播放和下载。

退出条件：下表验收通过；记录 T4 与本机 RTX 4060 Laptop 的同视频实测、冷/热启动、阶段耗时、内存和完整费用；确认大陆及北美真实网络体验。

### 阶段 4：迁移与切换

先备份并试恢复；暂停新任务，排空本地执行任务，对数据库/媒体做一致性迁移与清单校验；检查域名、登录授权域、支付回调和存储签名。切换时仅允许一个写入/接单入口。

上线后做合成探测和错误监测。保留旧镜像与备份；如需回退，先停止新接单并同步切换期间产生的订单、余额和任务记录，再恢复旧入口。不能直接恢复旧数据库覆盖新支付数据，也不能让本地与云端同时消费同一生产队列。

## 9. 验收清单

| 场景 | 必须满足 |
| --- | --- |
| GPU=0 持续一段空闲期 | 网页、登录、上传、历史、已完成结果播放下载正常；无业务 GPU 探测反复拉起副本 |
| 冷启动接单 | 已准备任务唤醒 T4，前端可见等待；GPU 未就绪不使网页启动失败 |
| 连续两任务 | 任务串行且不重复；无积压后进入缩容，保留模型与结果 |
| 超过 4 分钟的处理 | 不依赖公网 HTTP 长连接；执行中持久状态始终阻止正常空闲缩容 |
| 重启/取消/失联 | CPU、worker、TTS 分别中断均可恢复或明确失败；旧执行者不可覆盖新结果 |
| 正确性 | 时长对齐、MP4 浏览器兼容、MP3、WebVTT、完整解码与段落警告保持原行为 |
| 幂等及资金 | 重复请求、超时重试、回调重放不会重复扣款、退款或发布 |
| 上传安全 | 大文件中断/续签可恢复；冻结后旧上传 SAS 无法改变任务输入；跨用户对象不可访问 |
| 空间和费用 | 文件/模型不会撑满临时盘；预算阻断对已排队任务也有效；超时任务不持续拉起 GPU |
| 网络 | 大陆、东亚、北美实测登录/上传/拖动/下载；日本出口完成实际 YouTube 导入，记录失败情况 |
| 备份和切换 | 从备份恢复成功；余额/任务/媒体引用一致；切换与回退不会出现双写 |

对业务不变量编写必要回归测试，优先覆盖取消、幂等、失联恢复、播放、预算及余额；不为部署参数逐项写镜像式测试。

## 10. 尚未确认的事项及后续优化

- 当前 Azure 登录/订阅、Japan East 配额、空余容量和报价均未读取；本方案不是上线完成报告。
- 最大风险是 GPU 镜像体积、目标环境双容器资源约束、Files 吞吐及冷启动；风险消除前不能准确承诺工期或单视频成本。
- CPU VM 为低改造成本首期方案，存在单机故障窗口及自行维护责任。未来若改为 CPU ACA 多副本，应先把 Studio SQLite 迁至满足现有事务语义的共享数据库。
- 第一阶段跑通后，再按监测结果决定是否把转写/翻译等待与 CPU 渲染移出 GPU 副本、是否合并多条短任务减少冷启动、是否需要增加并发。不要在此次迁移同时改变算法、并发模型和计费规则。
- 本文件作为新的 Azure 方向实施基线；此前 CLOUD_STORAGE_PLAN.md 中的 R2 + Runpod 方案保留为备选，不与本方案混合配置。
