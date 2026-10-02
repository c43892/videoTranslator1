# 本地部署与模式切换 / Local deployment modes

同一个 Git 仓库支持以下两种独立部署。Windows（Docker Desktop + WSL2）和
Linux（Docker Engine + NVIDIA Container Toolkit）均可使用相同的 Python 命令。
需要 Python 3.10 或更高版本、Docker Compose v2、可在 Docker 内使用的 NVIDIA GPU，
以及足够存放镜像、模型和视频的磁盘空间。RTX 4060 Ti 16 GB 已验证；其他型号的
显存需求取决于实际输入，请先测试。首次启动需要联网下载 IndexTTS 2.5 和 Demucs 模型。

| 模式 | 本机运行 | 外部依赖 | 登录与计费 |
| --- | --- | --- | --- |
| `demo` | Studio 网页、数据库、CPU 工作流、IndexTTS 2.5、Demucs、直接视频下载 | OpenAI Whisper / 翻译 API；DeepSeek 网页聊天（可选） | 免登录，不检查或扣除余额 |
| `provider` | IndexTTS 2.5、Demucs、反向注册代理 | 已部署的 HTTPS 云端 Web App / GPU broker | 沿用云端的生产登录与计费 |

本地 Demo 不向云端注册 GPU，不使用 Azure T4。生产 provider 没有本地网页；
只通过出站 HTTPS 接收云端分配的 GPU 任务，无需开放本机入站端口。
切换只操作**本机的这两个 Compose 项目**，不会部署或修改云端 Web App。

## 在另一台机器部署

```powershell
git clone https://github.com/c43892/videoTranslator1.git
cd videoTranslator1
python deploy/local_service.py init
```

`init` 从已提交的示例建立 `.env.local-gpu` 和 `.env.home-gpu`，并为 Demo 自动生成
数据库密码及内部签名密钥。已有配置不被覆盖。两个真实配置文件均被 Git 忽略；
不要将原机器的 API 密钥、支付凭证或 Firebase 管理员 JSON 提交到仓库。

只需配置所选择的模式；另一模式的配置可以保留空白。

### 选择本地 Demo

编辑 `.env.local-gpu`：

- 设置 `OPENAI_API_KEY`，用于当前版本的 Whisper 转录和 OpenAI 翻译。
- 如果要使用网页聊天，设置 `DEEPSEEK_API_KEY`。这不改变视频翻译的 provider。
- 保留当前版本的模型默认值，或按项目支持的选项配置转录与翻译。
- 不需要 Firebase 或 Stripe 凭证。Demo overlay 固定 `AUTH_MODE=demo`、
  `PAYMENT_MODE=disabled`、本地 GPU 和禁用 T4，并启用本地任务接收。

启动 Docker 后执行：

```powershell
python deploy/local_service.py demo --build
```

命令先加载 GPU 并等待就绪，再启动网页及 CPU 服务。成功后访问
<http://127.0.0.1:8090/>。网页只监听本机回环地址，供本机使用。
“免费”指 Demo 无账户余额检查或扣费；外部 API 的调用仍由 API 账户付费。

### 选择云端 GPU provider

编辑 `.env.home-gpu`：

- `GPU_AGENT_SERVER_URL`：云端 Web App 的 HTTPS 地址。
- `GPU_WORKER_TOKEN`：为该 GPU 主机单独分配的至少 32 字符随机凭证。
  云端 broker 必须已经将这个凭证加入 `GPU_WORKER_TOKENS`；仅在本机生成
  一个新字符串不会获得云端授权。
- `GPU_PROVIDER_ID`：可留空（从凭证派生稳定 ID），或设置唯一 ID。
  一个已注册凭证的 ID/type 不能改变。不同主机要使用不同凭证和 ID。
- `GPU_TASK_VOLUME`、`GPU_MODEL_VOLUME`：本机 Docker 卷名。
  默认与本机 Demo 共用任务/模型卷，切换不需要重新下载模型。
  同名 Docker 卷在不同机器上互不共享。

```powershell
python deploy/local_service.py provider --build
```

脚本自动创建所需外部卷，先等待 GPU 就绪，再启动代理。只有云端成功接受注册、
GPU 已就绪且收到最近 30 秒内的心跳确认，才报告 provider 启动成功。
代理 Docker healthcheck 也使用该状态；“容器 Up”本身不代表云端可用。
状态文件仅含 provider ID、就绪/忙碌标记和心跳时间，不含凭证。

## 切换、停止、更新

镜像已构建且代码未更新时：

```powershell
python deploy/local_service.py demo
python deploy/local_service.py provider
python deploy/local_service.py status
python deploy/local_service.py stop
```

启动一种模式会先停止并移除另一种模式的容器，确认停止成功后才启动 GPU。
命令通过主机用户级文件锁防止两个切换命令并发执行。同一种模式重复启动可复用
其现有容器。数据库、视频结果、任务和模型卷会保留；脚本从不执行 `down -v`。
无需另一模式的 env 文件就能停止它，因此一台新机器只配置一种模式即可。

**请先让正在处理的视频完成，再切换。**切换会停止另一套的 CPU/GPU 工作进程，
不会无缝迁移执行中的任务；被中断的任务需要检查并重试。云端在失去心跳约 30 秒后
将该 provider 标记为离线，其他 provider / T4 的选择由既有云端策略决定。

更新后重新构建：

```powershell
git pull --ff-only
python deploy/local_service.py demo --build
# 或：python deploy/local_service.py provider --build
```

不要同时直接执行两份原始 Compose 的 `up`：切换命令的互斥保护仅适用于此入口。
现有云端生产部署继续使用云端 Compose，不使用本机 Demo overlay。

## 常用参数与检查

```powershell
python deploy/local_service.py demo --docker "C:\path\to\docker.exe"
python deploy/local_service.py provider --provider-env "C:\private\gpu.env"
python deploy/local_service.py demo --timeout 3600
```

默认就绪超时为 30 分钟，供首次模型下载使用。超时或注册被拒绝时会返回非零退出码，
另一套保持停止，避免为了回退而自动启动第二个 GPU 后端。排查后可以重试同一命令。
Docker Desktop/Engine 必须先启动；在 Windows 上 Docker 的 `unless-stopped`
重启策略只有在 Docker 已运行时才有效。睡眠、退出登录和断网会影响云端可用性。

```powershell
docker logs --tail 80 videotranslator-local-gpu-tts-1
docker logs --tail 80 videotranslator-home-gpu-agent-1
docker exec videotranslator-home-gpu-agent-1 python /opt/gpu-agent/runtime_status.py
```

Demo 的 `/api/v1/health/ready` 应返回成功，`/api/v1/chat/config` 应包含
`auth_mode=demo`、`billing_enabled=false` 和 `processing_available=true`。
provider 的云端在线状态可在生产 Web App 管理页面检查。
