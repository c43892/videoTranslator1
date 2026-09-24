# Azure 日本 T4 部署验收记录

日期：2026-09-23。环境为 **Stripe Sandbox 验证环境**，没有导入本地正式账本、用户视频或正式支付密钥。

## 实际部署

- 入口：`https://vidyi.cc`；`www.vidyi.cc` 自动跳转主域名。原 Azure 地址继续提供兼容访问。域名配置和验收见 [域名发布记录](VIDYI_DOMAIN_DEPLOYMENT.md)。
- CPU：`videotranslator-cpu`，Ubuntu 24.04，Standard_D2as_v4，2 vCPU / 8 GiB，64 GiB Standard SSD；公网 `40.115.182.114`，内网 `10.0.2.4`。
- Docker Compose：Caddy HTTPS、Studio、私有 engine-control、PostgreSQL 16。SQLite 和 PostgreSQL 在 CPU 磁盘持久化。
- GPU：`videotranslator-gpu`，Japan East T4，8 vCPU / 56 GiB，TTS + worker 双容器。min=0、max=1，无公网入口；PostgreSQL 持久任务视图驱动缩放。
- ACR：`vtranslatorjpe43892.azurecr.io`，Basic，管理密码关闭；CPU/GPU 使用各自托管身份拉取镜像。
- 存储：私有 Blob `uploads` / `results`；Azure Files `data` / `models`。CPU 使用 Blob 托管身份和用户委派 SAS。
- 网络：CPU 的 80/443 公开；SSH 限定部署者 IP；5432 只允许 ACA 子网，数据库网络连接强制 TLS；engine-control 仅在 Docker 私网。
- Firebase 已加入准确的 HTTPS 网站域名。Stripe 已创建独立 sandbox webhook。

## 固定镜像

| 服务 | ACR digest |
| --- | --- |
| Studio | `videotranslator/control@sha256:c1671fc9086a7880cc41fb616357ff31cfcdd71169e8a59041eb77a845d6aeb4` |
| Engine | `videotranslator/engine@sha256:3b6bb4134f94fd9cf974c2be207426a72adefc5ecdb8cca7e2a4156fd7dbbc74` |
| IndexTTS2 / Demucs | `videotranslator/tts@sha256:4929b9c14f15bf0078cb31dc4d5796d909ddd277d801421ae7d0c7073221b67b` |

镜像名需加上述 ACR 主机名前缀。依赖、上游提交和模型 revision 固定在 `deploy/azure-jp/`；所有凭据保存在被 Git 忽略的 `secrets/azure-jp/`。

## 已验证

- Studio 单元与回归测试 **184 项通过**；engine **101 项通过**；浏览器分块上传重试及 SAS 续期测试通过。
- Azure T4 上非 root IndexTTS2 完整 CUDA 模型就绪；单独 B0 首次启动约 629 秒。
- 真实公网 HTTPS 首页及健康检查；Firebase 签名 token 验证、匿名拒绝、sandbox/live 请求模式隔离。
- Stripe Sandbox 真实 Checkout 使用测试卡完成；真实 webhook 自动入账，重复请求和对账没有重复充值。服务重建后账本和余额仍保留。
- 真实 Blob 分块上传、续期、提交、CORS；上传 SAS 不允许读取，Blob 不允许匿名访问。
- 上传后的 CPU 检测成功，且不依赖 GPU。
- PostgreSQL scaler 在有任务时将 GPU 从 0 启动到 1；GPU 控制 API 需要独立 bearer，公网不能访问 5432/8080。
- PostgreSQL dump 已恢复到独立临时数据库，恢复出 3 条测试执行记录；两个 SQLite 在线快照重新打开并通过完整性检查。三份备份上传私有 `validation-backups` Blob 后下载校验 SHA-256 一致。
- GPU 实际完成 Demucs、Scribe 转写和 DeepSeek 翻译。失败任务自动返还测试余额。
- 完整任务 `job_eaa5bd5cfa3f1bdc` 成功：18 秒英文素材译成中文，4/4 片段均为 `translated`，无警告。成品 H.264/AAC、时长 18.006 秒，内嵌 mov_text 和独立 VTT；完整解码检查通过。
- 成品通过只读 SAS 下载成功，匿名访问被 `PublicAccessNotPermitted` 拒绝；Range 请求返回 206，可用于播放器拖动。10 秒 SAS 到期后实际返回 403。
- 任务入队至发布约 **8 分 37 秒**：等待 GPU 及首次模型就绪约 5 分 42 秒，流水线约 2 分 55 秒（包括 Demucs 后再次加载 TTS）。这是此次 18 秒素材的测量值，不是所有视频的性能保证。
- **0→1→0 验证通过**：22:04 UTC 实际观察到业务修订 `ScaledToZero`、0 副本；这是 PostgreSQL scaler 自动缩容，没有手动停 GPU。此时公网首页、健康检查均正常。

## 验证发现与修复

1. 初版自制测试素材音频与视频差约 0.45 秒，被现有 0.2 秒对齐校验正确拒绝；改用音视频均为 18 秒的素材，没有放宽算法校验。
2. SQLite 将枚举读回字符串，重复启动或重复检测回调访问 `.value` 会返回 500。改为字符串兼容返回，补充真实 SQLite 重启后的幂等和不重复扣款测试；已重新部署并实测重复启动返回正常状态。
3. Demucs 释放显存后，IndexTTS2 从 Files 重新加载可能超过原有 120 秒 tokenizer 请求超时。增加可配置 read timeout，云端设 600 秒、本地仍为 120 秒；保留连接超时及整条任务 1800 秒硬期限。
4. ACA Single revision 的异步发布可能重新激活旧的 min=1 验证修订。改为 Multiple 模式并显式退役旧修订；B0 清理脚本等发布完成并连续确认 0 副本，避免只凭 PATCH 返回判断清理成功。日常仅保留一个业务修订激活。
5. Windows CRLF 的 SMB 凭据文件导致挂载认证失败；实际部署使用 LF 文件。

## 未完成的生产验收

- **YouTube 匿名导入受 Azure 出口反爬限制**：真实日本 CPU 主机请求公开视频返回 “Sign in to confirm you're not a bot”。这不是已解决项；没有自动导出个人浏览器 cookies，也没有采购代理。文件上传路径可独立验证。
- Google 浏览器登录弹窗可到达账户选择页，但自动化接口在该弹窗无响应；Firebase token 后端验证已通过，浏览器完整登录流程尚未确认。
- 东亚 / 中国大陆网络实测、定期备份制度、自动清理 Azure Files 中间产物、持续费用控制和正式流量切换尚未完成；目前只完成了一次测试数据库备份恢复，不代表已建立生产备份制度。
- 结果 API 当前按 7 天有效期限制链接获取；这不等于磁盘和 Files 中间文件已经有自动物理清理。正式开放前需补充保留和清理策略。

## 预算与操作

US$20 是本次全部部署和验证资源的预算。原有 CAD20 Azure Budget 仅告警，且未覆盖托管基础设施资源组；不能当硬封顶。费用接口有延迟，返回空记录不代表免费。

查询得到的按量参考价格：T4 + 8 vCPU + 56 GiB 合计约 US$1.6776/活跃小时，CPU VM US$0.124/小时。模型加载、外部 API 等待和缩容冷却期间，只要 GPU 副本仍存在就可能计费。磁盘、ACR、存储、IP 等费用独立存在。

验证 VM 配置了 05:16 UTC 自动关机。基础 Compose 默认 `GPU_STARTS_ENABLED=false`、`CLOUD_ACCEPT_JOBS=false`；仅受控测试临时叠加 `compose.validation.yml`。当前验证累计准入上限为 3 个任务，每个最长 1800 秒。不要把这个上限直接当生产队列容量。

本次验证结束状态：已恢复基础 Compose，关闭新 GPU 任务准入；GPU 自动归零，CPU 网站保留在线供审阅，并将按已配置计划于 **2026-09-24 05:16 UTC（北京时间 13:16）** 关机。磁盘、ACR 和存储不会随关机自动删除。项目及托管资源组的费用查询目前尚无已入账记录，不能据此报告实际花费为零，也不能保证未来保留存储不会继续产生费用。

证据文件位于本地忽略目录 `vt-data/azure-tmp/`：`gpu-validation-result.json`、`cloud-job-result.json`、`gpu-scale-zero.json`、`cloud-validation-result.mp4`、`cloud-validation-result.vtt`。测试视频总长 18 秒，避免用这次单一短样本代替长视频和多语言验收。

暂停任务准入：在 VM 的 `/srv/videotranslator` 执行 `sudo docker compose -f compose.cpu.yml up -d web engine-control`。任务完成后等 GPU 自然归零，再停机释放 CPU 计算资源。不要删除 PostgreSQL、SQLite、Files 或镜像来代替停止计算。

本地入口仍保留；当前变更没有切换正式支付、覆盖本地库或提交 Git。完整运行说明见 [部署目录](../deploy/azure-jp/README.md)。
# 2026-09-23 家庭下载代理更新

网站已部署并验证 Windows 主动连接的 YouTube 下载代理：最多两个下载，异步排队，离线每 10 秒重试、最多 10 次。
新版 control 镜像及三个真实视频的公网验收见 [HOME_DOWNLOAD_WORKER.md](HOME_DOWNLOAD_WORKER.md)。
当前需用户在选定家庭电脑启动交付包；测试代理已退出。GPU 与 Stripe 测试开关、CPU 自动关机计划保持原设置。

## 2026-09-24 自动排队修复（覆盖上述准入状态）

用户要求正常启动译制后，已修复遗留验证开关造成的“队列已满”误报及等待任务不自动恢复的问题。
当前私有 `.env` 明确启用 `GPU_STARTS_ENABLED=true`、`CLOUD_ACCEPT_JOBS=true`，
并设置 `CLOUD_VALIDATION_MAX_JOBS=0`，解除历史累计验证次数限制。
并发 1、应用费用预算、1800 秒任务期限、GPU 空闲归零和 CPU 自动关机保持。
因此，上文仅重建 Compose 即可暂停的旧操作已不适用：需先把私有 `.env` 两个准入开关设为 false，再重建 web 与 engine-control。
原有等待任务已自动进入 GPU 启动阶段；家庭下载代理也已按用户要求保持运行。
镜像、行为和验证范围见 [TRANSLATION_QUEUE_FIX.md](TRANSLATION_QUEUE_FIX.md)。
