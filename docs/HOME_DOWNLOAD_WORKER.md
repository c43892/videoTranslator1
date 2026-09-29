# 家用 Windows 反向下载代理

实现入口：`deploy/home-download-worker/README.md`，安装包由该目录的显式文件清单构建。

## 结构

网页 prepare → Conversation + DownloadTask 原子入库 → 家用代理 HTTPS heartbeat/claim →
本地 yt-dlp → HTTPS 流式回传 → 私有 Blob → 原有媒体检查与报价 → 用户确认后的原有译制流程。

排队不持有浏览器请求或 GPU。服务端原有 SQLite 文档存储提供跨线程、跨进程事务锁；
`download_workers` 和 `download_tasks` 同库持久化，不需要额外数据库或消息队列。
代理密钥映射固定 worker ID；不接受客户端自行声明身份或提高并发上限。
每个密钥最多两个任务，代理本地也只有两个线程。排队按提交时间领取。

公开 API 均在 `/api/v1/download-workers` 下，用独立 Bearer token 验证：

- `POST /heartbeat`：注册、报告存活、续租至多两个正在执行的任务。
- `POST /claim`：原子领取一个任务，没有可用任务或两个位置已满时立即返回 null。
- `PUT /tasks/{id}/content`：携带独立租约令牌，流式上传，校验长度和上限。
- `POST /fail`：仅持有当前租约的代理可报告失败，不接收任意错误文本或远程下载地址。

重试、在线与忙碌的区分、45 秒租约、60 分钟任务上限详见安装包 README。
回传对象使用随机独立键；提交状态前再次校验租约，过期上传不能覆盖新任务。
家庭代理只拿到 YouTube URL、大小限制和本任务租约，不持有 Azure、Firebase 或支付密钥。
服务端保留媒体检查、时长报价和计费边界，代理不能启动 GPU 或扣费。

## 验证记录（2026-09-23）

- 全部 Python 单元测试：198 passed（包括新增 14 项代理测试）。
- Windows 原生代理与独立 HTTP 服务完整联调：原问题视频 `DlldBRDJXE4` 下载成功，
  回传后由真实 ffprobe 检查为 71,866 ms、18,156,951 字节，14 秒进入可确认报价状态。
  使用真实 yt-dlp、Node 和 FFmpeg，没有浏览器 cookies，没有启动 GPU。
- 首次验证发现系统临时盘不足代理预留的 12 GiB，代理正确拒绝任务；改用空间足够的 D 盘后通过。
- HTTP 鉴权、冒用身份拒绝、错误长度拒绝、过期回传拒绝、SQLite 重启恢复、并发争抢、
  10 次间隔重试、忙碌排队超过 100 秒、重新连接恢复均有测试覆盖。

## Azure 部署

- 网站 `https://vidyi.cc` 已启用 `YOUTUBE_DOWNLOAD_MODE=home-worker`。
- 新 control 镜像：
  `vtranslatorjpe43892.azurecr.io/videotranslator/control@sha256:8029ba643b64554ef4460dda71df4b1a25ffcb1ceb2b768fd59bb42ca2e0e236`。
- 镜像基于此前验证的不可变运行时，仅更新应用代码，未新增 Python 运行依赖。
- 服务器 `.env.before-home-worker-20260923` 和 `studio.env.before-home-worker-20260923`
  保存更新前配置。回滚时恢复两个文件并执行 `docker compose --env-file .env -f compose.cpu.yml up -d --no-deps web`。
- HTTPS 首页、ready endpoint 和新版排队 UI 均为 200；匿名 worker 请求为 401。
- GPU `videotranslator-gpu--pipeline-1790200248` 保持 0 个副本、ScaledToZero；
  `GPU_STARTS_ENABLED=false`、Stripe sandbox 保持原设置。
- 没有增加付费代理或新云资源；既有 CPU 自动关机和费用控制设置仍有效。
- 通用包：`vt-data/releases/videotranslator-home-worker.zip`，不含密钥。
  已配置域名和私有密钥的交付包：`secrets/home-download-worker/vidyi-home-worker.zip`，只交付到自己的电脑。

## 公网完整验收

`deploy/azure-jp/validate-home-worker.py` 使用专门的合成测试用户，经真实 Firebase 鉴权访问 vidyi.cc，
没有确认翻译或创建付款。

- 代理离线：102.5 秒后第 10 次重试仍不可用，状态转为 `import_failed`，错误为 `youtube_proxy_unavailable`。
- 经公开 prepare API 执行用户重试：重试计数归零，生成新的任务。
- 同时准备三个原问题视频，Windows 代理主动连接公网 HTTPS；观察到两个 `leased` 和一个 `queued`。
- 三条视频全部完整回传 Azure，私有 Blob 持久化并经真实媒体检查，均为 71,866 ms，进入可确认报价状态。
  本次三条任务总耗时 206.0 秒，其中出现一次连接超时，自动重连恢复。
- 合成用户余额前后相同，未启动 GPU；测试代理正常退出，临时下载目录清理完成。
- 结果保存在忽略目录 `vt-data/youtube-diagnostics/home-worker-cloud.json`。

上线代码已启用家庭代理模式。交付后需要用户在选定电脑执行 Install.cmd 和 Start.cmd；
当前验证进程没有留在后台，也没有替用户改变本机休眠或登录启动设置。
没有在线家庭代理时，新的 YouTube 导入将按十次重试规则失败，用户本地文件上传不依赖代理。

## 用户要求保持运行后的本机状态

2026-09-23 23:56 UTC，应用户“保持运行”要求，另行启动常驻代理。
运行目录为 `secrets/home-download-worker/runtime`，由 Windows 任务计划程序
`VideoTranslator Home Downloader` 在当前用户会话后台管理，登录时启动，异常退出后每分钟重启。
已确认任务 Running，vidyi.cc 成功心跳在 7 秒内，两个下载位可用。
代理日志为运行目录下 `worker.log`，最近心跳写入不含密钥的 `worker-status.json`。
密钥文件仅当前用户和 SYSTEM 可访问。未修改系统休眠设置；电脑需保持联网、不休眠。

2026-09-29 Windows 重启后发现旧运行目录和计划任务均不存在。已从仓库中的固定版本重新创建
`secrets/home-download-worker/runtime`，生成新的独立 worker token 并追加到云端允许列表，安装用户级
FFmpeg，使用本机已验证的 Node 运行时，并重新注册 `VideoTranslator Home Downloader` 登录任务。
恢复后任务处于 Running，公网心跳成功，容量为 2、活动任务为 0。临时 token 传输对象及本地中间
注册文件均已删除；长期 token 只保留在受 ACL 保护的 `config.json` 和云端私有 `studio.env` 中。
