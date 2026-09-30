# VideoTranslator 家用 Windows 下载代理

这是一台主动连接服务器的专用下载代理。电脑无需公网 IP、端口映射或 Docker。
它只下载服务器分配的公开单视频链接，通过 HTTPS 回传，然后由 Azure 继续检查、报价和翻译。
家庭电脑不执行 GPU 译制，也不需要登录你的 Google 账号。

## 安装到任意 Windows 10/11 电脑

1. 将整个文件夹解压到固定目录，例如 `D:\VideoTranslatorDownloader`。
2. 双击 `Install.cmd`。安装程序会检查并通过 Windows 的 winget 安装缺少的
   Python 3.12、Node.js LTS、FFmpeg，再创建本目录专用的 Python 环境。
   安装第三方组件时 Windows 可能要求管理员确认。没有 winget 时先安装 Microsoft App Installer，
   或手动安装上述三个依赖并加入 PATH。Python 依赖版本固定在 `requirements.txt`。
3. 输入服务器地址（默认 `https://vidyi.cc`）以及管理员提供的专用 worker token。
   不要使用 Firebase、Stripe、Azure 密钥代替 worker token。
4. 双击 `Start.cmd`。日志位于 `worker.log`，会自动轮转；临时视频成功或失败后自动清除。
5. 如需登录 Windows 后自动启动，双击 `Enable-Autostart.cmd`，或者在 PowerShell 执行：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Enable-Autostart.ps1
   ```

   这是当前用户的登录启动任务，不是在无人登录时运行的系统服务。电脑注销会影响任务。
   请在 Windows 电源设置中关闭自动休眠；屏幕可以关闭。安装程序不会替你修改电源设置。

`Stop.cmd` 会通知程序退出并取消正在传输的任务。`Start.cmd` 会清除停止标记。
若已设置自动启动，可在任务计划程序中停用或删除 `VideoTranslator Home Downloader`。
不要在电脑 A 下载时把相同配置复制到电脑 B 同时运行；每台电脑应使用不同 token。
替换电脑时先停止旧电脑，再迁移配置，或由管理员撤销旧 token 并生成新 token。

## 服务端配置

在 VideoTranslator 的运行环境（Azure 为 `/srv/videotranslator/studio.env`）中配置：

```dotenv
YOUTUBE_DOWNLOAD_MODE=home-worker
YOUTUBE_WORKER_TOKENS=<专用随机密钥，至少32字符>
```

建议用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 在安全终端生成。
多台电脑的密钥用逗号分隔；密钥不得提交 Git、写入浏览器代码或公开日志。
本包不包含真实密钥。服务端和代理配置应通过私有渠道交付。

部署包含此功能的新 control 镜像，更新 `CONTROL_IMAGE` 并重建 web 容器。
不需要修改 GPU、翻译引擎、数据库连接、域名解析或开放新的公网端口。
反向代理必须允许最长 30 分钟、至多 `MAX_UPLOAD_BYTES` 的请求体（默认 2 GiB）。

## 排队与恢复语义

- 代理每 5 秒发送心跳，每约 2 秒尝试领取空闲任务。
- 每个 token 代表一个代理；代理线程池和服务端均限制为最多两个任务。
  服务器持久化 FIFO 队列，浏览器无需保持请求连接；刷新页面或服务端重启后仍能继续。
- 两个任务都忙但心跳正常：后续任务一直排队，不消耗离线重试次数。
- 没有可用代理：首次检查后每 10 秒重试一次，重试第 10 次仍不可用则导入失败
  （正常调度下约 100 秒）。后台调度延迟可能使实际时间稍长，不会跳过或加速重试。
- 心跳超过 30 秒算离线；正在执行的任务有 45 秒可续租租约。
  断线后租约过期任务重新排队，之后执行同样的十次可用性重试。
- 一个任务总占用不能超过 60 分钟，下载最长 30 分钟，回传最长 30 分钟。
  旧租约的回传会被拒绝。失败后网页显示“重新下载”，重新开始一轮重试。
- 下载限制：公开单视频 URL、最长 4 小时、最高 1080p、最终文件最大 2 GiB。
  代理只调用 `yt-dlp` 的专用站点提取器，不对任意网页启用通用提取器。
  程序会检查临时磁盘空间（默认至少 12 GiB），用后删除临时文件。
- 停电、休眠、家庭上行带宽及来源网站对家庭 IP 的限制会影响可用性。
  不发生第三方代理费用；家庭网络、电费和既有 Azure 服务计费仍适用。

## 更新和故障排查

- `worker.log` 显示连接失败：检查 `config.json` 地址、密钥、互联网连接及服务器是否运行。
- 401：密钥不匹配或服务器尚未启用 home-worker；不要把密钥贴到公开聊天或工单。
- 连续下载失败：先在该电脑检查来源网站能否访问，以及 Node 和 FFmpeg 路径。
  本版本不导出浏览器 cookies，也不会自动使用你的 Google 账号。
- yt-dlp 出现上游兼容问题时，需要更新并验证 `requirements.txt` 的版本后重新运行安装。
- 升级前先停止程序；解压新包时保留 `config.json`，重跑安装后启动。
  不要覆盖正在运行的目录或删除其他目录。
- 回滚：服务端设置 `YOUTUBE_DOWNLOAD_MODE=direct` 并重启 web，恢复原来的云端直连下载。
  家庭代理可随时停用；停用不影响已导入视频的翻译。
