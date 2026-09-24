# Azure YouTube 下载排查

日期：2026-09-23；目标视频 `https://www.youtube.com/watch?v=DlldBRDJXE4`。

## 结论

Azure 出口直接下载仍受 YouTube 拦截。现已按用户选择部署**家用 Windows 反向下载代理**，
经 vidyi.cc 公网验证三个原问题视频均完整回传、检查和进入报价流程。
该路径要求家庭代理在线，详见 [部署和验收记录](HOME_DOWNLOAD_WORKER.md)。
以下为此前直连失败的历史排查：相同视频使用本地 Studio 下载器完整下载成功，H.264 + AAC，
71.866 秒，18,156,952 字节；Azure CPU 容器返回 `Sign in to confirm you're not a bot`。

## 已实测路径

| 路径 | 结果 |
| --- | --- |
| 已部署 yt-dlp 2026.08.19 + Node 22 | 需要登录/机器人验证 |
| `android_vr` | 无法提取，需要登录 |
| `web_embedded` | 无法提取，需要登录 |
| `web_safari` | 无法提取，需要登录 |
| yt-dlp nightly 2026.09.16.232951 + `mweb` + bgutil PO provider 2.0.0 | 机器人验证失败，无视频文件 |
| 显式匿名 Visitor Data + 会话 PO Token + 视频 PO Token，跳过网页配置请求 | Provider 就绪、两次令牌生成均 HTTP 200；视频仍要求机器人验证 |

测试通过 Azure VM Run Command 执行。新依赖只安装在一次性容器的 tmpfs 中；provider 仅接入 Docker 私网、没有发布端口，没有挂载应用凭据。使用内存和 CPU 限额，并在命令结束时删除临时容器。没有更新现有网站镜像或依赖，没有使用个人 Google Cookie，没有购买代理，也没有启动 GPU 任务。

## 历史候选方案（已由家庭代理方案取代）

需取得项目可用的代理/下载服务，或另行授权的专用账号会话，先从 Azure 验证目标视频**完整下载**，再改动下载模块。不能把仅生成令牌或成功读取标题当成修复成功，也不能把个人电脑代下载当成云端持续可用方案。

已有代理可以优先试验。无已有服务时，采购/注册之前需要确认用户授权。2026-09-23 查阅的 IPRoyal 官方按量价为 1 GB / US$7.35（非订阅，结算税费和最终金额以订单为准）；这是可考虑的有限测试流量，**尚未购买、尚未验证 YouTube 可用性**。视频代理流量费用可能明显影响现有每分钟售价，试验成功不代表可直接采用其零售流量价格长期运营。

## 参考

- [yt-dlp 官方 PO Token 指南](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)
- [bgutil provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
- [IPRoyal 按量价格](https://iproyal.com/pricing/residential-proxies/)

本地视频保存在忽略目录 `vt-data/youtube-diagnostics/DlldBRDJXE4.mp4`，原诊断数据同目录；本轮一次性验证脚本保存在 `vt-data/azure-tmp/try-youtube-*.sh`。
