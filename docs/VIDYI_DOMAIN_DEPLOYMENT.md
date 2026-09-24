# vidyi.cc 域名发布记录

日期：2026-09-23。网站：https://vidyi.cc 。本次仅更新入口和相关配置，继续运行现有 Stripe Sandbox 部署。

## 托管与 DNS

Azure Japan East 的 `videotranslator-cpu` 已通过 Azure API 核实：公网地址 `40.115.182.114`，IP 资源 `videotranslator-cpuPublicIP` 为 Static。服务目录 `/srv/videotranslator`；Caddy 代理至 Docker 私网中的 `web:8000`。

在已登录的外部 Chrome Porkbun 域名管理中，原 DNS 仅有两条默认停放记录，已逐条编辑，无其他 MX/TXT 等记录被更改。注册商、权威名称服务器、域名自动续费设置未更改。

| 原记录 | 当前记录 | TTL |
| --- | --- | --- |
| `ALIAS vidyi.cc → pixie.porkbun.com` | `A vidyi.cc → 40.115.182.114` | 600 秒 |
| `CNAME *.vidyi.cc → pixie.porkbun.com` | `CNAME www.vidyi.cc → vidyi.cc` | 600 秒 |

Cloudflare 1.1.1.1 和 Google 8.8.8.8 查询已返回新记录。未添加未经核实的 IPv6 地址。

## HTTPS 与应用配置

- Caddy 接受 `vidyi.cc`，并保留 `vtranslator-jpe-43892.japaneast.cloudapp.azure.com`；`www.vidyi.cc` 保留路径及查询字符串后 301 跳转 `https://vidyi.cc`。HTTP 自动 308 跳转 HTTPS。
- 主域名和 www 均取得 Let's Encrypt 有效证书，当前到期时间为 **2026-12-22 21:13:53 UTC**。标准 TLS 证书链和主机名校验通过。
- Caddy 自动证书维护已启用，日志已记录续期窗口。证书及 ACME 状态持久化在 `/srv/videotranslator/caddy-data`；未来续期仍需服务运行、DNS 正确和 80/443 可达。尚未经历一次实际续期周期。
- 私有 `.env` 中 `PUBLIC_HOST=vidyi.cc`，`studio.env` 中 `PUBLIC_APP_URL=https://vidyi.cc`。配置生成和验收脚本的默认域名同步更新，避免重部署覆盖。
- Firebase Authorized domains 添加 `vidyi.cc`，原有域名保留。Google OAuth handler 继续使用现有 Firebase authDomain；www 会在加载应用前跳到主域名，无需单独登录配置。
- 应用使用 Firebase token / Bearer 鉴权，无需迁移服务器会话 Cookie 域。新域名使用自己的浏览器登录状态，旧域名登录状态不会自动跨域共享。
- Azure Blob CORS 增加准确来源 `https://vidyi.cc`，保留原 Azure 来源；允许 PUT/GET/HEAD/OPTIONS 和 `content-type`、`x-ms-*`，Blob 容器继续保持私有。
- 现有 Stripe **测试** webhook 同一 endpoint 更新为 `https://vidyi.cc/api/v1/webhooks/stripe`，签名 secret 未轮换。正式密钥及正式支付未切换。

## 实测结果

- HTTPS 首页 200，HTTP 308；www HTTPS 301；原 Azure HTTPS 地址 200。
- `/api/v1/health/ready`、`/api/v1/chat/config`、已鉴权 `/api/v1/me`、已完成任务查询均 200。
- 匿名身份接口返回 401，sandbox 请求携带 live 模式返回 409。
- 既有云端翻译结果通过私有 SAS 返回 206 Range 视频内容，CORS 返回新域名；浏览器上传预检 OPTIONS 返回 200 和准确来源。
- 新建一次未付款的 Stripe Sandbox Checkout，直接读取 Stripe 确认 success_url/cancel_url 均为新域名，随后主动 expire 该测试会话；没有付款或新增积分。
- Chrome 已打开主页并验证登录弹窗可到达 Google 账户选择页，OAuth context 指向新域名。浏览器自动化在账户选择弹窗停滞，完整 Google 登录返回尚未确认；独立测试用户 Firebase token 的后端验证已通过。
- GPU 修订 `videotranslator-gpu--pipeline-1790200248` 为 `ScaledToZero`，副本 0。此次域名验收未启动 GPU 任务。

## 保留限制与费用

本次未新增付费资源，DNS 和 Let's Encrypt 证书没有额外购买费用。现有 Azure 资源仍按原方式计费。

既有 US$20 总部署验证预算仍有效。CPU 自动关机计划继续为 **每日 05:16 UTC**；下一次为 **2026-09-24 05:16 UTC（北京时间 13:16 / 多伦多 01:16）**，关机后网站会下线。长期在线需要单独确认新的持续费用预算；域名绑定没有取消该计划。仅 CPU 已查询按量参考价 US$0.124/小时，约 US$89.28/30 天，另有磁盘、IP、ACR、存储、流量等费用。关机不消除存储等费用，也不是账户硬限额。

网站仍为 Stripe Sandbox，`GPU_STARTS_ENABLED=false`、`CLOUD_ACCEPT_JOBS=false`，尚未对公众开放付费翻译任务。YouTube 匿名下载受 Azure 出口反爬限制，未因域名绑定而解决；中国大陆和东亚真实用户网络访问仍需实测。

## 回退与维护

VM 的原 `Caddyfile` 和私有环境文件、本地私有环境文件已保留 `.before-domain` 备份；本地私有目录还保存原 Stripe webhook 配置，均不提交版本库。回退时可恢复文件并只重建 web/proxy，必要时将同一 Stripe 测试 endpoint URL 改回 Azure 地址；原域名和 Firebase / CORS 授权仍保留可用。不要回退用户账本、媒体文件或删除 Caddy 证书数据。

DNS 如需彻底回退，可按上表恢复原停放记录；这会使新域名不再提供网站。正常后续发布应保留新 DNS 和证书持久目录。
