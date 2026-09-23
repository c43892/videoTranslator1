# 云端媒体存储方案：Cloudflare R2 + Runpod Serverless

> 2026-09-23 选型更新：用户已选择 Azure 日本东部 T4 作为改造方向。当前实施基线见 [Azure 日本东部 T4 改造方案](AZURE_JAPAN_T4_MIGRATION_PLAN.md)。本文保留为此前 R2 + Runpod 备选设计，不是当前部署配置。

核查日期：2026-09-23。

状态：技术方案建议，可作为后续实施基线；尚未创建云资源、配置凭据、迁移文件或验证跨地区性能。

## 1. 选型和职责

- 用户原始音视频、输出 MP4/MP3、字幕和结果清单：Cloudflare R2 Standard 私有桶。
- 计算：Runpod Serverless Flex worker；临时磁盘只存计算期间的工作文件。
- 账号、余额、上传会话、任务状态、归属、文件位置和过期时间：应用数据库。R2 不替代事务数据库。
- 模型权重：独立模型缓存，可用 Runpod 网络卷或平台模型缓存；不与用户结果的保存周期绑定。
- 第一阶段私有下载：应用鉴权后签发 R2 S3 GET URL；第一阶段不引入公共桶或额外 CDN 缓存。

核心原因：R2 的 S3 兼容接口适合项目的 ObjectStorage 抽象；Standard 按存储和请求计费，R2 本身不收互联网出站流量费，适合 GPU 跨云读取及用户重复下载。

## 2. 区域安排

设计支持两个逻辑存储位置：北美（创建桶时 location hint 为 wnam/enam，匹配实际计算区域）和亚太（apac）。桶名需在实施时确定。

先验证一个计算区域与匹配的存储位置，再启用第二个区域；不把“日本有 Runpod 机房”视为所选 Serverless GPU 一定在日本可用。

每个上传会话和任务记录 storage_location_id、bucket、object_key；同一个任务的输入和输出默认放在同一位置。接单前选定地区，不能让后续轮询按用户当前 IP 重新选桶。第一阶段不自动跨区复制所有视频。

R2 location hint 是尽力安排，不保证东京等指定城市，也不等于亚太和北美自动双活复制。需要特定城市保证时，应重新评估带明确区域的对象存储。

东亚和北美需要分别实测大文件上传、GPU 拉取、播放拖动及下载。中国大陆用户比例尚未明确；若属于核心市场，需追加大陆实际网络下的登录、上传和播放验证，不能承诺 R2 或现有 Firebase 的体验。

## 3. 上传链路

1. 用户登录后向 Studio 创建上传会话；服务端生成对象键、记录用户归属并检查当前 2 GiB 文件上限。
2. 小文件可使用预签名 PUT；建议超过 100 MiB 使用 S3 multipart，默认 16 MiB/片、最多 3 片并行，失败只重试对应分片。
3. 应用负责创建、查询、续签、完成和取消 multipart；浏览器直接 PUT 分片到 R2，不经过 Studio 或 GPU 转发文件正文。
4. 上传 ID 和已完成分片信息保存在服务端；刷新页面后可以恢复，浏览器可能需要用户重新选择同一文件并核对大小及指纹。
5. 分片签名短期有效（建议 15 分钟，可续签）；限制允许的来源域、方法和请求头，并暴露 ETag 等必要响应头。CORS 不替代鉴权。
6. CompleteMultipartUpload 由服务端执行，校验分片清单、实际大小和对象存在性，再进入 CPU 媒体检查；重复完成请求不能创建重复任务或扣款。
7. 已确认的输入对象必须不可被用户旧上传权限覆盖。大文件采用服务端完成的 multipart；小文件 PUT 应写入 staging 对象，由服务端固化为独立的不可变输入键。

应用声明的大小不足以限制存储层实际写入量，因此还要限制活跃上传会话、签发分片数量，并在完成时核查实际大小。媒体时长、格式和可解码性由 CPU 检查确认后再提交 GPU。

## 4. GPU 输入和结果提交

- 队列保存稳定的存储位置和对象键，不把即将过期的浏览器 URL 当成长期任务输入。
- Worker 真正开始时，通过任务级凭证向应用换取输入 GET 和本次 attempt 输出 PUT/multipart 授权；处理时间较长时允许续签。
- GPU worker 下载到临时目录，运行现有处理流程，并保留 FFmpeg 可解码性、时长和音视频流检查。
- 输出写到独立 attempt 键；服务端验证对象及清单后，以事务将该 attempt 发布为任务结果。
- 只有持久上传和发布成功，任务才能显示完成并允许计算实例退出；GPU 关闭不影响已经保存的文件。
- 同一任务重试必须幂等，失败 attempt 不能覆盖已发布结果，不能重复扣款。

将视频下载、上传、FFmpeg 和外部 API 等待放在 GPU worker 内会延长 GPU 计费时间；首版追求流程完整，实测后再决定是否拆分 CPU 后处理。

## 5. 播放与下载

用户请求结果 → Studio 验证用户归属、任务状态和过期时间 → 生成短期签名 GET URL → 浏览器直接从 R2 播放/下载。

保留 MP4 faststart 和 HTTP Range 播放能力；验证 206、拖动和音频播放。设置正确 Content-Type；下载场景设置 Content-Disposition，避免返回 MIME 不明的对象。

当前签名时效 15 分钟，可沿用，但必须在长视频播放或拖动遇到过期时重新鉴权续签，并恢复播放位置。数据库只存对象键，不保存签名 URL；日志隐藏 URL 签名。

R2 S3 预签名 URL 仅适用于其 S3 API 域名，不能直接换成自定义 CDN 域名。第一阶段不声称拥有全球边缘缓存。后续如需要自定义域名和缓存，另加 Cloudflare Worker 私有鉴权网关，并验证 Range、缓存隔离和额外计费。

## 6. 保存与清理

沿用当前产品配置作为结果保存基线：输出保留 7 天；尚在等待用户输入的任务保留窗口为 72 小时。保存时间与签名 URL 有效期是两回事。

- 结果删除时间以应用 output_expires_at 为准，界面展示到期时间。
- 输入至少保留至任务结束及既有重试窗口结束；具体清理条件必须和任务状态、retry_expires_at 对齐。
- 本地临时文件在 worker 结束时清理；未发布输出在确认没有活跃 attempt 后清理。
- 未完成 multipart 使用独立清理策略；R2 默认在 7 天后中止，实施时可依据产品上传期限调整。
- 桶生命周期作为兜底，不能仅按文件创建时间删除仍在排队或运行的输入。
- 这份文档不会开启删除策略，也不会改变或删除现有本地文件。

## 7. 当前代码与实施范围

已有基础：

- packages/videotranslator/videotranslator/ports.py：ObjectStorage 包含上传/下载 URL、上传、下载、存在性和删除接口。
- application/jobs.py：已有上传归属、续签、完成时大小校验、结果鉴权和过期检查。
- web/app.js：已有云端 upload_url 直传分支；当前本地配置走 Studio 上传入口。
- config.py：当前上限 2 GiB、输出保留 7 天、签名有效期 900 秒。

需要实现：

1. S3/R2 ObjectStorage 适配器（例如 boto3），流式下载和分片上传，新增云端 profile。
2. 独立可选 multipart 接口与会话持久化，前端续传及进度；保留单 PUT 兼容。
3. 存储位置记录及确定性路由，输入固化及 attempt 输出发布。
4. 适配 R2 的 CPU inspection worker；现有 inspection_worker.py 仍直接实例化 LocalObjectStorage，不能只切换主 API 的存储类。
5. Runpod JobBackend 和真实引擎容器入口，替换当前依赖本机 docker exec/cp 的调度。
6. 云端结果播放/下载、签名刷新和生命周期协调。

R2 API 凭据只放服务端 Secret，不能进浏览器、Git 或日志；Worker 优先使用任务范围授权。按用户对象路径命名本身不构成权限控制。

## 8. 成本基线与验收

R2 Standard 公布价格：0.015 美元/GB-month，Class A 4.50 美元/百万次，Class B 0.36 美元/百万次；免费额度包括每月 10 GB-month、100 万次 A 类和 1000 万次 B 类请求。

例如持续存储 100 GB 一个月，未计免费额度时存储费约 1.50 美元；持续存储 1000 GB 约 15 美元。请求费、GPU、应用服务及其他供应商可能收取的网络费用另计；读取视频可能产生多次 Range 请求。

上线前验收：

- 东亚和北美各测一个小音频、约 200 MB 视频及接近 2 GiB 的文件，记录真实传输耗时、失败率和完整成本。
- 断网和刷新后续传、分片 URL 过期续签、超额或重复提交。
- 用户之间不能访问对方任务；已提交输入不能被旧上传权限改写。
- CPU 媒体检查、GPU 排队超过 URL TTL、重试和输出发布幂等。
- worker 退出后结果仍可播放；Range 拖动、长时间播放续签和下载文件名正确。
- 到期清理不删除活跃任务输入；数据库和对象存储状态可协调恢复。

## 官方参考

- R2 价格：https://developers.cloudflare.com/r2/pricing/
- R2 数据位置：https://developers.cloudflare.com/r2/reference/data-location/
- R2 预签名 URL：https://developers.cloudflare.com/r2/api/s3/presigned-urls/
- R2 CORS：https://developers.cloudflare.com/r2/buckets/cors/
- R2 分片上传：https://developers.cloudflare.com/r2/objects/upload-objects/
- R2 生命周期：https://developers.cloudflare.com/r2/buckets/object-lifecycles/
- Runpod 配置：https://docs.runpod.io/serverless/endpoints/endpoint-configurations
- Runpod 网络卷：https://docs.runpod.io/storage/network-volumes
