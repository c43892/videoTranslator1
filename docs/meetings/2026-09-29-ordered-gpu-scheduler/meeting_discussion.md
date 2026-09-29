# Detailed discussion record

## Original framing

Use one consistent queue. Prefer local GPU, use T4 when local is absent or busy, and support a finite ordered list for future GPU services. If all are busy, keep the job queued and check every 10 seconds.

## Premise and essence review

Facts:

- The former scheduler treated a recent local heartbeat as availability even while local was processing a video.
- Local and T4 workers shared one deployment-wide advisory lock, preventing concurrent videos.
- Jobs already have a durable creation time, provider assignment and deadline in PostgreSQL.
- The Azure scaler reads `gpu_runnable_work`; assigned Azure jobs must stay visible until terminal.
- The application already retries capacity-waiting admissions on a 10-second cadence.

Judgments:

- “Available” must mean both online and below configured provider capacity.
- FIFO order is the `cloud_executions.created` order for unassigned queued jobs.
- Assignment must be sticky for the life of an attempt to avoid duplicate side effects.
- A provider should own an independent execution lock; otherwise spillover cannot create parallel capacity.

Assumptions:

- Production has one independently runnable slot for `local` and one for `azure_t4`.
- Additional independent services will use unique provider names and one worker deployment per configured slot.
- Scale-to-zero providers such as T4 are considered online by configuration; outbound providers use heartbeats.

Unknowns retained as operational checks:

- Exact T4 cold-start time varies.
- A future provider may need a custom readiness adapter before being inserted ahead of T4.

Essence: this is a capacity-aware ordered dispatch problem, not merely a fallback toggle. The durable queue owns order; provider workers only claim when their slot is the earliest available slot in the configured sequence.

## Antithesis review

Rejected option: keep one global lock and merely change the local heartbeat predicate. This would wake T4 but still serialize execution, so it would not satisfy busy spillover.

Rejected option: reassign an in-flight attempt when a provider disappears. GPU calls and file publication are not transactionally movable; reassignment could duplicate work or publish stale results. Fail closed and retry as a new attempt remains safer.

Rejected option: infer arbitrary future provider health from “not busy.” An offline service could suppress lower priorities forever. Future non-scale-to-zero providers therefore need recent worker heartbeats; explicitly configured scale-to-zero providers are the exception.

Wrong-codeification risk: treating every registered local agent as an immediately usable whole-video slot would be incorrect because the present CPU orchestration worker is single-slot. Multiple local agents behind the broker remain failover within the `local` slot until matching CPU workers and provider identities are provisioned.

## Decision

Adopt:

1. Ordered configuration: `GPU_PROVIDER_PRIORITY=local,azure_t4`.
2. Capacity configuration: `GPU_PROVIDER_CAPACITIES=local=1,azure_t4=1`.
3. Availability = online/readiness and active assignments below capacity.
4. One stable advisory lock per provider.
5. Oldest unassigned queued execution is selected first.
6. Sticky provider assignment for an attempt.
7. Ten-second scheduler polling when no work can be claimed.
8. Azure scaler exposes unassigned work when every higher-priority provider is unavailable or full.
9. Application admission allows two active jobs globally and per user so the second job can reach T4; remaining work continues through the existing capacity queue and its 10-second checks.

## Responsibility and risk

- Implementation engineer owns code, tests, image publication and immutable deployment references.
- Operations owner must drain active work before changing all workers, deploy control/schema before workers, and preserve prior image/config for rollback.
- Permissions owner confirms no new secret is required and no credential value is logged or committed.
- Cost risk increases because local and T4 may run concurrently. Existing admission budgets and T4 max replica 1 remain in force.
