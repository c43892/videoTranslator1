# Action items

| Item | Owner | Done condition | Rollback/guard |
|---|---|---|---|
| Implement ordered availability | Implementation | Target tests prove local-first and busy spillover | Revert scheduler commit |
| Separate provider locks | Implementation | Lock IDs differ and local/T4 can own distinct jobs | Restore prior image only after drain |
| Update scaler predicate | Implementation | View exposes queued work when local is busy/offline | Recreate prior view from prior control image |
| Align admission capacity | Implementation | Two active jobs can be admitted; later jobs remain capacity queued | Restore prior web image/config |
| Publish immutable images | Operations | Digests recorded; no mutable tag used for deployment | Retain prior digests |
| Production rollout | Operations | Health checks pass and no active job was interrupted | Drain and restore all prior digests/config |
| Acceptance | Operations | Observe local assignment, then T4 assignment while local busy, and 10-second queued retry when both busy | Disable admissions on anomaly |
