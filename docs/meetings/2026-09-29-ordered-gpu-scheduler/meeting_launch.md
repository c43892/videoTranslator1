# Meeting launch: ordered GPU scheduler

- Date: 2026-09-29
- Classification: formal decision meeting
- Trigger: production scheduling structure, public queue behavior, GPU cost and rollback risk all change.
- Decision requested: make one FIFO queue dispatch to the first available provider in a finite order; default `local` then `azure_t4`; wait and recheck every 10 seconds when all providers are busy.
- Participants: adversarial reviewer, technical feasibility architect, implementation engineer, permissions/data-governance owner.
- Required outputs: agreed contract, implementation plan, deployment/rollback plan, tests and acceptance evidence.

The user's instruction is the authority for changing routing behavior and deploying it. It does not authorize exposing credentials, deleting data, interrupting active jobs, or removing cost controls.
