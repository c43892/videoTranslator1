# Meeting summary

The accepted design is a durable FIFO with ordered, capacity-aware providers. Production order is `local` then `azure_t4`; each currently has capacity one. A free local slot claims first. If local is offline or busy, T4 may claim the next job. When both are busy, jobs remain queued and are reconsidered every 10 seconds.

Provider assignments are sticky per attempt and provider locks are independent. This enables one local and one T4 execution concurrently without allowing duplicate ownership. The Azure scaler uses the same availability rule. Future services are added as named providers with explicit order, capacity and readiness semantics.

Deployment must not interrupt an active attempt. Rollback restores the previous immutable images and configuration after draining work.
