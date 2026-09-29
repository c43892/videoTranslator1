# Executor-specific conclusions

## Implementation engineer

- Add shared parsing for provider order, capacities, always-available providers, stable provider locks and poll interval.
- Change selection from online-only fallback to online-and-free ordered selection.
- Keep assigned work recoverable only by its assigned provider.
- Update the scaler view to expose work when all providers ahead of Azure are offline or full.
- Add regression tests for local priority, local-busy T4 spillover, all-busy queueing, distinct locks and the 10-second default.

## Production operator

- Confirm no active execution before rollout.
- Back up Compose and immutable image references.
- Deploy schema/control first, then the local worker and Azure worker from the same engine digest.
- Set order and capacities identically on all components.
- Verify local idle routing, local-busy scaler visibility, both-busy queueing and eventual assignment.
- Roll back all components together; never mix global-lock and provider-lock workers while admitting jobs.

## Security and data owner

- No new credential material is introduced.
- Provider names, capacity and poll settings are non-secret configuration.
- Preserve the existing separation between GPU-agent tokens, broker internal token, database credentials and cloud identities.
