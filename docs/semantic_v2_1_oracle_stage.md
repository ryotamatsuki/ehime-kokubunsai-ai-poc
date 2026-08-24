# Semantic Operations v2.1 — Oracle Stage

This marker commit closes the pre-redesign oracle stage.

- The new 200-case holdout was sealed first at `b8d48553e9d85ac911af7d326ef997bde041f3f0` and must remain unopened.
- `semantic_oracle_ceiling_v2.py` may use only the already-exposed frozen v1 100-case regression set.
- No sparse-frame, state-reducer v2.1, prompt, production Modal, Streamlit, or event-data changes are allowed before this marker.
- The next stage may start only after the oracle result is recorded from CI.
