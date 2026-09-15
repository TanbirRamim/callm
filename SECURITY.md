# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/TanbirRamim/callm/security/advisories/new)
instead. You will get an acknowledgement within 72 hours and a status update at least weekly
until the issue is resolved.

Useful reports include the callm version, a minimal reproduction and the impact you observed.

## Supported versions

Security fixes are released for the latest minor version.

## Scope and limitations

callm's security features reduce risk; they do not eliminate it.

- **PII redaction** uses pattern matching (plus optional NER). Unusual formats, free-text
  identifiers (addresses, account numbers without checksums) and names without NER are not
  detected. Treat it as defence in depth, not as a compliance guarantee.
- **Prompt-injection detection** is heuristic (optionally ML-assisted). Novel or paraphrased
  attacks can evade it. Keep least-privilege tool design and treat model output as untrusted.
- **Cache contents** are stored locally (SQLite) or in your Redis. Responses are stored after
  PII masking of the *request*; model responses themselves are stored as returned. Protect the
  storage location accordingly or disable caching for sensitive workloads.
- **Telemetry** never stores prompt or completion text.

Bypasses of the redaction or detection logic with realistic inputs are welcome as regular
issues with test cases, unless they expose a way to leak data that callm claims to protect.
