# Reporting security issues

Please report suspected vulnerabilities privately to cesar.valdez@mila.quebec
before opening a public issue. Include the affected version, reproduction steps,
impact, and any proposed fix. Omit API keys, credentials, and private datasets.

Security fixes target the current release line (0.1.x). Please reproduce against
the latest release when possible. The project does not offer a guaranteed
response time or a bug bounty.

Adapters can execute local code with the caller's permissions. Use trusted task
configs and tool implementations. See the README's [execution boundary](README.md#trusted-execution)
for subprocess and cancellation limitations. Problems that violate this documented
boundary or expose credentials are appropriate for a private report.
