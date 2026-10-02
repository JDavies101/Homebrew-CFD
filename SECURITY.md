# Security policy

## Supported versions

Homebrew CFD is pre-release software. I fix security issues in the latest release only.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: on the repository page, open the
**Security** tab and choose **Report a vulnerability**. Do not open a public issue for a
security problem.

I aim to acknowledge a report within a week and to publish a fix, with credit if wanted,
in the next release.

## Verifying a download

Releases are built only by GitHub Actions from a tagged commit, never on a local machine.
Each release lists SHA-256 checksums (`SHA256SUMS.txt`) and carries a build-provenance
attestation. To check an installer:

```
gh attestation verify HomebrewCFD-<version>-setup.exe --repo JDavies101/Homebrew-CFD
```

The installer is not code-signed yet, so Windows SmartScreen may show an "unknown publisher"
prompt on first run.

## Scope

The application reads case files as data only (JSON; no pickle, eval or embedded scripts)
and makes no network calls.
