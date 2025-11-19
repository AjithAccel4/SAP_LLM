# Security and Compliance Guide

## Overview

This document provides enterprise-level security and compliance guidelines for dependency management in the SAP LLM project, updated for 2025 best practices.

## Table of Contents

1. [Supply Chain Security](#supply-chain-security)
2. [Hash Verification](#hash-verification)
3. [Security Scanning Tools](#security-scanning-tools)
4. [SBOM Generation](#sbom-generation)
5. [Commercial Licensing](#commercial-licensing)
6. [Compliance Frameworks](#compliance-frameworks)

---

## Supply Chain Security

### Overview

Supply chain attacks targeting software dependencies are a critical threat. This project implements multiple layers of defense:

1. **Exact Version Pinning**: All dependencies use `==` for deterministic builds
2. **Hash Verification**: Cryptographic hashes verify package integrity
3. **SBOM Generation**: Complete inventory of all components
4. **Automated Scanning**: Daily vulnerability checks
5. **License Compliance**: Automated license verification

### Threat Model

**Threats Mitigated:**
- ✅ Typosquatting attacks
- ✅ Package substitution
- ✅ Dependency confusion
- ✅ Compromised package maintainers
- ✅ Malicious package updates

**Residual Risks:**
- ⚠️ Zero-day vulnerabilities (mitigated by daily scans)
- ⚠️ Git dependencies (detectron2) - use specific tags
- ⚠️ Compromised build systems (mitigated by hash verification)

---

## Hash Verification

### What is Hash Verification?

Hash verification uses cryptographic hashes (SHA-256) to ensure that installed packages are bit-for-bit identical to what was tested. This prevents:

- Package tampering during download
- Man-in-the-middle attacks
- Compromised package indexes
- Supply chain poisoning

### Implementation

#### Using pip-tools (Recommended for Enterprise)

```bash
# Install pip-tools
pip install pip-tools==7.3.0

# Generate hashed requirements from requirements.in
pip-compile --generate-hashes --output-file requirements-lock-hashed.txt requirements.in

# Install with hash verification (enterprise production)
pip install --require-hashes -r requirements-lock-hashed.txt
```

#### Hashed Requirements Format

```
torch==2.1.0 \
    --hash=sha256:abc123... \
    --hash=sha256:def456... \
    --hash=sha256:ghi789...
```

Multiple hashes support different platform wheels (Linux, macOS, Windows).

### Benefits

| Feature | Standard requirements.txt | Hashed requirements |
|---------|--------------------------|---------------------|
| Version pinning | ✅ Yes | ✅ Yes |
| Package integrity | ❌ No | ✅ Yes |
| Supply chain protection | ❌ Limited | ✅ Strong |
| Reproducibility | ✅ Good | ✅ Excellent |
| Compliance ready | ⚠️ Basic | ✅ Advanced |

### Limitations

1. **Git Dependencies**: Cannot generate hashes for git URLs (e.g., detectron2)
   - **Workaround**: Install separately with pinned tag/commit

2. **Platform-Specific**: Different platforms have different hashes
   - **Solution**: pip-tools generates all platform hashes automatically

3. **Update Process**: More complex when updating dependencies
   - **Solution**: Automated via pip-compile

### Enterprise Deployment Process

```bash
# Development: Generate hashed lock file
pip-compile --generate-hashes requirements.in

# CI/CD: Verify hashes during build
pip install --require-hashes -r requirements-lock-hashed.txt

# Production: Deploy with verified packages
# All packages verified bit-for-bit before deployment
```

### Regenerating Hashes

```bash
# After updating any dependency version
pip-compile --upgrade --generate-hashes requirements.in

# Review changes
git diff requirements-lock-hashed.txt

# Commit updated hashes
git add requirements-lock-hashed.txt
git commit -m "chore: Update dependency hashes"
```

---

## Security Scanning Tools

### pip-audit (Recommended - Open Source)

**License**: Apache 2.0 (Free for commercial use)
**Database**: PyPI Advisory Database (open source)
**Maintainer**: Python Packaging Authority (PyPA)

#### Why pip-audit?

- ✅ **Free for commercial use** - No licensing fees
- ✅ **Open source database** - Transparent and community-driven
- ✅ **Official PyPA tool** - Well-maintained and trustworthy
- ✅ **Automatic fixes** - Suggests remediation versions
- ✅ **Multiple formats** - JSON, plain text, CycloneDX

#### Usage

```bash
# Basic scan
pip-audit -r requirements.txt

# Detailed scan with descriptions
pip-audit -r requirements.txt --desc

# JSON output for automation
pip-audit -r requirements.txt --format json --output audit-report.json

# Fix vulnerabilities automatically (interactive)
pip-audit -r requirements.txt --fix

# Dry-run to see what would be fixed
pip-audit -r requirements.txt --fix --dry-run
```

#### Enterprise Integration

```yaml
# GitHub Actions (already configured)
- name: Run pip-audit
  run: pip-audit -r requirements.txt --desc
```

### Safety (Commercial License Required)

**⚠️ IMPORTANT: Safety requires a commercial license for production use**

**License**: Safety DB is free for non-commercial/personal use only
**Commercial Use**: Requires Safety Pro subscription ($$$)
**Database**: SafetyDB (proprietary)

#### Licensing Terms

From Safety documentation:
> Safety DB is licensed for non-commercial use only. For all commercial projects,
> Safety must be upgraded to use a PyUp API key.

#### When to Use Safety

- ❌ **Do NOT use** for commercial/enterprise projects without a license
- ❌ **Do NOT use** in CI/CD for commercial code (violates ToS)
- ✅ **Use pip-audit instead** for free commercial use
- ✅ **Purchase Safety Pro** if you need additional features:
  - Priority vulnerability notifications
  - Advanced remediation suggestions
  - Commercial support
  - Private vulnerability database

#### Migration from Safety to pip-audit

```bash
# Old approach (requires commercial license)
# safety check -r requirements.txt

# New approach (free for commercial use)
pip-audit -r requirements.txt --desc
```

#### If You Have a Safety License

```bash
# Set your API key
export SAFETY_API_KEY="your-api-key-here"

# Run Safety with commercial license
safety check -r requirements.txt --key=$SAFETY_API_KEY

# Or configure in CI/CD
# Add SAFETY_API_KEY to GitHub Secrets
# Uncomment safety-scan job in .github/workflows/dependency-scan.yml
```

### Bandit (Code Security Linter)

Scans Python code for security issues (separate from dependency scanning).

```bash
# Install
pip install bandit==1.7.6

# Scan codebase
bandit -r sap_llm/

# Generate report
bandit -r sap_llm/ -f json -o bandit-report.json
```

### Comparison Matrix

| Feature | pip-audit | Safety (Free) | Safety (Pro) |
|---------|-----------|---------------|--------------|
| **Cost** | Free | Free (non-commercial) | $$$  (subscription) |
| **Commercial Use** | ✅ Yes | ❌ No | ✅ Yes |
| **Database** | PyPI Advisory | SafetyDB | SafetyDB Pro |
| **Auto-fix** | ✅ Yes | ❌ No | ✅ Yes |
| **CI/CD Ready** | ✅ Yes | ⚠️ Personal only | ✅ Yes |
| **Support** | Community | None | Commercial |
| **Updates** | Daily | Weekly | Real-time |
| **Recommended For** | All projects | Personal projects | Enterprise (if budget) |

**Enterprise Recommendation**: Use **pip-audit** (it's free, open source, and maintained by PyPA)

---

## SBOM Generation

### What is an SBOM?

A Software Bill of Materials (SBOM) is a complete inventory of all components in your software. Required by:

- US Executive Order 14028 (Cybersecurity)
- NIST frameworks (SSDF, SP 800-218)
- EU Cyber Resilience Act
- Many enterprise procurement requirements

### CycloneDX Format

This project uses CycloneDX, an OWASP standard for SBOM.

```bash
# Install CycloneDX
pip install cyclonedx-bom==4.4.0

# Generate SBOM
cyclonedx-py requirements -r requirements.txt -o sbom.json --format json
cyclonedx-py requirements -r requirements.txt -o sbom.xml --format xml
```

### Automated Generation

SBOMs are automatically generated in CI/CD:

- **Format**: CycloneDX 1.5 (JSON + XML)
- **Frequency**: Every build, PR, and daily
- **Retention**: 90 days
- **Location**: GitHub Actions artifacts

### SBOM Contents

An SBOM includes:

- Component name and version
- Supplier information
- Dependency tree
- License information
- Package hashes
- Vulnerability status (when integrated with scanning)

### Usage

```bash
# Download SBOM from GitHub Actions artifacts
# Import into vulnerability management system
# Share with customers/compliance teams
# Required for government contracts
```

---

## Commercial Licensing

### Project Dependencies

All dependencies use **commercial-friendly licenses**:

| License | Commercial Use | Redistribution | Attribution Required |
|---------|----------------|----------------|---------------------|
| MIT | ✅ Yes | ✅ Yes | ✅ Yes |
| Apache-2.0 | ✅ Yes | ✅ Yes | ✅ Yes |
| BSD-3-Clause | ✅ Yes | ✅ Yes | ✅ Yes |
| BSD-2-Clause | ✅ Yes | ✅ Yes | ✅ Yes |
| ISC | ✅ Yes | ✅ Yes | ⚠️ Optional |

### Blocked Licenses

The following licenses are **automatically blocked** by CI/CD:

- ❌ **GPL-2.0** - Copyleft, requires source disclosure
- ❌ **GPL-3.0** - Copyleft, additional restrictions
- ❌ **AGPL-3.0** - Network copyleft (most restrictive)
- ❌ **LGPL** - Limited copyleft (requires review)

### License Scanning

```bash
# Install pip-licenses
pip install pip-licenses==4.3.3

# List all licenses
pip-licenses

# Export for review
pip-licenses --format=markdown --output-file=licenses.md

# Check for restricted licenses
pip-licenses --fail-on GPL-2.0 --fail-on GPL-3.0 --fail-on AGPL-3.0
```

### Compliance Process

1. **Automated Scanning**: Every PR and daily builds
2. **Block on Violation**: CI fails if GPL/AGPL detected
3. **Manual Review**: Any new licenses require legal review
4. **Documentation**: All licenses tracked in DEPENDENCY_VERSIONS.md

---

## Compliance Frameworks

### SLSA (Supply-chain Levels for Software Artifacts)

This project implements SLSA Level 2 practices:

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Version control | Git + GitHub | ✅ |
| Build service | GitHub Actions | ✅ |
| Provenance | Automated workflows | ✅ |
| Reproducible builds | Exact version pinning | ✅ |
| Dependency tracking | requirements-lock.txt | ✅ |

### NIST SP 800-218 (Secure Software Development)

| Practice | Implementation |
|----------|----------------|
| PO.1.1: Define security requirements | Security policy documented |
| PO.3.1: Check for known vulnerabilities | Daily pip-audit scans |
| PO.3.2: Identify components/dependencies | SBOM generation |
| PS.1.1: Protect all code integrity | Hash verification |
| PS.2.1: Build/deployment automated | GitHub Actions CI/CD |
| PW.4.1: Review/test security | Security scan in PR |

### SOC 2 Type II

Relevant controls implemented:

- **CC6.1**: Logical access controls (exact versions, no wildcards)
- **CC6.6**: Vulnerability management (automated scanning)
- **CC7.1**: System monitoring (daily scans, alerts)
- **CC7.2**: Anomaly detection (hash verification)

### ISO 27001

Relevant controls:

- **A.12.6.1**: Management of technical vulnerabilities (automated scanning)
- **A.14.2.1**: Secure development policy (documented in DEPENDENCY_MANAGEMENT.md)
- **A.14.2.5**: Secure system engineering (hash verification, SBOM)

---

## Best Practices Checklist

### For All Deployments

- [ ] Use exact version pinning (`==`) in all requirements files
- [ ] Run pip-audit before each deployment
- [ ] Generate and archive SBOM for each release
- [ ] Verify all licenses are commercial-compatible
- [ ] Document any dependency changes in CHANGELOG

### For Enterprise/Government Deployments

- [ ] Enable hash verification with pip-compile
- [ ] Install using `--require-hashes` flag
- [ ] Generate SBOM in CycloneDX format
- [ ] Perform quarterly dependency audits
- [ ] Maintain vulnerability response SLA (24h critical, 1w high)
- [ ] Archive all SBOMs for compliance (7+ years)
- [ ] Purchase commercial security scanning if required (Safety Pro)
- [ ] Implement SLSA Level 3+ if needed

### For Continuous Improvement

- [ ] Review and update dependencies monthly
- [ ] Monitor PyPI Advisory Database for new CVEs
- [ ] Participate in security bug bounty programs
- [ ] Conduct annual third-party security audits
- [ ] Train development team on supply chain security
- [ ] Test disaster recovery (rollback procedures)

---

## Resources

### Official Documentation

- [pip-audit](https://github.com/pypa/pip-audit) - PyPA official tool
- [pip-tools](https://github.com/jazzband/pip-tools) - Requirements management
- [CycloneDX](https://cyclonedx.org/) - SBOM standard
- [PyPI Advisory Database](https://github.com/pypa/advisory-database) - Vulnerability DB

### Security Frameworks

- [SLSA](https://slsa.dev/) - Supply chain security levels
- [NIST SP 800-218](https://csrc.nist.gov/publications/detail/sp/800-218/final) - Secure software development
- [OWASP Dependency-Check](https://owasp.org/www-project-dependency-check/) - Dependency analysis

### Compliance

- [NTIA SBOM](https://www.ntia.gov/page/software-bill-materials) - US SBOM initiative
- [EU Cyber Resilience Act](https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act) - EU requirements

### Commercial Tools (if needed)

- [Safety Pro](https://safetycli.com/) - Commercial vulnerability scanning
- [Snyk](https://snyk.io/) - Developer security platform
- [Sonatype Nexus](https://www.sonatype.com/products/sonatype-nexus-repository) - Repository management

---

**Last Updated**: 2025-11-19
**Maintained By**: SAP LLM Security Team
**Review Cycle**: Quarterly
**Next Review**: 2026-02-19
