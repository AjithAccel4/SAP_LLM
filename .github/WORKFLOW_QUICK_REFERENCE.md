# GitHub Actions Workflow Quick Reference

## 🚀 Quick Commands

### View Workflows
```bash
# List recent workflow runs
gh run list

# View specific run
gh run view <run-id>

# Watch running workflow
gh run watch
```

### Trigger Workflows Manually
```bash
# Run full test suite
gh workflow run test.yml

# Run specific test suite
gh workflow run test.yml -f test_suite=unit
gh workflow run test.yml -f test_suite=integration
gh workflow run test.yml -f test_suite=performance
gh workflow run test.yml -f test_suite=security

# Run security scan
gh workflow run security.yml

# Create release
gh workflow run release.yml -f version=1.2.3 -f prerelease=false

# Deploy to environment
gh workflow run cd.yml -f environment=staging
gh workflow run cd.yml -f environment=production
```

### Download Artifacts
```bash
# List artifacts from a run
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>

# Download specific artifact
gh run download <run-id> -n test-results
```

## 📋 Workflow Triggers

| Workflow | Auto Trigger | Manual | Schedule |
|----------|--------------|--------|----------|
| **ci.yml** | Push/PR to main, develop, feature/* | ✅ | ❌ |
| **cd.yml** | Tags: v*.*.* | ✅ | ❌ |
| **test.yml** | Push/PR | ✅ | Daily 2 AM UTC |
| **security.yml** | Push/PR | ✅ | Daily 3 AM UTC |
| **release.yml** | Tags: v*.*.* | ✅ | ❌ |

## 🎯 Workflow Matrix

### CI Workflow
```yaml
Python Versions: [3.9, 3.10, 3.11]
Jobs:
  - Code Quality (15 min)
  - Unit Tests (30 min)
  - Integration Tests (45 min)
  - Docker Build (45 min)
  - Artifacts (15 min)
Total: ~45 min
```

### Test Workflow
```yaml
Python Versions: [3.9, 3.10, 3.11]
Test Groups: [core, api, models, stages, utils]
Integration Suites: [api, pmg, apop, shwl, knowledge_base]
Jobs:
  - Unit Tests (Matrix)
  - Integration Tests (Matrix)
  - Performance Tests
  - Security Tests
  - E2E Tests
Total: ~90 min
```

### Security Workflow
```yaml
Scans:
  - Dependency (Safety, pip-audit)
  - Code (Bandit, Semgrep)
  - Container (Trivy, Grype)
  - Secrets (Gitleaks, TruffleHog)
  - CodeQL
  - IaC (Checkov)
  - License
Total: ~40 min
```

### CD Workflow
```yaml
Stages:
  1. Build Production Images (60 min)
  2. Deploy Staging (30 min)
  3. Smoke Tests Staging (15 min)
  4. Deploy Production - Canary (45 min)
     - 10% traffic (5 min monitor)
     - 50% traffic (3 min monitor)
     - 100% traffic
  5. Smoke Tests Production (15 min)
Total: ~90 min
```

### Release Workflow
```yaml
Steps:
  1. Validate Release (10 min)
  2. Build Artifacts (20 min)
  3. Build Docker (45 min)
  4. Release Tests (30 min)
  5. Generate Notes (10 min)
  6. Create Release (15 min)
  7. Publish PyPI (10 min)
  8. Update Docs (15 min)
Total: ~30 min
```

## 🔄 Common Workflows

### Feature Development
```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "feat: Add my feature"

# Push and create PR
git push origin feature/my-feature
gh pr create --base develop

# CI/Test/Security workflows run automatically
# Review checks in PR
# Merge when all checks pass
```

### Hotfix
```bash
# Create hotfix branch from main
git checkout -b hotfix/critical-bug main

# Fix and commit
git commit -m "fix: Critical bug"

# Push and create PR to main
git push origin hotfix/critical-bug
gh pr create --base main

# Fast-track review and merge
# Create patch release
git tag -a v1.2.4 -m "Hotfix 1.2.4"
git push --tags
```

### Release Process
```bash
# 1. Update version in pyproject.toml
sed -i 's/version = ".*"/version = "1.2.3"/' pyproject.toml

# 2. Commit version bump
git add pyproject.toml
git commit -m "chore: Bump version to 1.2.3"
git push origin main

# 3. Create and push tag
git tag -a v1.2.3 -m "Release 1.2.3

- Feature 1
- Feature 2
- Bug fix 1"

git push origin v1.2.3

# 4. Release workflow runs automatically
# 5. CD workflow deploys to production
# 6. Monitor deployment in Actions tab
```

## 🛠️ Debugging

### View Failed Jobs
```bash
# List failed runs
gh run list --status failure

# View logs of failed run
gh run view <run-id> --log-failed

# Re-run failed jobs
gh run rerun <run-id> --failed
```

### Cancel Running Workflow
```bash
# Cancel specific run
gh run cancel <run-id>

# Cancel latest run of a workflow
gh run cancel $(gh run list --workflow=ci.yml --limit=1 --json databaseId -q '.[0].databaseId')
```

### Download Logs
```bash
# Download logs for offline analysis
gh run download <run-id> --log

# View specific job log
gh run view <run-id> --log --job=<job-id>
```

## 📊 Status Badges

Add to README.md:

```markdown
![CI](https://github.com/qorsync/sap-llm/workflows/Continuous%20Integration/badge.svg)
![Tests](https://github.com/qorsync/sap-llm/workflows/Comprehensive%20Testing/badge.svg)
![Security](https://github.com/qorsync/sap-llm/workflows/Security%20Scanning/badge.svg)
[![codecov](https://codecov.io/gh/qorsync/sap-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/qorsync/sap-llm)
```

## 🔐 Security Quick Check

```bash
# Run security scan locally
bandit -r sap_llm/
safety check
pip-audit

# Check for secrets
git secrets --scan
gitleaks detect

# Scan Docker image
trivy image sap-llm:latest
```

## 📈 Performance Monitoring

```bash
# Run performance tests locally
pytest tests/ -m performance --benchmark-only

# Monitor metrics in workflow
gh run view <run-id> --log | grep "benchmark"

# Download benchmark results
gh run download <run-id> -n performance-results
```

## 🎯 Coverage Targets

| Test Type | Target | Status |
|-----------|--------|--------|
| Unit Tests | ≥ 80% | ✅ |
| Integration | ≥ 60% | ✅ |
| Overall | ≥ 70% | ✅ |

## ⚡ Optimization Tips

1. **Use Caching**:
   - Pip cache enabled by default
   - Docker layer caching enabled
   - Pre-commit hooks cached

2. **Parallel Execution**:
   - Tests run with `-n auto` (pytest-xdist)
   - Matrix jobs run in parallel
   - Independent workflows run concurrently

3. **Selective Testing**:
   ```bash
   # Run only changed tests
   pytest --lf  # last failed
   pytest --ff  # failed first
   ```

4. **Skip Tests** (when safe):
   ```bash
   gh workflow run cd.yml -f skip_tests=true
   ```

## 🔔 Notifications

Set up notifications in `.github/workflows/*.yml`:

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 📱 Mobile Monitoring

Use GitHub Mobile app:
- View workflow runs
- Check job status
- Read logs
- Approve deployments

## 🆘 Emergency Procedures

### Rollback Production
```bash
# Option 1: Use GitHub UI
# Go to Actions → CD workflow → Re-run previous successful deployment

# Option 2: Rollback Helm release
kubectl rollout undo deployment/sap-llm-prod -n sap-llm-prod

# Option 3: Deploy previous version
git tag  # find previous version
gh workflow run cd.yml -f environment=production
```

### Stop Deployment
```bash
# Cancel running deployment
gh run cancel <run-id>

# Manual intervention in Kubernetes
kubectl rollout pause deployment/sap-llm-prod -n sap-llm-prod
```

### Emergency Hotfix
```bash
# Bypass normal workflow (use with caution)
git checkout -b emergency-hotfix main
# make critical fix
git commit -m "fix: Emergency fix for production issue"
git tag -a v1.2.4 -m "Emergency hotfix"
git push origin main --tags
# Manually monitor deployment
```

## 📚 References

- [GitHub Actions Docs](https://docs.github.com/actions)
- [gh CLI Manual](https://cli.github.com/manual/)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)

---

**Quick Access**: Bookmark this page for fast reference during development and deployment.
