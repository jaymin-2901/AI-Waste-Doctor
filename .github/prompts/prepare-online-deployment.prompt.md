---
name: Prepare Online Deployment
description: Audit and prepare this Python AI application for reliable public online use, covering the FastAPI API, Streamlit frontend, model artifact, hosting configuration, security, and verification.
argument-hint: Describe the target users, preferred hosting platform, and whether deployment should be free or production-grade.
agent: agent
---
Prepare this repository for deployment so general users can access the AI Waste Doctor online:

${input:deployment_goal:Describe the target users, preferred hosting platform, budget, expected traffic, and whether the Windows desktop app must also remain supported}

Work from the actual repository and follow this workflow:

1. Audit the current architecture and clearly separate the local Windows PySide6 desktop app (`main.py`) from the online components. Treat the FastAPI service (`api.py`) and Streamlit browser frontend as the hosted path unless repository evidence requires another design.
2. Inspect the deployment configuration, dependency files, runtime version, model files, environment variables, CORS policy, upload handling, health checks, README instructions, and any platform-specific assumptions. Identify missing files, stale URLs, duplicated or contradictory settings, and configuration that could cause a clean deployment to fail.
3. Decide whether the existing Render API plus Streamlit Cloud architecture is sufficient for the stated goal. If another architecture is better, explain the tradeoff before changing it. Do not claim that a desktop GUI can run directly for arbitrary online users.
4. Apply the smallest necessary repository changes for a reproducible deployment. Keep secrets out of source control. Add or update documentation with exact build commands, start commands, required environment variables, model-artifact requirements, platform setup steps, public URLs, and rollback or redeploy notes.
5. Add practical production safeguards appropriate to this image API, including input size/type limits, useful error responses, bounded request time, CORS restricted to the deployed frontend when its URL is known, and health checks that distinguish service availability from model readiness. Avoid changes that make the local desktop application regress.
6. Validate locally with the cheapest meaningful checks: compile/import checks, dependency/config consistency checks, a local API health request, a representative `/predict` request when dependencies and model files permit, and a Streamlit smoke check when available. Report checks that could not run and why.
7. End with a deployment runbook for a non-developer: repository setup, model verification, API deployment, frontend deployment, environment configuration, smoke tests, troubleshooting, and ongoing maintenance. Include exact example commands and URLs with placeholders where values are unknown.

Respond using this structure:

- Architecture decision: what runs locally and what runs online.
- Findings: blockers first, then warnings and assumptions.
- Changes made: files changed and the reason for each.
- Local validation: exact commands and results.
- Deployment runbook: numbered steps for a first-time deploy.
- Public smoke test: exact `curl` or equivalent checks for `/`, `/health`, and `/predict`.
- Remaining risks: only unresolved items that require credentials, hosting access, a trained model, or user decisions.

Do not stop at generic advice when the repository can be inspected or edited. Do not report deployment as complete unless the available validation supports that claim.
