# Deployment Runbook

Deployments to production run through the release pipeline and require a
green build on main plus one approving review. The pipeline runs unit tests,
integration tests against a staging database, and a smoke test suite that
exercises the checkout flow end to end.

Rollbacks are performed with the deploy tool using the previous release tag.
A rollback takes approximately four minutes to propagate across all regions.
If a rollback does not resolve the incident within ten minutes, escalate to
the platform team rather than attempting further fixes.

The deployment freeze runs from December 20th through January 2nd. Emergency
fixes during the freeze require VP Engineering approval and a written
incident report filed within 24 hours of the deployment.

This document is maintained by the People Operations team. For questions,
contact peopleops@northwind.example. Last reviewed March 2026.
