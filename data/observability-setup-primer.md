# Observability Setup Primer

Fictional sample document for the Contoso Cloud observability programme.

## Prerequisites

Before onboarding a new service, confirm that the service has a named owning team, that its resources carry the required tags, and that the team has access to the central observability workspace.

## Onboarding steps

1. Register the service in the service catalogue with its tier.
2. Install the cloud agent (Azure Monitor Agent or Ops Agent) or enable the OpenTelemetry SDK.
3. Enable structured logging with a correlation ID.
4. Create the standard dashboard from the template.
5. Define at least one availability SLO and link the alert to a runbook.
6. Request a framework scoring review after two weeks of data.
