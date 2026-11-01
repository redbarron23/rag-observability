# GCP and Azure Metrics Comparison

Sample document for the fictional Contoso Cloud observability programme. It sets no Contoso policy. The Azure and Google Cloud capabilities it describes are real vendor features, last reviewed against the vendor documentation linked in each section on 1 November 2026.

## OpenTelemetry support

GCP accepts OTLP traces, metrics and logs natively through its Telemetry API (telemetry.googleapis.com), so OpenTelemetry SDKs and Collectors can export without a translation layer. Azure Monitor supports OpenTelemetry through the Microsoft OpenTelemetry Distro (.NET, Node.js, Python) and the Azure Monitor OpenTelemetry Distro (Java), which export to Application Insights. Direct OTLP ingestion into Azure Monitor is in preview.

Sources: [Google Cloud: Telemetry (OTLP) API overview](https://docs.cloud.google.com/stackdriver/docs/reference/telemetry/overview), [Microsoft Learn: Enable OpenTelemetry in Application Insights](https://learn.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-enable), [Microsoft Learn: Metrics in Azure Monitor](https://learn.microsoft.com/en-us/azure/azure-monitor/metrics/data-platform-metrics).

## Native agents

On Azure the Azure Monitor Agent collects guest OS performance counters and logs from VMs, configured centrally through data collection rules (DCRs). On GCP the Ops Agent collects host metrics (CPU, disk, memory, network, processes) and logs, and is configured with a single YAML file per VM that overrides the agent's built-in defaults.

Sources: [Microsoft Learn: Azure Monitor Agent overview](https://learn.microsoft.com/en-us/azure/azure-monitor/agents/azure-monitor-agent-overview), [Google Cloud: Configure the Ops Agent](https://docs.cloud.google.com/stackdriver/docs/solutions/agents/ops-agent/configuration).

## Retention and query

Azure platform metrics are kept for 93 days in the Azure Monitor metrics store, which is read with Metrics Explorer or the Metrics REST API, not KQL. KQL works only after a diagnostic setting exports the metrics to a Log Analytics workspace, where the workspace's retention applies.

GCP keeps agent, custom and some service metrics for 24 months and other metrics for 6 weeks, and is queried with PromQL. MQL is no longer offered in the console.

Sources: [Microsoft Learn: Metrics in Azure Monitor](https://learn.microsoft.com/en-us/azure/azure-monitor/metrics/data-platform-metrics), [Microsoft Learn: Diagnostic settings](https://learn.microsoft.com/en-us/azure/azure-monitor/data-collection/diagnostic-settings), [Google Cloud: Cloud Monitoring quotas and limits](https://docs.cloud.google.com/monitoring/quotas), [Google Cloud: Monitoring Query Language](https://docs.cloud.google.com/monitoring/mql).
