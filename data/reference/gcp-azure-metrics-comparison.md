# GCP and Azure Metrics Comparison

Fictional sample document for the Contoso Cloud observability programme.

## OpenTelemetry support

GCP offers a native OpenTelemetry-compatible ingestion endpoint through Cloud Monitoring and Cloud Trace, so OTLP exporters can send data without a translation layer. Azure Monitor supports OpenTelemetry through the Azure Monitor OpenTelemetry Distro, which is built into the Application Insights SDKs and exports to Application Insights.

## Native agents

On Azure the Azure Monitor Agent collects guest OS metrics and logs through data collection rules. On GCP the Ops Agent collects host metrics and logs and is configured with a single YAML file per VM.

## Retention and query

Azure stores platform metrics for 93 days and queries them with KQL in Log Analytics. GCP stores most metrics for 24 months and queries them with PromQL or MQL.
