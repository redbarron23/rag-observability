# Distributed Tracing Use Cases

Fictional sample document for the Contoso Cloud observability programme.

## Latency investigation

Traces show where time is spent across services, for example a slow checkout caused by a downstream inventory call taking 1.8 seconds.

## Dependency mapping

Trace data produces an up-to-date service dependency map, which helps teams find undocumented dependencies before a migration.

## Error propagation

A trace links an error at the edge to the failing component several hops away, shortening mean time to resolution.

## Cost attribution

Spans tagged with the owning service allow shared infrastructure costs, such as message brokers, to be attributed to the teams that use them.
