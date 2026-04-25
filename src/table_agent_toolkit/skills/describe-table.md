---
description: Describe what a tabular dataset contains in plain language
allowed-tools:
  - mcp__table-agent-toolkit__describe_table
---

Describe the tabular file at: $ARGUMENTS

Call `describe_table` on the file, then write a clear description covering:

**Overview** — what this dataset appears to be about, how many rows and columns.

**Columns** — for each column: what it likely represents, its type, and anything notable.
Group related columns together if it makes the description clearer.

**Data quality** — flag any columns with missing values, unexpected types, or suspicious
distributions (e.g. a numeric column with near-zero variance, a categorical column
with very high cardinality).

**Notable findings** — anything interesting or worth investigating further.
Write for someone who hasn't seen this data before. Be concise but complete.
