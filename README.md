<div align="center">
  <img src="logo.png" alt="table-agent-toolkit logo" width="600" />
</div>

# table-agent-toolkit

An MCP server and Claude Code skill pack for working with tabular data files.

> [!NOTE]
> This MCP server will not expose any data online. However, the AI agent may choose to "inspect" the file contents before invoking the tool. Best way to try to prevent this is by specifying this behavior in the system prompt (CLAUDE.md or AGENTS.md).

## Installation

Requires [uv](https://docs.astral.sh/uv/). Install it from [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it.

Clone this repository and run `install.sh` or run the following command from your terminal for a 1-click installation:

```bash
curl -fsSL https://raw.githubusercontent.com/brains-group/table-agent-toolkit/main/scripts/quick-install.sh | bash
```

This registers the MCP server with both **Claude Code** and the **Claude desktop app**, and installs the bundled skills into `~/.claude/commands/table-agent-toolkit/`. Restart Claude Code and the desktop app afterward if they are already running.

Verify that the MCP server is running via:

**Claude Code**

Run `/mcp` to see list of regitered servers. You should see `table-agent-toolkit` in the list.

**Claude desktop app**

Click on your profile icon (bottom left) > Settings (Or press `Cmd+,` on Mac) > Developer. You should see `table-agent-toolkit` in the list of MCP servers.

## Supported file formats

| Format     | Extensions                        |
| ---------- | --------------------------------- |
| CSV        | `.csv`                            |
| TSV        | `.tsv`                            |
| Excel      | `.xls`, `.xlsx`, `.xlsm`, `.xlsb` |
| Parquet    | `.parquet`                        |
| Feather    | `.feather`                        |
| ORC        | `.orc`                            |
| JSON       | `.json`                           |
| JSON Lines | `.jsonl`, `.ndjson`               |
| Stata      | `.dta`                            |
| SAS        | `.sas7bdat`                       |
| SPSS       | `.sav`                            |

Stata, SAS, and SPSS support requires the optional `stats` extra:

```bash
pip install "table-agent-toolkit[stats]"
```

## MCP tools

### `summarize_table`

Returns a human-readable statistical summary of a tabular dataset. For every column it reports the dtype, null count, and either numerical distribution statistics (min, max, mean, std, quartiles) or value frequencies for categorical and boolean columns.

| Parameter   | Type     | Description                 |
| ----------- | -------- | --------------------------- |
| `file_path` | `string` | Path to the input data file |

---

### `generate_synthetic_data`

Generates synthetic tabular data that statistically resembles the original. Useful when you need more rows for testing or modeling.

| Parameter     | Type                  | Description                                                  |
| ------------- | --------------------- | ------------------------------------------------------------ |
| `file_path`   | `string`              | Path to the input data file                                  |
| `num_rows`    | `integer`             | Number of synthetic rows to generate                         |
| `backend`     | `string`              | Synthesis model — one of `tabicl`, `ctgan`, `tvae`           |
| `output_path` | `string` _(optional)_ | Destination path. Defaults to `<input_stem>_synthetic.<ext>` |

---

## Example dataset

`adult-sample.csv` -- a trimmed down version (1000 rows) of the [Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) dataset from the UCI Machine Learning Repository. It contains 48,842 rows of US Census data with demographic features (age, education, occupation, etc.) and a binary income label (`<=50K` / `>50K`). It is useful for demonstrating the summarization and synthetic-generation tools.

---

## Trying it out

MCP servers provide a list of "tools" that the agent can choose to invoke based on the context. To see the tools in action, you can now prompt your agent of choice like the following: *Make me a synthetic version of the @adult-sample.csv dataset using TabICL, let's say 500 rows.*. The agent should then invoke the `generate_synthetic_data` tool with the appropriate parameters and will report back the path to the generated file.


