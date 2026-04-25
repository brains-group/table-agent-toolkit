import json
import platform
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


COMMANDS_DIR = Path.home() / ".claude" / "commands" / "table-agent-toolkit"
SERVER_NAME = "table-agent-toolkit"
SERVER_CMD = "table-agent-toolkit-serve"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def _desktop_config_path() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def install_skills():
    skills_src = files("table_agent_toolkit").joinpath("skills")
    COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for skill_file in skills_src.iterdir():
        if skill_file.name.endswith(".md"):
            dest = COMMANDS_DIR / skill_file.name
            shutil.copy2(str(skill_file), dest)
            copied.append(skill_file.name)
    return copied


def register_claude_code(exe: str):
    result = subprocess.run(
        ["claude", "mcp", "add", "--scope", "user", SERVER_NAME, "--", exe],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "already exists" not in result.stderr:
            print(f"Warning: MCP registration may have failed:\n{result.stderr}", file=sys.stderr)
            return False
    return True


def register_claude_desktop(exe: str):
    config_path = _desktop_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass
    data.setdefault("mcpServers", {})[SERVER_NAME] = {"command": exe}
    config_path.write_text(json.dumps(data, indent=2) + "\n")
    return config_path


def register_codex(exe: str) -> Path | None:
    if not CODEX_CONFIG_PATH.exists():
        return None
    import tomllib
    import tomli_w

    try:
        data = tomllib.loads(CODEX_CONFIG_PATH.read_text())
    except Exception:
        data = {}
    data.setdefault("mcp_servers", {})[SERVER_NAME] = {"command": exe}
    CODEX_CONFIG_PATH.write_text(tomli_w.dumps(data))
    return CODEX_CONFIG_PATH


def main():
    print("Setting up table-agent-toolkit...")

    exe = shutil.which(SERVER_CMD)
    if exe is None:
        print(f"  Warning: '{SERVER_CMD}' not found in PATH; MCP registration skipped.", file=sys.stderr)
    else:
        print("  Registering with Claude Code...", end=" ", flush=True)
        if shutil.which("claude"):
            ok = register_claude_code(exe)
            print("done" if ok else "skipped (already registered)")
        else:
            print("skipped (claude CLI not found)")

        print("  Registering with Claude Desktop...", end=" ", flush=True)
        config_path = register_claude_desktop(exe)
        print(f"done ({config_path})")

        print("  Registering with Codex...", end=" ", flush=True)
        codex_path = register_codex(exe)
        if codex_path:
            print(f"done ({codex_path})")
        else:
            print("skipped (no ~/.codex/config.toml found)")

    print(f"  Installing skills to {COMMANDS_DIR}...", end=" ", flush=True)
    copied = install_skills()
    print(f"done ({len(copied)} skills)")
    for name in copied:
        print(f"    /{name[:-3]}")

    print("\nAll set! Restart Claude Code and Claude Desktop if they are already running.")
