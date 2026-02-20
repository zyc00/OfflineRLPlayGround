Search past Claude Code conversation logs for experiment results, tables, and commands.

## Arguments

$ARGUMENTS - Search query describing what you're looking for. Can be keywords, a description of the experiment, a time range, or a question. Examples:
  - "GAE vs MC1 efficiency table"
  - "PPO finetuning from 60 to 100"
  - "pretrain critic commands"
  - "2月17号凌晨的ranking实验"
  - "IQL Q-network SNR problem results"

## Instructions

1. **Locate conversation logs** by listing all `.jsonl` files:
   ```
   ls -la ~/.claude/projects/-home-jigu-projects-OfflineRLPlayGround/*.jsonl
   ```

2. **Search across all conversation logs** using a Python script. For each `.jsonl` file, parse every line as JSON and extract:
   - `timestamp` — convert to UTC+8 local time
   - `message.role` — "user" or "assistant"
   - `message.content` — text content (handle both string and array formats)
   - For tool_use entries, extract `Bash` commands (`input.command`) that contain `python` or relevant keywords

3. **Filter by the user's query**:
   - If the query mentions a time range (e.g. "凌晨2-5点", "Feb 17"), filter by timestamp
   - If the query mentions keywords (e.g. "GAE", "MC1", "efficiency"), grep for those keywords in the text content
   - If the query is about commands, search tool_use entries for Bash commands
   - Always search for markdown tables (lines containing `|`) near matching content

4. **Display results** in a readable format:
   - Show the timestamp (UTC+8), role, and relevant text
   - For tables: show the COMPLETE table (all rows), not truncated
   - For commands: show the full command string
   - For experiment results: show surrounding context (what was asked, what was the result)
   - Group related messages together (question + answer pairs)

5. **Summarize findings** at the end:
   - List all tables found with a brief description
   - List all commands found
   - Note which session file each result came from (so the user can `claude --resume <session-id>`)

## Tips for searching

- Conversation content may be in English or Chinese
- Tables use markdown pipe syntax: `| col1 | col2 |`
- Experiment commands typically start with `python -m` or `python -u -m`
- Key metrics: SR (success rate), rho (Spearman correlation), loss, trajectories
- The user often pastes experiment stdout output in their messages
- Assistant messages contain analysis, tables, and summaries
- Tool results from sub-agents may contain detailed exploration reports — these are usually very long and can be skipped unless specifically relevant

## Python helper template

```python
import json, datetime, re, glob, sys

query_keywords = [k.lower() for k in "$ARGUMENTS".split()]

files = sorted(glob.glob(
    "/home/jigu/.claude/projects/-home-jigu-projects-OfflineRLPlayGround/*.jsonl"
))

for fpath in files:
    session_id = fpath.split("/")[-1].replace(".jsonl", "")
    with open(fpath, 'r') as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
                ts = d.get('timestamp')
                if not ts:
                    continue
                dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                local_dt = dt + datetime.timedelta(hours=8)

                msg = d.get('message', {})
                role = msg.get('role', '')
                content = msg.get('content', '')

                # Extract text
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get('type') == 'text':
                            parts.append(c.get('text', ''))
                    text = '\n'.join(parts)
                else:
                    continue

                text_lower = text.lower()
                if any(kw in text_lower for kw in query_keywords):
                    # Print match with context
                    pass
            except:
                pass
```

Adapt the script based on what the user is searching for. Always print complete tables when found.
