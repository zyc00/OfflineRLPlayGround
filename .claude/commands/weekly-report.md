扫描指定时间段内的所有 Claude Code 对话记录和实验日志，生成适合做 PPT 汇报的周报。

## Arguments

$ARGUMENTS - 时间范围或附加说明。例如:
  - "本周" / "this week" — 最近7天（默认）
  - "2月10日-2月17日" — 指定日期范围
  - "上周" — 上一个7天
  - "本周 重点offline RL" — 时间范围 + 主题过滤
  - "保存" / "save" — 同时保存到文件

如果留空，默认整理最近7天。

## Instructions

### Phase 1: 数据收集

同时从两个数据源收集信息：

**数据源 A: 对话日志 (.jsonl)**

用 Python 脚本扫描所有对话日志，提取指定时间段内的内容：

```python
import json, datetime, re, glob, os

# Parse time range from arguments
# Default: last 7 days
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
start_date = now - datetime.timedelta(days=7)
end_date = now

files = sorted(glob.glob(
    "/home/jigu/.claude/projects/-home-jigu-projects-OfflineRLPlayGround/*.jsonl"
))

sessions = []  # List of {session_id, start_time, end_time, topics, messages}

for fpath in files:
    session_id = os.path.basename(fpath).replace(".jsonl", "")
    file_mtime = datetime.datetime.fromtimestamp(
        os.path.getmtime(fpath),
        tz=datetime.timezone(datetime.timedelta(hours=8))
    )
    # Quick filter: skip files not modified in time range
    if file_mtime < start_date:
        continue

    messages = []
    tables = []
    commands = []
    timestamps = []

    with open(fpath, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                ts = d.get('timestamp')
                if not ts:
                    continue
                dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                local_dt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))

                if local_dt < start_date or local_dt > end_date:
                    continue

                timestamps.append(local_dt)
                msg = d.get('message', {})
                role = msg.get('role', '')
                content = msg.get('content', '')

                # Extract text
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            if c.get('type') == 'text':
                                parts.append(c.get('text', ''))
                            elif c.get('type') == 'tool_use' and c.get('name') == 'Bash':
                                cmd = c.get('input', {}).get('command', '')
                                if 'python' in cmd:
                                    commands.append(cmd)
                    text = '\n'.join(parts)
                else:
                    continue

                # Collect tables
                if '|' in text:
                    for tbl_match in re.finditer(
                        r'(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n?)+)', text
                    ):
                        tables.append(tbl_match.group(0))

                if role in ('user', 'assistant') and len(text.strip()) > 20:
                    messages.append({'role': role, 'text': text[:3000], 'time': local_dt})

            except Exception:
                pass

    if timestamps:
        sessions.append({
            'session_id': session_id,
            'start': min(timestamps),
            'end': max(timestamps),
            'message_count': len(messages),
            'tables': tables,
            'commands': commands,
            'messages': messages,  # Use for topic extraction
        })

# Sort by start time
sessions.sort(key=lambda s: s['start'])

# Print summary for each session
for s in sessions:
    print(f"\n{'='*60}")
    print(f"Session: {s['session_id']}")
    print(f"Time: {s['start'].strftime('%m-%d %H:%M')} ~ {s['end'].strftime('%m-%d %H:%M')}")
    print(f"Messages: {s['message_count']}")

    if s['commands']:
        print(f"\n--- Commands ({len(s['commands'])}) ---")
        for cmd in s['commands'][:15]:
            print(f"  $ {cmd[:200]}")

    if s['tables']:
        print(f"\n--- Tables ({len(s['tables'])}) ---")
        for tbl in s['tables'][:10]:
            print(tbl[:1000])

    # Print key user messages (likely contain research questions)
    user_msgs = [m for m in s['messages'] if m['role'] == 'user']
    if user_msgs:
        print(f"\n--- User messages (first 5) ---")
        for m in user_msgs[:5]:
            preview = m['text'][:300].replace('\n', ' ')
            print(f"  [{m['time'].strftime('%H:%M')}] {preview}")

    # Print key assistant conclusions (longer messages likely contain analysis)
    asst_msgs = [m for m in s['messages'] if m['role'] == 'assistant' and len(m['text']) > 200]
    asst_msgs.sort(key=lambda m: len(m['text']), reverse=True)
    if asst_msgs:
        print(f"\n--- Key assistant analyses (top 3 longest) ---")
        for m in asst_msgs[:3]:
            preview = m['text'][:500].replace('\n', ' ')
            print(f"  [{m['time'].strftime('%H:%M')}] {preview}...")
```

运行这个脚本（根据 $ARGUMENTS 调整时间范围），收集所有对话的摘要信息。

**数据源 B: 实验日志**

读取 `experiments/log.md`，提取时间范围内的实验条目。

### Phase 2: 分析与整理

基于收集到的数据，识别这一周的**研究主线**：

1. **按主题聚类**: 将多个 session 按研究主题分组（例如"Offline RL pretrain"、"Online finetuning"、"Data efficiency"）
2. **识别逻辑链**: 找出每个主题下的思路演进——从动机 → 假设 → 实验 → 结论
3. **提取关键数据**: 从对话和实验日志中收集所有结果表格

### Phase 3: 输出周报

按照以下 PPT 友好格式输出。每个大节对应 PPT 的一组 slides：

---

# Weekly Report: [日期范围]

## Overview (1 slide)
> **本周核心工作**: 用2-3句话概括本周做了什么、解决了什么问题
>
> **关键结论**: 用1-2句话说最重要的发现

## 研究主线 1: [主题名称]

### 动机与问题 (1 slide)
- 为什么要做这个实验？解决什么问题？
- 之前的方法有什么不足？

### 方法设计 (1 slide)
- 提出了什么方法/做了什么改进？
- 核心思想是什么？
- （如果有公式或算法，简要说明）

### 实验设置 (1 slide)
- 关键超参数表格（只列重要的，不要全部罗列）
- 对比了哪些 baseline / ablation

### 实验结果 (1-2 slides)
- 核心结果表格（完整保留，不省略）
- 关键曲线的描述（如果对话中有讨论）
- 用 **加粗** 标注最好的结果

### 分析与结论 (1 slide)
- 结果说明了什么？
- 有什么 insight？
- 有什么失败的尝试？原因是什么？

## 研究主线 2: [主题名称]
（同上结构）

...

## 失败实验与教训 (1 slide)
- 列出本周失败的尝试
- 每个失败的原因分析
- 从中学到了什么

## 下周计划 (1 slide)
- 基于本周结论，下一步要做什么
- 还有哪些未验证的假设
- 需要讨论的问题

## 附录: 本周实验索引

| # | 实验名称 | 时间 | 核心结果 | Session ID |
|---|----------|------|----------|------------|
| 1 | ... | ... | ... | `claude --resume xxx` |

---

## Rules

- **语言**: 中文为主，专业术语用英文（如 success rate, GAE, PPO）
- **表格**: 必须完整保留所有实验结果表格，不省略行列
- **数字**: 保留具体数字，不模糊化
- **每个主题的结论**: 必须有明确的 takeaway，不能只列数据不下结论
- **PPT 导向**: 每个小节标注 "(1 slide)" 帮助估算 PPT 页数，内容要简洁、有逻辑
- **失败实验**: 也要记录，包含原因分析
- **如果 $ARGUMENTS 包含 "保存" 或 "save"**: 将报告保存到 `experiments/weekly/YYYY-MM-DD.md`
- **如果 $ARGUMENTS 指定了主题**: 只整理该主题相关的内容
- 如果某些对话太短或只是工具开发/环境配置，可以跳过不纳入
- 对话中的思路演进很重要——不只是列结果，要体现 "为什么这么做 → 做了什么 → 发现了什么 → 所以下一步做什么" 的逻辑链
