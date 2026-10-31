# piccosta

The piccosta research harness is the best harness for LLMs ever released.

piccosta is one agent harness for every model you use:
- Claude, through your Claude Code login (your Claude plan);
- GPT, through your Codex login (your ChatGPT plan);
- the opencode-go models.

It has one session format, a terminal UI, a print mode for scripts, pi's rpc protocol (for T3 Code) and ACP (for Zed and other editors). It is written in Rust, based on the [pi coding agent](https://github.com/earendil-works/pi) and then rewritten and changed.

## What's in it

### Every model, on the plans you already pay for

- **Claude through Claude Code.** A Claude plan only works inside Claude Code, so piccosta runs Claude Code as a pipe. piccosta's own prompt and tools go in, and none of Claude Code's builtins, settings or CLAUDE.md. When a turn ends on tool calls, the query stays open for the results, so the prompt cache carries from turn to turn.
- **GPT through your Codex login**, over the Responses API. Its websocket sends only what's new each turn.
- **All 28 opencode-go models**, through four request formats: OpenAI chat completions, OpenAI Responses, Anthropic Messages and Google's generateContent.
- **Your own models** go in `models.json`, in pi's format.
- **One thinking scale for all of them**, from `off` to `max`, mapped to what each model actually takes.

### A terminal UI for several sessions at once

- **Sessions side by side.** The sidebar groups sessions by project, and up to four panes run at once.
- **Finding sessions again:**
  - Alt+↑↓ walks the list and shows earlier sessions without starting them until you type.
  - Ctrl+S searches the list, and `/resume all` reaches every project's sessions.
- **No double runs.** If another piccosta is running a session, you follow it live from its file instead of running it twice.
- **Typeset math.** RaTeX typesets formulas and shows them as images in Ghostty, kitty and WezTerm (through tmux too). Terminals that can't show images get Unicode.
- **The keys Claude Code taught you:**
  - Enter queues a message while a run goes, and Esc stops the run.
  - Ctrl+X Ctrl+S sends the queue now. Running commands move to the background so the model reads it straight away.
  - ↑ takes the queue back.
- **Copy and paste with the mouse.** Drag to select and it copies when you let go. Double-click selects a path, triple-click a line, and right-click pastes. Shift+arrows select in the editor.
- **Commands:**
  - `/compact` tells you what it saved.
  - `/fork`, `/rewind`, `/diff` (new files included), `/export`, `/usage`.
  - `/help` lists the rest, and Tab completes them.

### Models that check each other

`/pair <model>` makes a pair, and `/pair a b` (or `--pair a,b`) makes a trio. After each answer, every other model takes a turn checking it against what you asked. They verify, fix, finish, and say what they changed. Then it's your turn again, so they can't keep each other going. Any models can work together, across providers, and the group is saved with the session.

### An agent loop that doesn't get stuck

- **Long commands go to the background.** A command still running after a minute moves there by itself, and its result comes back as a message when it ends.
- **Loop guards.** piccosta stops a response that repeats itself, and a run that makes the same tool call five times in a row.
- **Nothing left running.** If piccosta is killed outright, as an editor stops an agent, a watcher process stops the commands it started.
- **Retries.** Overloads and dropped connections are retried with backoff. Used-up limits and withdrawn models fail straight away.
- **Compaction keeps your words.** When the context fills, older turns are summarized, and everything you typed before the cut stays word for word on top of the summary.
- **Failed edits say why.** A failed edit tells the model where the closest match is, or what changed since it last read the file. A missing path gets "did you mean" suggestions.

### Subagents

The model hands a task to a job: search, scout, worker or reviewer. You set which models may do each job, in order. The first one with a login runs it, and the next takes over if it fails. Each subagent is a piccosta of its own, with its own saved session, and it shows under its parent session in the sidebar while it works.

### Web tools

- **Search** goes through your own SearXNG, or through a search-capable model on an account you already have, so it needs no search API key.
- **Pages** come back as Markdown, read with Readability. PDFs come back as text. GitHub URLs become shallow clones the model can read.
- **Long results** are stored, so the model can page through them later instead of fetching again.

### Editors and scripts

- **Print mode.** `piccosta -p` answers and exits, and reads stdin too. `--mode json` prints one event per line. Failures exit non-zero.
- **T3 Code.** `--mode rpc` is pi's protocol, including approvals in T3's supervised mode.
- **Zed and other editors.** `--mode acp` is the Agent Client Protocol. It has Ask, Edits and Auto modes, permission cards that show each edit's diff, and live command output in the editor's terminal. Zed's pickers set the model and thinking level.
- **Your prompt files.** piccosta reads `AGENTS.md` and `CLAUDE.md` from the project and the folders above it. Sessions are saved in pi's JSONL format.

## Install

You need a recent stable Rust; it's built and used with 1.94.

```sh
git clone <this repo> piccosta && cd piccosta
cargo build --release
ln -s "$PWD/target/release/piccosta" ~/.local/bin/piccosta   # or anywhere on your PATH
```

Log in with each account's own tool, then check that piccosta finds the logins:

```sh
claude auth login        # Claude Code (claude.ai/code), for claude-bridge/* models
codex login              # the Codex CLI, for openai-codex/* models
opencode auth login      # or set OPENCODE_API_KEY, for opencode-go/* models
piccosta auth status
```

Claude's login stays inside Claude Code, and piccosta never sees it. For Codex, piccosta reads the token the Codex CLI keeps in `~/.codex/auth.json`, read-only, and sends it only to OpenAI. For opencode-go, it reads the key from opencode's own key file or from `OPENCODE_API_KEY`, and sends it only to opencode. piccosta stores no keys and never prints one.

## Use

```sh
piccosta                                     # the terminal UI, on your default model
piccosta --model claude-bridge/claude-opus-5-5 --thinking xhigh
piccosta -c                                  # this directory's most recent session
piccosta -r                                  # pick a session to resume (or -r <id or name>)
piccosta -p "summarize README.md"            # answer and exit; stdin is read too
piccosta -p --mode json "..."                # one event per line, for scripts
```

`piccosta --help` lists every flag. In the terminal UI:
- `alt+↑↓` moves between sessions;
- `^s` finds one;
- Delete twice moves the shown session to the trash;
- `/help` lists every key and command.

Inside tmux, add these to `~/.tmux.conf` so images, Shift+Enter and focus changes get through:

```
set -g allow-passthrough on
set -g extended-keys on
set -g extended-keys-format csi-u
set -g focus-events on
```

## In an editor or GUI

**Zed**, in `settings.json`:

```json
"agent_servers": {
  "piccosta": { "type": "custom", "command": "/path/to/piccosta", "args": ["--mode", "acp"] }
}
```

Zed's model, thinking and mode pickers work. Its Ask, Edits and Auto modes decide which tool calls ask first.

**T3 Code:** add a pi provider and set its binary path to `piccosta`. T3 speaks pi's rpc mode, which piccosta implements. Its supervised mode asks before each tool call.

Add `--yolo` to either to run every tool without asking.

## Settings

piccosta keeps its files in `~/.local/share/piccosta/agent`, or `$PICCOSTA_AGENT_DIR`: `settings.json`, `models.json`, sessions and subagent roles (`agents/<name>.md`).

`settings.json` uses pi's keys:
- `defaultProvider` and `defaultModel`;
- `defaultThinkingLevel` and `modelThinkingLevels`;
- `compaction`, `retry`, `shellCommandPrefix`, `transport` and `subagents`.

A project's `.pi/settings.json` is merged over them.

## License

piccosta is FSL-1.1-ALv2, licensor Synoros. You can use and change it freely, but selling it as a competing product needs Synoros. `LICENSE.md` has the license and the notices of the MIT projects piccosta is based on (pi, pi-web-access, pi-claude-bridge, cc-session-io).
