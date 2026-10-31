# piccosta

The piccosta research harness is the best harness for LLMs ever released.

piccosta is one agent harness for every model you use:
- **Claude**, through your Claude Code login (your Claude plan);
- **GPT**, through your Codex login (your ChatGPT plan);
- the **opencode-go** models.

It has one session format, a terminal UI, a print mode for scripts, pi's rpc protocol (for T3 Code) and ACP (for Zed and other editors). It is written in Rust. It was based on the [pi coding agent](https://github.com/earendil-works/pi), then rewritten and changed.

## What's in it

### Every model, on the plans you already pay for

- **Claude through Claude Code.** A Claude plan only works inside Claude Code, so piccosta runs Claude Code as a pipe: piccosta's own prompt and tools go in, with none of Claude Code's builtins, settings or CLAUDE.md. A turn that ends on tool calls keeps its query open for the results, so the prompt cache carries from turn to turn.
- **GPT through your Codex login**, over the Responses API. Its websocket sends only what's new each turn.
- **All 28 opencode-go models**, through four request formats: OpenAI chat completions, OpenAI Responses, Anthropic Messages and Google's generateContent.
- **Your own models** in `models.json`, in pi's format.
- **One thinking scale for all of them**, from `off` to `max`, set to what each model actually takes.

### A terminal UI for several sessions at once

- **Sessions side by side.** The sidebar groups sessions by project, and up to four panes run side by side.
- **Sessions you can find again:**
  - Alt+↑↓ walks the list, showing earlier sessions without starting them until you type.
  - Ctrl+S searches the list, and `/resume all` reaches every project's.
- **No double runs.** A session another piccosta is running is followed live from its file, never run twice.
- **Typeset math.** Formulas are typeset by RaTeX and shown as images in Ghostty, kitty and WezTerm (through tmux too), or drawn as Unicode where a terminal can't show images.
- **The keys Claude Code taught you:**
  - Enter queues a message while a run goes, and Esc stops the run.
  - Ctrl+X Ctrl+S sends the queue now: running commands move to the background so the model reads it straight away.
  - ↑ takes the queue back.
- **Copy and paste with the mouse:** drag to select, and it copies when you let go. Double-click selects a path, triple-click a line. Right-click pastes. Shift+arrows select in the editor.
- **Commands people reach for:**
  - `/model` becomes the default, and `/effort` is saved for that model.
  - `/compact` says what it saved.
  - `/fork`, `/rewind`, `/diff` (new files included), `/export`, `/usage`.
  - `/help` has the rest, and Tab completes them.

### Models that check each other

`/pair <model>` makes a pair, and `/pair a b` (or `--pair a,b`) a trio. After each answer, the other models take a turn each checking it against what you asked: they verify, fix, finish, and say what they changed. Then it's your turn again, so they can't keep each other going. Any models can work together, across providers, and the group is saved with the session.

### An agent loop that doesn't get stuck

- **Long commands go to the background.** A command still running after a minute moves there by itself, and its result comes back as a message when it ends. Nothing waits in the foreground, and the model is told not to sleep or poll.
- **Loop guards.** A response stuck repeating itself is stopped, and so is a run that makes the same tool call five times in a row.
- **Nothing left running.** If piccosta is killed outright, as an editor stops an agent, a watcher process stops the commands it started.
- **Retries that know the difference.** Overloads and dropped connections are retried with backoff, while used-up limits and withdrawn models fail straight away.
- **Compaction that keeps your words.** When the context fills, older turns are summarized, and everything you typed before the cut is kept word for word on top of the summary.
- **Edits that recover.** A failed edit tells the model where the closest match is, or what changed since it last read the file. A missing path gets "did you mean" suggestions.

### Subagents

The model hands a task to a job: search, scout, worker or reviewer. You set which models may do each job, in order. The first one with a login runs it, and the next takes over if it fails. Each subagent is a piccosta of its own, with its own saved session. While it works, it shows under its parent session in the sidebar.

### Web tools

- **Search** goes through your own SearXNG, or through a search-capable model on an account you already have, so no search API key is needed.
- **Pages** come back as Markdown, read with Readability. PDFs come back as text. GitHub URLs become shallow clones the model can read.
- **Long results** are stored, so the model can page through them later instead of fetching again.

### Editors and scripts

- **Print mode.** `piccosta -p` answers and exits, reading stdin too. `--mode json` prints one event per line. Failures exit non-zero.
- **T3 Code.** `--mode rpc` is pi's protocol, including approvals in T3's supervised mode.
- **Zed and other editors.** `--mode acp` is the Agent Client Protocol: Ask, Edits and Auto modes, permission cards that show each edit's diff, and live command output in the editor's terminal. Zed's pickers set the model and thinking level.
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

piccosta reads those logins where the tools keep them, and never prints a key.

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
- `/help` has every key and command.

Inside tmux, add these to `~/.tmux.conf`. They let images, Shift+Enter and focus changes through:

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

piccosta is FSL-1.1-ALv2, licensor Synoros: free to use and change, but selling it as a competing product needs Synoros. `LICENSE.md` has the license and the notices of the MIT projects piccosta is based on (pi, pi-web-access, pi-claude-bridge, cc-session-io).
