# piccosta

The piccosta research harness is the best harness for LLMs ever released.

piccosta is one agent harness for every model you use:
- **Claude**, through your Claude Code login (your Claude plan);
- **GPT**, through your Codex login (your ChatGPT plan);
- the **opencode-go** models.

It has one session format, a terminal UI, a print mode for scripts, pi's rpc protocol (for T3 Code) and ACP (for Zed and other editors). It is written in Rust. It was based on the [pi coding agent](https://github.com/earendil-works/pi), then rewritten and changed.

## What's in it

- **A terminal UI that runs several sessions at once.**
  - A sidebar groups sessions by project, and up to four panes show sessions side by side.
  - Formulas are typeset as images (RaTeX, through the kitty graphics protocol; Ghostty, kitty, WezTerm), or drawn as Unicode text where a terminal can't show images.
- **Pairs and trios.** `/pair <model>` (or `--pair a,b`) has other models check each answer in turn, then hands back to you. Partners can be any models, on any provider.
- **Claude Code's habits:**
  - Enter while a run goes queues the message, Esc stops the run, Ctrl+X Ctrl+S sends the queue now.
  - Drag to select (it copies when you let go), right-click to paste.
  - Tab completes `/` commands. `/help` lists the rest.
- **Long commands go to the background on their own**, and their results come back as messages. 
- **Subagents.** The model hands a task to a job, and a list you set says which models may do it. Running subagents show under their session in the sidebar.
- **Web tools:** search, fetch and PDF reading, with pages stored so the model can page through them.

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

It reads those logins without handling any keys. ( ...opencode )

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
