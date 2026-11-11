# piccosta

One agent harness for every model you use, in Rust: Claude, GPT and opencode-go models, ACP agents and your own endpoints, with one session format, a terminal UI, print mode, pi's rpc protocol (T3 Code) and ACP (Zed and other editors). It stores no credentials of its own: it reads each account's existing login.

*To install, see below, or paste this to your coding agent:*

*Install piccosta (https://github.com/Synoros-io/piccosta) on this machine by following its README: put the release binary on my PATH, or build it if there's no release for this platform. Run `piccosta auth status`, and tell me how to log in to any account it doesn't find. Set `defaultProvider` and `defaultModel` in piccosta's settings.json to a model I'm logged in to. If I use Zed or T3 Code, set them up with `piccosta zed` or `piccosta t3`. Finish with `piccosta -p "say hello"` to show it works.*

## Features

### Models

- Claude through your Claude Code login: piccosta's own prompt and tools, kept open between turns so the prompt cache carries; session records are appended per turn, not rewritten.
- GPT through your Codex login, over the Responses API, with a websocket that falls back to HTTP streaming.
- opencode-go models over OpenAI chat completions, OpenAI Responses, Anthropic Messages and Google generateContent; thinking carries across turns.
- ACP agents as models: a provider with `"api": "acp"` and a `command` in `models.json` (Antigravity through `agy acp`); `interactiveOnly` keeps one to user-picked interactive sessions.
- Your own endpoints: `/models add` or `piccosta models add <url>` (see [Your own models](#your-own-models)).
- New models show up on their own: each account's model list is fetched at start and with `/models refresh`; per-model overrides go in `models.json` (`modelOverrides`, pi's format).
- `"serviceTier": "priority"` asks a Codex model for its fast mode.
- One thinking scale, `off` to `max`, mapped to what each model takes.

### Model scale

- Kinds of work: critical, general, review, reading, bugs, search, tests, visual, prose, monitoring, exploration; each has its models, tools and instructions.
- Plan usage (Claude, Codex, opencode Go) is tracked and shown in the pane header and `/usage`; past `lowUsagePercent` (80) of a window, picks prefer another plan.
- `/scale` shows it, `/scale edit` edits `scale.json`, `/scale <kind> <model[:level]>` sets one kind; `--use <kind>` starts a session on its pick.

### Managers

- The assistant: a standing background session (`piccosta assistant`, or `"manager": {"autostart": true}`); talk to it in the sidebar or with `piccosta tell <text>`.
- Project managers: the first session in a directory becomes its manager; `/manager` makes one elsewhere.
- They list, read, start, message and stop sessions in any project, on any model or kind of work, check costs, keep schedules and send desktop notifications.
- Sessions they start are real sessions that keep running when the manager restarts; the manager hears the result and reports.
- Check-in timers: a session sets a message to itself (`in` or `at`, optional `every`) instead of polling; they survive restarts and are delivered to idle sessions too.
- `/timer <interval> <prompt>`: your own timer on the shown session (`30m`, `2h`, `5s`); it fires only when the session is idle and sends your prompt word for word with when it was set and the latest known plan usage; skipped while the session is busy or not running; agents can't change it.
- `/timer` lists your timers; `/timer off [id]` cancels.
- Managers check in on working sessions every `manager.checkMinutes` (30) and remind about idle ones after `manager.inactiveMinutes`.
- Slow commands always go to the background.
- Each manager has an append-only ledger (older halves archived monthly, `ledger search`); the assistant also keeps a short memory (`ledger remember`, `forget`).
- Managers are told to keep to their project and to ask before spending your Claude or Codex plan; these are instructions to the model, not enforced limits.
- `/advisor`: a session that advises the pane's manager (or the project's) on direction and research; you can talk to both.

### Pairs and trios

- `/pair <model>` (`--pair`): two models go back and forth in one chat until one ends with `[done]` or `pairRounds` (3) is reached.
- `/trio <model>` (`--trio`): the other model checks the answer once, then waits for you.
- `/pair off` or `/trio off` returns to one model. Works across providers, in print mode and in manager sessions.

### Subagents

- The model hands a task to a kind of work on the scale; it runs in a piccosta of its own with its own saved session.
- A model without a login is skipped; a failed run falls to the next model.
- Subagents show under their parent in the sidebar, where you can follow them live.
- Add your own kinds in `scale.json`.

### Terminal UI

- Sidebar of sessions grouped by project, managers first; up to four panes side by side.
- Ctrl+N starts a session in any directory with flags (`myproject --use critical --pair <model> --name "lane A"`).
- Pane contents, scroll, unsent input and open sessions are restored by `piccosta -c`; `/restart` relaunches on a fresh binary with everything as it was.
- Session list searched with Ctrl+S, `/resume all`, `-r <id or name>`, `-c`; Page Up/Down walks it.
- A session index makes manager, ACP, TUI and usage/cache listings fast on large histories.
- A session runs in one process; others follow it live and send input to it, including `-r`/`--session` on a session live in another piccosta.
- `piccosta tui` lists running terminal UIs; `piccosta open <id or name>` opens a session in one; `piccosta restart` relaunches one on a fresh binary.
- `--single`: the terminal UI on one session, no sidebar.
- `/fast on|off`: fast mode on Codex and Claude from the next request.
- Delete twice moves a session to the trash; idle sessions close themselves.
- Enter queues a message, Esc stops the run, Ctrl+X Ctrl+S sends the queue, ↑ takes it back, Ctrl+C stops/clears/quits.
- Editor: cross-session history, Tab completion, word moves and deletes, undo.
- Mouse: drag copies, double-click selects a path, triple-click a line, right-click pastes; over ssh through the terminal.
- Typeset math: formulas as images in Ghostty, kitty and WezTerm (including through tmux), Unicode elsewhere.
- Commands: `/model`, `/effort`, `/compact`, `/autocompact`, `/new`, `/clear`, `/fork`, `/rewind`, `/rename`, `/close`, `/delete`, `/status`, `/context`, `/usage`, `/cost`, `/scale`, `/diff`, `/copy`, `/export`, `/init`, `/review`, `/agents`, `/ps`, `/stop`; `/help` lists the rest.

### Agent loop

- Commands running past two seconds (three in a manager) move to the background and report back as a message (`PICCOSTA_LONGCMD_SOFT_S`).
- Stops a response that keeps repeating itself, and the same tool call with the same result five times in a row.
- `--tools a,b` or `-xt a,b` includes or excludes tools; `--no-tools` for none; `--no-session` keeps the session in memory only.
- `--append-system-prompt <text or file>` (repeatable) adds to the system prompt; `SYSTEM.md` in the agent dir replaces the base prompt and `APPEND_SYSTEM.md` adds to it.
- `piccosta cache [hours]` reports prompt-cache misses across sessions.
- A watcher stops commands piccosta started if it is killed.
- Retries overloads and dropped connections with backoff; used-up limits fail at once.
- Compaction summarizes older turns and keeps everything you typed word for word.
- Idle sessions compact before their prompt cache expires (`compaction.idle`, `compaction.idleMinTokens`, `compaction.cacheSeconds`).
- Failed edits say where the closest match is; `read` and `bash` give "did you mean" for missing paths.
- Optional working-rules constitution: `PICCOSTA_CONSTITUTION_FILE=built-in` or a path, or `PICCOSTA_CONSTITUTION` for the text; the `constitution` setting (`true` or a path) does the same.
- `AGENTS.md` and `CLAUDE.md` load from the project and the folders above it.

### Web tools

- Search through your own SearXNG, or a model on an OpenAI-Responses account (GPT through Codex); no search API key.
- Pages as Markdown (Readability), PDFs as text, GitHub URLs as shallow clones.
- Long results are stored so the model can page through them.

### Plugins

- Executables in `<agent dir>/plugins/` describe their tools once per session and share one process per plugin per session; a failing plugin doesn't stop the session.
- `plugins.callTimeoutSeconds` sets the call deadline (120, at most 3600).
- Optional Linux computer-use plugin in `plugins/computer`.

### Editors and scripts

- `piccosta -p` answers and exits and reads stdin; `--mode json` prints one event per line; failures exit non-zero.
- `--mode rpc`: pi's protocol, for T3 Code, with approvals in supervised mode.
- `--mode acp`: Agent Client Protocol for Zed and others, with Ask, Edits and Auto modes, diff permission cards, live command output and model and thinking pickers.
- Sessions are pi's JSONL format, so every front end opens the same sessions.

## Install

Linux (any distribution): one static x86-64 binary, tried on Arch, Alpine 3.24, Debian 12 and Ubuntu 24.04. It needs the `ca-certificates` bundle. Read-only scale kinds run commands in a Landlock sandbox (Linux 6.12 or newer, else set `"readOnly": false` on the kind in `scale.json`).

```sh
mkdir -p ~/.local/bin
curl -fL https://github.com/Synoros-io/piccosta/releases/latest/download/piccosta-x86_64-linux -o ~/.local/bin/piccosta
chmod +x ~/.local/bin/piccosta
```

Build from source (stable Rust; musl-gcc for the TLS library: `musl` on Arch, `musl-tools` on Debian and Ubuntu):

```sh
git clone https://github.com/Synoros-io/piccosta && cd piccosta
rustup target add x86_64-unknown-linux-musl
cargo build --profile dist --target x86_64-unknown-linux-musl
ln -s "$PWD/target/x86_64-unknown-linux-musl/dist/piccosta" ~/.local/bin/piccosta
```

`cargo build --release` builds faster against your own glibc, and runs only on a glibc at least as new.

macOS and native Windows build from source with `cargo build` and have not been run on a real machine. They need CMake and a C compiler (Windows: MSVC Build Tools, and NASM or `AWS_LC_SYS_PREBUILT_NASM=1`; macOS: Xcode Command Line Tools). Windows commands need Git for Windows (or `shellPath` in settings.json). Read-only kinds run without the Landlock guard there; plugins on macOS contain by process group only.

Log in with each account's own tool and check that piccosta finds the logins:

```sh
claude auth login        # for claude-bridge/* models
codex login              # for openai-codex/* models
opencode auth login      # or set OPENCODE_API_KEY, for opencode-go/* models
piccosta auth status
```

Then set `defaultProvider` and `defaultModel` in `settings.json` (see Settings), or pick with `/model`. Codex's token is read from `~/.codex/auth.json` (`$CODEX_HOME`) and opencode's key from its key file or `OPENCODE_API_KEY`, both read-only; a Codex login already in piccosta's `login/auth.json` is refreshed in place.

## Use

```sh
piccosta                                     # the terminal UI, on your default model
piccosta --model claude-bridge/claude-opus-5-5 --thinking xhigh
piccosta -c                                  # this directory's most recent session
piccosta -r                                  # pick a session to resume (or -r <id or name>)
piccosta -p "summarize README.md"            # answer and exit; stdin is read too
piccosta -p --mode json "..."                # one event per line, for scripts
```

`piccosta --help` lists every flag; `/help` lists every key and command. Add `--yolo` to skip permissions.

Inside tmux, so images, Shift+Enter and focus changes get through, add to `~/.tmux.conf`:

```
set -g allow-passthrough on
set -g extended-keys on
set -g extended-keys-format csi-u
set -g focus-events on
```

## In an editor or GUI

```sh
piccosta zed [dir]    # opens Zed with piccosta as its agent, adding it to Zed's settings first
piccosta t3 [dir]     # starts T3 Code with its Pi provider running piccosta (turn the Pi provider on in T3's settings the first time)
```

If either isn't on your PATH: `"gui": {"zed": "/path/to/zed", "t3": "/path/to/t3"}` in `settings.json`.

By hand, Zed's `settings.json`:

```json
"agent_servers": {
  "piccosta": { "type": "custom", "command": "/path/to/piccosta", "args": ["--mode", "acp"] }
}
```

T3 Code: add a pi provider and set its binary path to `piccosta`.

## Your own models

```sh
piccosta models add <base-url> --name myserver
piccosta models add <base-url> --name myserver --key-env MYSERVER_API_KEY
your-key-command | piccosta models add <base-url> --key-stdin
piccosta models remove myserver
```

`/models add` and `/models remove <name>` do the same in the terminal UI. The CLI also takes `--key-command CMD` and `--api`. A pasted or piped key goes in `login/auth.json` (mode 0600), bound to that URL's origin; `models.json` refers to it as `"apiKey": "@stored"`.

`models.json` in the agent dir:

```json
{
  "providers": {
    "myserver": {
      "baseUrl": "https://your-host/v1",
      "api": "openai-completions",
      "apiKey": "$MYSERVER_API_KEY"
    }
  }
}
```

- `api`: `openai-completions`, `openai-responses`, `anthropic-messages`, `google-generative-ai`, or `acp`.
- `apiKey`: `@stored`, `$VAR`, or `!command` (15-second deadline); omit it for a server with no key.
- `models`: optional list, `[{"id": "my-model", "contextWindow": 128000, "maxTokens": 8192}]`; without it, `<baseUrl>/models` is used.
- ACP agent: `{"api": "acp", "command": ["agy", "acp"], "interactiveOnly": true}`; optional `configOptions`, `modelFilter`, `sessionMeta`.

Models appear as `myserver/<id>`: `/model`, or `piccosta -m myserver/my-model`.

## Settings

Files live in `$PICCOSTA_AGENT_DIR`, else `$XDG_DATA_HOME/piccosta/agent`, else `~/.local/share/piccosta/agent` (`%APPDATA%/piccosta/agent` on Windows): `settings.json`, `models.json`, `scale.json`, sessions and the fetched model lists. macOS sockets use `$PICCOSTA_RUNTIME_DIR` (default `$XDG_DATA_HOME/piccosta/run`).

`settings.json` (pi's keys): `defaultProvider`, `defaultModel`, `defaultThinkingLevel`, `modelThinkingLevels`, `compaction`, `retry`, `shellCommandPrefix`, `shellPath`, `transport`; piccosta's own `manager` and `gui`. A project's `.piccosta/settings.json` is merged over them, except:

- `models.freshSeconds` (900): don't refetch a model list fetched more recently.
- `models.timeoutSeconds` (10): time allowed for one list.
- `models.codexClientVersion`: the Codex version to ask Codex's list as.
- `auth.commandTimeoutSeconds` (15): bound for `!command` key lookups.
- `web.fetch.maxSizeMB` (5): largest page fetched, PDFs aside.
- `usage.refreshSeconds` (300): how often plan usage is polled.
- `models.json` `compat.reasoningReplayField`: the field a model's thinking is sent back in.
- `clientId` and `tokenUrl` in the `openai-codex` entry of `login/auth.json`: replace OpenAI's refresh client and endpoint.

## License

piccosta is FSL-1.1-ALv2, licensor Synoros. `LICENSE.md` has the license and the notices of the MIT projects piccosta is based on (pi, pi-web-access, pi-claude-bridge, cc-session-io).
