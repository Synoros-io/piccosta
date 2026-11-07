# piccosta

The piccosta research harness is a new standard for LLM harnesses as of November 2026.
Created to make coding agents more reliable in research, turned into a highly performant rust based harness that outperforms anything released to the public, for multi agent management, for cost and tool reliability, for customization, for system resource usage, and with many modern features. It is an active area of research.

piccosta is one agent harness for every model you use:
- Claude, through your Claude Code login (your Claude plan), within tos via the agent sdk as of OCT 2026;
- GPT, through your Codex login (your ChatGPT plan);
- the opencode and opencode go models
- BYO or connect

It stores no credentials of its own: it uses each account's existing login, and reads those files without writing them. The one exception is a login already in piccosta's own `login/auth.json` (the TypeScript piccosta kept them there), which it reads first, refreshing a Codex login there and saving it back. It has one session format, a terminal UI, a print mode for scripts, pi's rpc protocol (for T3 Code) and ACP (for Zed and other editors). The idea is to allow the user to customize the frontend with any pre-existing one, or with a custom one their agents build, on top of the existing minimal tui or otherwise. It is written in Rust, initially based on the [pi coding agent](https://github.com/earendil-works/pi) and then tested, substantially improved, rewritten and repackaged.

*To install see below, or paste this to your coding agent:*

*Install piccosta (https://github.com/Synoros-io/piccosta) on this machine by following its README: put the release binary on my PATH, or build it if there's no release for this platform. Run `piccosta auth status`, and tell me how to log in to any account it doesn't find. Set `defaultProvider` and `defaultModel` in piccosta's settings.json to a model I'm logged in to. If I use Zed or T3 Code, set them up with `piccosta zed` or `piccosta t3`. Finish with `piccosta -p "say hello"` to show it works.*

## Features

- **Claude through Claude Code.** A Claude plan only works inside Claude Code, so piccosta runs Claude Code as a pipe. piccosta's own prompt goes in, with your AGENTS.md and CLAUDE.md files in it, and piccosta's own tools; Claude Code adds none of its builtins or settings, and doesn't load CLAUDE.md a second time. When a turn ends on tool calls, the query stays open for the results, so the prompt cache carries from turn to turn. Rate-limit messages name the limit and when it resets.
- **GPT through your Codex login**, over the Responses API. Its websocket sends only what's new each turn, and falls back to plain HTTP streaming when it can't be used.
- **The opencode-go models**, through four request formats: OpenAI chat completions, OpenAI Responses, Anthropic Messages and Google's generateContent. Thinking carries across turns, Gemini's signed thoughts included.
- More will be added with time. It is strictly against antigravity's TOS, so I can't release a version doing that.
- **Your own models:** `/models add` or `piccosta models add <url>` adds an endpoint without writing JSON (see [Your own models](#your-own-models)).
- **New models show up on their own.** piccosta offers what each connected account serves: Codex's own list for your ChatGPT login, the models Claude Code offers your Claude login, opencode-go's `/models`, and `<baseUrl>/models` for a `models.json` provider that names no models. The lists are fetched as piccosta starts and with `/models refresh`; one that can't be fetched keeps the last one, and an account you aren't logged in to offers nothing. What you set for a model goes over what its provider reports, in `models.json`'s `providers.<provider>.modelOverrides.<id>` (pi's format), e.g. `{"contextWindow": 200000, "maxTokens": 8000, "input": ["text"], "thinkingLevelMap": {"medium": "medium"}}`; an object such as `thinkingLevelMap` or `compat` changes only the keys you name. `"serviceTier": "priority"` asks a Codex model for its fast mode (about twice as fast, counted higher against your plan).
- **One thinking scale for all of them**, from `off` to `max`, mapped to what each model actually takes. A level a model doesn't take runs as its nearest, and piccosta says so.

### Model scale

Sliding scale : critical, general, review, reading code, bug hunting, web search, tests, visual work, writing prose, monitoring and exploration. 
- piccosta records how much of your Claude, Codex and opencode Go plans is used: as the plans report it with each response, and by asking each one, as its own tools do, when piccosta starts, every 5 minutes (`usage.refreshSeconds` in settings) and after a turn ends. The pane header shows the session's plan (`· 5h 19% · 7d 66%`), and `/usage` every plan with when each window resets. Once one passes 80% of a window (`lowUsagePercent` in `scale.json`), the scale's next picks prefer another plan's model. A session already running stays on its model.
- The defaults ship with piccosta. Your changes go in one file, `scale.json` in piccosta's folder: `/scale edit` writes all of it there and opens it in your editor, and `/scale <kind> <model[:level]> …` sets one kind's models. A kind you change keeps the rest of its defaults; `"off": true` turns one off.
- `/scale` shows it all with the plans' usage. `--use <kind>` starts a session on its pick, on the command line, in `/new` and in Ctrl+N's box.

### Managers

- **The assistant** is a standing session that runs as a background service, inspired by Grokbot, OPENAI dots, and meta muse. Start it with `piccosta assistant`, or set `"manager": {"autostart": true}` in settings and piccosta writes and starts its systemd user unit when the terminal UI opens (a plain background process where there's no systemd). It heads the sidebar; you talk to it there, or with `piccosta tell <text>` from any shell. Name it with `/rename` on its row, or by telling it its name.
- **Project managers.** The first session ever opened in a directory becomes that project's manager, `/manager` makes one anywhere else, and the assistant can make one for you. It looks after its project's sessions only, like a scoped version of an assistant.
- **What managers can do:** list projects and sessions; read, start, message and stop sessions in any project, on any model, by kind of work from the scale, with partners; see what sessions cost; keep schedules; notify you on the desktop.
- **They report back.** A session a manager starts is a real session, listed under its project. Given work, it runs in a process of its own that carries on when the manager restarts, and when its turn ends the manager hears what it said, reads it and tells you what happened. Without work it waits for you. While any of them are at work, it gets a check-in every half hour (`manager.checkMinutes` in settings, 0 for none), once its own turn has ended, with how long since each one last replied and the last line it wrote; one not at work that goes quiet for half an hour (`manager.inactiveMinutes`) gets it a reminder to close it or update it. A manager, or a session's run of its own, tries a failed model request again up to six times rather than three, about two minutes in all, before its turn ends in the error.
- **They never make you wait.** Anything a manager runs that takes more than a few seconds goes to the background.
- **A ledger that keeps everything.** Each manager has an append-only ledger. Past 256 KB its older half moves to a monthly archive, and nothing is deleted. When a manager compacts, it keeps your newest words and its ledger's newest entries word for word, and `ledger search` reaches all the rest. The ledger records what's happening, not facts about you. The assistant also keeps a short memory of what should hold from day to day (`ledger remember` and `forget`), all of it in its prompt.
- **Spending:** a manager asks before it starts work on your Claude or Codex plan on its own initiative, unless you've told it it needn't; work you asked for is your go-ahead.

### Pairs and trios

Like Devin Fusion, piccosta can run two models together in one chat. A pair is two models going back and forth. A trio is the same two models with you in the loop.

- `/pair <model>` (or `--pair <model>`): your session's model answers, the other model checks it, fixes it or builds on it, and they carry on in the one conversation you read, without waiting for you. It stops when a model ends its reply with a line saying only `[done]` (counted from the second model's first reply), or after 3 rounds. `pairRounds` in settings.json changes the limit. Then it's your turn.
- `/trio <model>` (or `--trio <model>`): the session's model answers, the other checks that answer once, says what it changed or that it holds up, and waits for you.
- `/pair off` or `/trio off` goes back to working alone.

Each model is told who the other is and how the exchange ends, and each turn is labelled with the model that wrote it. Any two models work together, across providers. The setup is saved with the session, and works in print mode and in sessions a manager starts (with `pair` or `trio`). Older sessions that had a partner open as trios.

### Subagents

The model hands a task to a kind of work on the scale (critical, general, review, reading, bugs, search, tests, visual, prose, monitoring, exploration), which runs it on its models with its own tools and instructions; a model without a login is skipped, and if a run fails, the next model takes over. Each subagent is a piccosta of its own, with its own saved session. While it works, it shows under its parent session in the sidebar, where you can open it and follow it live. Each kind's tools and instructions are in the scale's file with its models (`/scale edit`); add your own kinds there.

### A terminal UI for many sessions at once

- **Projects and panes.** The sidebar groups sessions by project, with each project's manager first. Up to four panes run side by side.
- **Starting a session anywhere.** Ctrl+N lists this directory, your home folder (`~`, outside every project), your recent projects and the git repositories up to three folders under home (hidden and build folders skipped, and none inside another repository). Type to narrow it, or type any path, with flags after it: `myproject --use critical --pair openai-codex/gpt-6.1-sol --name "lane A"`.
- **Coming back to where you were.** The terminal UI keeps what each pane shows, how far it's scrolled, what's typed and not sent, and the sessions open besides. `piccosta -c` brings it all back; `/restart` starts piccosta again on a freshly built binary, everything as it was.
- **Finding sessions again:**
  - Page Up/Down (or Alt+↑↓) walks the list, showing earlier sessions without starting them until you type.
  - Ctrl+S searches the list, and `/resume all` reaches every project's sessions.
  - `-r` resumes by id, the start of one, or a name; `-c` resumes this directory's latest.
- **Sessions are never run twice.** The piccosta running a session answers on a socket beside its file; another one shows that session live from its file, and what you type there goes to the one running it, which reads it at its next step. A message for a session nothing runs waits in its file for the next one to open it.
- **Cleaning up.** Delete twice moves the shown session to the trash, where a file manager can restore it. If it can't go to the trash (it's on another disk, say), the session stays where it is and piccosta says so. A session you leave that's idle closes on its own, and an empty one goes away.
- **The keys Claude Code taught you:**
  - Enter queues a message while a run goes, and Esc stops the run.
  - Ctrl+X Ctrl+S sends the queue now. Running commands move to the background so the model reads it straight away.
  - ↑ takes the queue back. Ctrl+C stops a run, then clears the input, then quits on a second press.
- **An editor that behaves:** history across sessions, Tab completion of commands, word moves and deletes (Alt+B/F/D, Ctrl+Backspace), Ctrl+Shift+Delete to clear, and Ctrl+_ to undo.
- **Copy and paste with the mouse.** Drag to select and it copies when you let go. Double-click selects a path, triple-click a line, and right-click pastes. Shift+arrows select in the editor. Copies go to your desktop's clipboard, or through the terminal over ssh.
- **Typeset math.** RaTeX typesets formulas and shows them as images in Ghostty, kitty and WezTerm (through tmux too). Terminals that can't show images get Unicode.
- **Commands:**
  - `/model`, `/effort` (`auto` returns to the default), `/compact` (it says what it saved), `/autocompact`.
  - `/new`, `/clear`, `/fork`, `/rewind`, `/rename`, `/close`, `/delete`.
  - `/status`, `/context`, `/usage`, `/cost`, `/scale`.
  - `/diff` (new files included), `/copy`, `/export`, `/init`, `/review`, `/agents`, `/ps`, `/stop`.
  - `/help` lists the rest, and Tab completes them.

### An agent loop that doesn't get stuck

- **Long commands go to the background.** A command still running after two seconds (three in a manager) moves there by itself (`PICCOSTA_LONGCMD_SOFT_S` sets the seconds, and 0 sends every command there), and its result comes back as a message when it ends. A timeout only ever stops a command; it never holds the session.
- **Loop guards.** piccosta stops a response that keeps repeating itself, and a run that makes the same tool call five times in a row.
- **Nothing left running.** If piccosta is killed outright, as an editor stops an agent, a watcher process stops the commands it started.
- **Retries.** Overloads and dropped connections are retried with backoff. Used-up limits and withdrawn models fail straight away.
- **Compaction keeps your words.** When the context fills, older turns are summarized, and everything you typed before the cut stays word for word on top of the summary.
- **Failed edits say why.** A failed edit tells the model where the closest match is, or what changed since it last read the file. A missing path gets "did you mean" suggestions.
- **A working prompt.** piccosta ships a constitution with working rules: research before building, verify by running, nothing hardcoded, clean up after yourself. It's off unless you turn it on: `PICCOSTA_CONSTITUTION_FILE=built-in` for the one piccosta ships, `PICCOSTA_CONSTITUTION_FILE=<path>` for your own file, or `PICCOSTA_CONSTITUTION` holding the text itself. Your `AGENTS.md` and `CLAUDE.md` files load from the project and the folders above it.

### Web tools

- **Search** goes through your own SearXNG, or through a model on an account you already have that speaks the OpenAI Responses API (GPT through your Codex login, for one), so it needs no search API key. Claude models can't search for it, so with only a Claude login, use SearXNG.
- **Pages** come back as Markdown, read with Readability. PDFs come back as text. GitHub URLs become shallow clones the model can read.
- **Long results** are stored, so the model can page through them later instead of fetching again.

### Plugins

Executables in `<agent dir>/plugins/` describe their tools once when a session opens. Calls share
one process per plugin per session; failures name the plugin and leave the session running.
`plugins.callTimeoutSeconds` sets the call deadline (120 seconds by default, capped at 3600).
Linux uses watcher-owned cgroups when the user's systemd cgroup is delegated, otherwise a child
subreaper; without either delegation or usable pidfd signalling, plugins refuse to start.
Windows retains the plugin's Job Object. On macOS, containment is **process-group only**:
helpers that call `setsid` can outlive the session; a notice says so on the first plugin start.

### Editors and scripts

- **Print mode.** `piccosta -p` answers and exits, and reads stdin too. `--mode json` prints one event per line. Failures exit non-zero.
- **T3 Code.** `--mode rpc` is pi's protocol, including approvals in T3's supervised mode.
- **Zed and other editors.** `--mode acp` is the Agent Client Protocol. It has Ask, Edits and Auto modes, permission cards that show each edit's diff, and live command output in the editor's terminal. Zed's pickers set the model and thinking level.
- **One session format.** Sessions are saved in pi's JSONL format, so the terminal, Zed, T3 Code and print mode all open the same sessions.

## Performance

Measured on piccosta 0.2.0.

| | piccosta 0.2.0 | pi 1.2.0 | Claude Code 2.1.316 |
|---|---|---|---|
| Whole 20-step run, wall time | 150 ms (140–170) | 1,240 ms (1,230–1,250) | 2,080 ms (2,050–2,190) |
| CPU, user | 50 ms (40–60) | 440 ms (420–450) | 1,390 ms (1,360–1,420) |
| CPU, system | 90 ms (80–110) | 140 ms (130–160) | 370 ms (350–370) |
| Peak memory, largest process | 17.2 MB (15.5–17.5) | 130 MB (129–131) | 237 MB (223–237) |
| Peak memory, all its processes | 26.6 MB (22.7–27.0) | 128 MB (128–130) | 250 MB (248–250) |
| Start to first model request | 10.9 ms (10.2–12.7) | 249 ms (239–258) | 284 ms (277–321) |
| Between a response and the next request (runs the tool) | 6.6 ms (6.0–7.4) | 47.8 ms (47.7–48.2) | 67.9 ms (67.3–69.8) |
| Last response to exit | 53 ms (33–61) | 76 ms (58–86) | 98 ms (72–139) |
| `--version` | 2.3 ms | 176 ms | 8.6 ms |
| Terminal UI, idle: memory | 12.2 MB | 129 MB | 148 MB |
| Terminal UI, idle: CPU | 0.3 ms per second | 0.0 ms per second | 3.3 ms per second |
| On disk | one 32 MB static binary | 154 MB, plus Node.js | 314 MB |

- **What each harness was:** each ran in a fresh config folder, so none read the user's own settings. piccosta was the static release binary.
- **How CPU and memory were measured:** CPU counts each harness and the commands it ran. Peak memory comes from GNU time (largest process) and from sampling /proc every 50 ms (all its processes).
- **Idle numbers:** taken over 30 seconds, after a 5-second settle, with nothing typed. Claude Code's idle figure is on its first-run welcome screen.
- **Not measured:** the Codex CLI.

## Install

piccosta runs on Linux, any distribution. A release is one static binary with no libraries to
match, tried on Arch, Alpine 3.24, Debian 12 and Ubuntu 24.04. It needs a CA certificate bundle
(the `ca-certificates` package) to reach the models. The scale's read-only kinds (review, reading,
search and exploration) run their commands in a Landlock sandbox, which needs Linux 6.12 or newer
(Landlock ABI 6); on an older kernel their commands don't run, unless the kind has
`"readOnly": false` in `scale.json`.

macOS and native Windows can be built from source with stable Rust (`cargo build`, then put
`target/debug/piccosta` or `target/debug/piccosta.exe` on PATH). On Windows, install the MSVC
Build Tools with the C++ workload; on macOS, install Xcode Command Line Tools. The TLS library
also needs CMake and a C compiler; MSVC builds need NASM, or
`AWS_LC_SYS_PREBUILT_NASM=1` to use AWS-LC's packaged assembly. These platforms have not yet
been run on a real machine.
Windows commands need Git for Windows, including Git Bash: piccosta searches Git's usual
install directories and PATH, or uses `shellPath` from settings.json. No PowerShell translation.
On macOS and Windows, read-only kinds run **without** the Landlock accident guard and print
one warning per process. TeX uses the existing text fallback when graphics are unavailable.
Windows uses Job Objects to stop command trees and local, owner-only named pipes for runtimes.
Windows `/restart` closes running commands rather than carrying them into the replacement.

The Linux release binary is for x86-64. To install it:

```sh
mkdir -p ~/.local/bin
curl -fL https://github.com/Synoros-io/piccosta/releases/latest/download/piccosta-x86_64-linux -o ~/.local/bin/piccosta
chmod +x ~/.local/bin/piccosta   # or put it anywhere on your PATH
```

To build it you need a recent stable Rust, and musl-gcc (the `musl`
package on Arch, `musl-tools` on Debian and Ubuntu) for the TLS library's C code. This builds
the x86-64 binary; on other architectures, use `cargo build --release` below:

```sh
git clone https://github.com/Synoros-io/piccosta && cd piccosta
rustup target add x86_64-unknown-linux-musl
cargo build --profile dist --target x86_64-unknown-linux-musl
ln -s "$PWD/target/x86_64-unknown-linux-musl/dist/piccosta" ~/.local/bin/piccosta   # or anywhere on your PATH
```

`cargo build --release` builds faster against your own glibc, and runs only on a glibc at least
as new.

Log in with each account's own tool, check that piccosta finds the logins, and set the model it starts on (`defaultProvider` and `defaultModel` in `settings.json` in piccosta's folder, `~/.local/share/piccosta/agent` unless Settings below says otherwise, or `/model` in the terminal UI):

```sh
claude auth login        # Claude Code (claude.ai/code), for claude-bridge/* models
codex login              # the Codex CLI, for openai-codex/* models
opencode auth login      # or set OPENCODE_API_KEY, for opencode-go/* models
piccosta auth status
```

Claude's login stays inside Claude Code, and piccosta never sees it. For Codex, piccosta reads the token the Codex CLI keeps in `~/.codex/auth.json` (`$CODEX_HOME/auth.json` when that's set), read-only, and sends it only to OpenAI. For opencode-go, it reads the key from opencode's own key file or from `OPENCODE_API_KEY`, and sends it only to opencode. piccosta stores no keys, apart from refreshing a Codex login already in its own `login/auth.json`, and never prints one.

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
- `pgup`/`pgdn` (or `alt+↑↓`) moves between sessions, and `alt+pgup`/`alt+pgdn` or the mouse wheel scrolls;
- `^s` finds one;
- Delete twice moves the shown session to the trash, or leaves it and says so when it can't go there;
- `/help` lists every key and command.

Inside tmux, add these to `~/.tmux.conf` so images, Shift+Enter and focus changes get through:

```
set -g allow-passthrough on
set -g extended-keys on
set -g extended-keys-format csi-u
set -g focus-events on
```

## In an editor or GUI

`piccosta zed [dir]` opens Zed on a directory with piccosta as its agent, adding piccosta to Zed's settings first if it isn't there (a copy of the file as it was is kept). `piccosta t3 [dir]` starts T3 Code with its Pi provider running piccosta, telemetry off unless your environment already sets `T3CODE_TELEMETRY_ENABLED` or `DO_NOT_TRACK`; the first time, turn the Pi provider on in T3's settings. If either isn't on your PATH, `"gui": {"zed": "/path/to/zed", "t3": "/path/to/t3"}` in piccosta's `settings.json` says where it is.

By hand, **Zed**, in `settings.json`:

```json
"agent_servers": {
  "piccosta": { "type": "custom", "command": "/path/to/piccosta", "args": ["--mode", "acp"] }
}
```

Zed's model, thinking and mode pickers work. Its Ask, Edits and Auto modes decide which tool calls ask first.

**T3 Code:** add a pi provider and set its binary path to `piccosta`. T3 speaks pi's rpc mode, which piccosta implements. Its supervised mode asks before each tool call.

Add `--yolo` to skip permissions.

## Your own models

Use `/models add` in the terminal UI, or:

```sh
piccosta models add <base-url> --name myserver
piccosta models add <base-url> --name myserver --key-env MYSERVER_API_KEY
# Or pipe a key from your password manager:
your-key-command | piccosta models add <base-url> --key-stdin
piccosta models remove myserver
```

The guided UI probes the URL (also `/v1/models`), asks for a key only when needed, proposes a name, and shows the models before saving. Paste a key in its hidden input, or choose an environment variable or command. The CLI also takes `--key-command CMD` and `--api` for one of the four formats below. Without a model list, it asks for a model id. A pasted or piped key goes only in the private credential store (`login/auth.json`, mode 0600), bound to that URL's origin; requests elsewhere are refused. `models.json` refers to it with `"apiKey": "@stored"`. Other providers and overrides are kept. New models are pickable at once; an already running session's prompt and tools stay unchanged. `/models remove <name>` removes a provider in the UI too.

The JSON reference, in `models.json` in the agent dir (see Settings below):

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

- `api`: `openai-completions` for `/v1/chat/completions`; `openai-responses`, `anthropic-messages` or `google-generative-ai` for the others.
- `apiKey`: `@stored` for a key added to the credential store, `$VAR` to read it from the environment, or `!command` to use a command's output (a password manager, say). Literal keys still work, but the add flow never writes one here. Key commands have a 15-second deadline; timeout or cancellation stops their whole process group. Leave it out for a local server that takes no key.
- Without a `models` list, piccosta offers everything `<baseUrl>/models` lists. To name them yourself, or for a server with no `/models`: `"models": [{"id": "my-model", "contextWindow": 128000, "maxTokens": 8192}]`.

The models show up as `myserver/<id>`: pick one with `/model`, or `piccosta -m myserver/my-model`. `/models refresh` fetches the lists again.

## Settings

piccosta keeps its files in `$PICCOSTA_AGENT_DIR` when it's set, else `$XDG_DATA_HOME/piccosta/agent`, else `~/.local/share/piccosta/agent` on Linux and macOS (XDG layout), or `%APPDATA%/piccosta/agent` on Windows: `settings.json`, `models.json`, `scale.json`, sessions, and the fetched model lists (`models/`). `HOME` remains an override; Windows otherwise uses `USERPROFILE`. Existing `CODEX_HOME` and Claude config overrides still apply.

macOS runtime sockets use `$PICCOSTA_RUNTIME_DIR`, default `$XDG_DATA_HOME/piccosta/run` (or `~/.local/share/piccosta/run`), with hashed session names to fit Unix socket limits. Set it to a shorter, private directory if your home path is too long. Linux sockets remain beside session files; Windows named pipes need no socket files.

`settings.json` uses pi's keys:
- `defaultProvider` and `defaultModel`;
- `defaultThinkingLevel` and `modelThinkingLevels`;
- `compaction`, `retry`, `shellCommandPrefix`, `shellPath` (override the command shell) and `transport`;
- piccosta's own `manager` and `gui`.

A project's `.piccosta/settings.json` is merged over them, except the model lists' settings, which only the agent dir's file sets:
- `models.freshSeconds`: a list fetched more recently than this isn't fetched again at start (900);
- `models.timeoutSeconds`: how long one list may take (10);
- `models.codexClientVersion`: the Codex version piccosta asks Codex's list as, since the list leaves out models newer than that. Without it, piccosta asks as the Codex CLI last did, else as `codex --version` says.

`auth.commandTimeoutSeconds` in the agent dir's `settings.json` bounds every `!command` key lookup, including its output pipes (15 seconds by default; a positive whole number). Raise it for password managers that need longer to unlock, for example `{"auth": {"commandTimeoutSeconds": 30}}`.

`web.fetch.maxSizeMB` is the largest page the web tools fetch, a PDF aside (5).

In `models.json`, a model's `compat.reasoningReplayField` names the field its thinking goes back to it in, when the server streams it as `reasoning` but wants it back under another name (opencode's gateway gets `reasoning_content` already).

A ChatGPT login in piccosta's `login/auth.json` is refreshed with OpenAI's client and token endpoint; `clientId` and `tokenUrl` in its `openai-codex` entry replace them.

## License

piccosta is FSL-1.1-ALv2, licensor Synoros. You can use and change it freely, but selling it as a competing product needs Synoros. `LICENSE.md` has the license and the notices of the MIT projects piccosta is based on (pi, pi-web-access, pi-claude-bridge, cc-session-io).
