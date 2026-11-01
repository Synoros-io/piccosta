# piccosta

The piccosta research harness is the best harness for LLMs ever released, as of October 2026.

piccosta is one agent harness for every model you use:
- Claude, through your Claude Code login (your Claude plan), within tos via the agent sdk as of OCT 2026;
- GPT, through your Codex login (your ChatGPT plan);
- the opencode and opencode go models
- BYO or connect

It has one session format, a terminal UI, a print mode for scripts, pi's rpc protocol (for T3 Code) and ACP (for Zed and other editors). The idea is to allow the user to customize the frontend with any pre-existing one, or with a custom one their agents build, on top of the existing tui or otherwise. It is written in Rust, based on the [pi coding agent](https://github.com/earendil-works/pi) and then tested, rewritten and repackaged.

*To install, paste this to your coding agent:*

*Install piccosta (https://github.com/Synoros-io/piccosta) on this machine by following its README: put the release binary on my PATH, or build it if there's no release for this platform. Run `piccosta auth status`, and tell me how to log in to any account it doesn't find. Set `defaultProvider` and `defaultModel` in piccosta's settings.json to a model I'm logged in to. If I use Zed or T3 Code, set them up with `piccosta zed` or `piccosta t3`. Finish with `piccosta -p "say hello"` to show it works.*

## Features

### Every model on the plans you already pay for

- **Claude through Claude Code.** A Claude plan only works inside Claude Code, so piccosta runs Claude Code as a pipe. piccosta's own prompt goes in, with your AGENTS.md and CLAUDE.md files in it, and piccosta's own tools; Claude Code adds none of its builtins or settings, and doesn't load CLAUDE.md a second time. When a turn ends on tool calls, the query stays open for the results, so the prompt cache carries from turn to turn. Rate-limit messages name the limit and when it resets.
- **GPT through your Codex login**, over the Responses API. Its websocket sends only what's new each turn, and falls back to plain HTTP streaming when it can't be used.
- **All 28 opencode-go models**, through four request formats: OpenAI chat completions, OpenAI Responses, Anthropic Messages and Google's generateContent. Thinking carries across turns, Gemini's signed thoughts included.
- More will be added with time. It is strictly against antigravity's TOS, so I can't release a version doing that.
- **Your own models** go in `models.json`, in pi's format: any endpoint that speaks one of those four formats.
- **One thinking scale for all of them**, from `off` to `max`, mapped to what each model actually takes. A level a model doesn't take runs as its nearest, and piccosta says so.

### The model scale

Every kind of work has its models: critical, general, review, reading code, bug hunting, web search, tests, visual work, monitoring and exploration. Each has a first choice and the models that take over when a plan runs low.
- piccosta records how much of your Claude and Codex plans is used, as the plans report it with each response, and moves to the next model once one passes 80% of a window.
- The defaults ship with piccosta. `modelScale` in `settings.json` changes any of them, adds your own or turns one off, and a project can have its own.
- `/scale` shows it all with the plans' usage. `--use <kind>` starts a session on its pick, on the command line, in `/new` and in Ctrl+N's box.

### Managers

- **The assistant** is a standing session that runs as a background service, inspired by Grokbot, OPENAI dots, and meta muse. Start it with `piccosta assistant`, or set `"manager": {"autostart": true}` in settings and piccosta writes and starts its systemd user unit when the terminal UI opens (a plain background process where there's no systemd). It heads the sidebar; you talk to it there, or with `piccosta tell <text>` from any shell.
- **Project managers.** The first session ever opened in a directory becomes that project's manager, and `/manager` makes one anywhere else. It looks after its project's sessions only, like a scoped version of an assistant.
- **What managers can do:** list, read, start and message sessions in any project, on any model, by kind of work from the scale, with partners; see what sessions cost; keep schedules; notify you on the desktop.
- **They report back.** A session a manager starts runs in the background, and its end comes back to the manager, which reads the last turn before it tells you what happened. A session it started that goes quiet for half an hour gets it a reminder to close it or update it.
- **They never make you wait.** Anything a manager runs that takes more than a few seconds goes to the background.
- **A ledger that keeps everything.** Each manager has an append-only ledger. Past 256 KB its older half moves to a monthly archive, and nothing is deleted. When a manager compacts, it keeps your newest words and its ledger's newest entries word for word, and `ledger search` reaches all the rest. The ledger records what's happening, not facts about you. The assistant also keeps a short memory of what should hold from day to day (`ledger remember` and `forget`), all of it in its prompt.
- **Spending:** a manager asks before it starts sessions on your Claude or Codex plan, unless you've told it it needn't.

### Models that check each other

Like Devin Fusion, way to run any two models as a "pair", with yourself in the loop. `/pair <model>` makes a pair, and `/pair a b` (or `--pair a,b`) makes a trio. After each answer, every other model takes a turn checking it against what you asked. They verify, fix, finish, and say what they changed. Then it's your turn again, so they can't keep each other going. Any models can work together, across providers. The group is saved with the session, and works in print mode too.

### Subagents

The model hands a task to a job: search, scout, worker, reviewer or monitor. Each job runs on its kind of work from the scale; a model without a login is skipped, and if a run fails, the next model takes over. Each subagent is a piccosta of its own, with its own saved session. While it works, it shows under its parent session in the sidebar, where you can open it and follow it live. You can write your own jobs as Markdown files.

### A terminal UI for many sessions at once

- **Projects and panes.** The sidebar groups sessions by project, with each project's manager first. Up to four panes run side by side.
- **Starting a session anywhere.** Ctrl+N lists this directory, your home folder (`~`, outside every project), your recent projects and every git repository under home. Type to narrow it, or type any path, with flags after it: `myproject --use critical --pair openai-codex/gpt-6.1-sol --name "lane A"`.
- **Coming back to where you were.** The terminal UI keeps what each pane shows, how far it's scrolled, what's typed and not sent, and the sessions open besides. `piccosta -c` brings it all back; `/restart` starts piccosta again on a freshly built binary, everything as it was.
- **Finding sessions again:**
  - Alt+↑↓ walks the list, showing earlier sessions without starting them until you type.
  - Ctrl+S searches the list, and `/resume all` reaches every project's sessions.
  - `-r` resumes by id, the start of one, or a name; `-c` resumes this directory's latest.
- **Sessions are never run twice.** A running piccosta locks its session file; another one shows that session live from its file, and what you type there goes into its inbox.
- **Cleaning up.** Delete twice moves the shown session to the trash, where a file manager can restore it. A session you leave that's idle closes on its own, and an empty one goes away.
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

- **Long commands go to the background.** A command still running after a minute moves there by itself, and its result comes back as a message when it ends. A timeout only ever stops a command; it never holds the session.
- **Loop guards.** piccosta stops a response that keeps repeating itself, and a run that makes the same tool call five times in a row.
- **Nothing left running.** If piccosta is killed outright, as an editor stops an agent, a watcher process stops the commands it started.
- **Retries.** Overloads and dropped connections are retried with backoff. Used-up limits and withdrawn models fail straight away.
- **Compaction keeps your words.** When the context fills, older turns are summarized, and everything you typed before the cut stays word for word on top of the summary.
- **Failed edits say why.** A failed edit tells the model where the closest match is, or what changed since it last read the file. A missing path gets "did you mean" suggestions.
- **A working prompt.** piccosta's own system prompt carries working rules: research before building, verify by running, nothing hardcoded, clean up after yourself. Your `AGENTS.md` and `CLAUDE.md` files load from the project and the folders above it.

### Web tools

- **Search** goes through your own SearXNG, or through a search-capable model on an account you already have, so it needs no search API key.
- **Pages** come back as Markdown, read with Readability. PDFs come back as text. GitHub URLs become shallow clones the model can read.
- **Long results** are stored, so the model can page through them later instead of fetching again.

### Editors and scripts

- **Print mode.** `piccosta -p` answers and exits, and reads stdin too. `--mode json` prints one event per line. Failures exit non-zero.
- **T3 Code.** `--mode rpc` is pi's protocol, including approvals in T3's supervised mode.
- **Zed and other editors.** `--mode acp` is the Agent Client Protocol. It has Ask, Edits and Auto modes, permission cards that show each edit's diff, and live command output in the editor's terminal. Zed's pickers set the model and thinking level.
- **One session format.** Sessions are saved in pi's JSONL format, so the terminal, Zed, T3 Code and print mode all open the same sessions.

## Performance

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
(the `ca-certificates` package) to reach the models.

To install the release binary:

```sh
mkdir -p ~/.local/bin
curl -fL https://github.com/Synoros-io/piccosta/releases/download/v0.10.0/piccosta-x86_64-linux -o ~/.local/bin/piccosta
chmod +x ~/.local/bin/piccosta   # or put it anywhere on your PATH
```

To build it you need a recent stable Rust, and musl-gcc (the `musl`
package on Arch, `musl-tools` on Debian and Ubuntu) for the TLS library's C code:

```sh
git clone https://github.com/Synoros-io/piccosta && cd piccosta
rustup target add x86_64-unknown-linux-musl
cargo build --profile dist --target x86_64-unknown-linux-musl
ln -s "$PWD/target/x86_64-unknown-linux-musl/dist/piccosta" ~/.local/bin/piccosta   # or anywhere on your PATH
```

`cargo build --release` builds faster against your own glibc, and runs only on a glibc at least
as new.

Log in with each account's own tool, check that piccosta finds the logins, and set the model it starts on (`defaultProvider` and `defaultModel` in `~/.local/share/piccosta/agent/settings.json`, or `/model` in the terminal UI):

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

`piccosta zed [dir]` opens Zed on a directory with piccosta as its agent, adding piccosta to Zed's settings first if it isn't there (a copy of the file as it was is kept). `piccosta t3 [dir]` starts T3 Code with its Pi provider running piccosta, telemetry off; the first time, turn the Pi provider on in T3's settings. If either isn't on your PATH, `"gui": {"zed": "/path/to/zed", "t3": "/path/to/t3"}` in piccosta's `settings.json` says where it is.

By hand, **Zed**, in `settings.json`:

```json
"agent_servers": {
  "piccosta": { "type": "custom", "command": "/path/to/piccosta", "args": ["--mode", "acp"] }
}
```

Zed's model, thinking and mode pickers work. Its Ask, Edits and Auto modes decide which tool calls ask first.

**T3 Code:** add a pi provider and set its binary path to `piccosta`. T3 speaks pi's rpc mode, which piccosta implements. Its supervised mode asks before each tool call.

Add `--yolo` to skip permissions.

## Settings

piccosta keeps its files in `~/.local/share/piccosta/agent`, or `$PICCOSTA_AGENT_DIR`: `settings.json`, `models.json`, sessions and subagent roles (`agents/<name>.md`).

`settings.json` uses pi's keys:
- `defaultProvider` and `defaultModel`;
- `defaultThinkingLevel` and `modelThinkingLevels`;
- `compaction`, `retry`, `shellCommandPrefix`, `transport` and `subagents`.

A project's `.pi/settings.json` is merged over them.

## License

piccosta is FSL-1.1-ALv2, licensor Synoros. You can use and change it freely, but selling it as a competing product needs Synoros. `LICENSE.md` has the license and the notices of the MIT projects piccosta is based on (pi, pi-web-access, pi-claude-bridge, cc-session-io).
