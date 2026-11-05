// The command line, parsed the way pi parses it (packages/coding-agent/src/cli/args.ts), for the
// flags piccosta has so far. ds, the bench and pi-subagents spawn pi with exact flags, so the
// quirks are kept: `-p` takes the next word as the message unless it starts with "@" or "-" (but
// "---" is a message); a value flag at the end of the line with no value is ignored; an unknown
// `--x` takes the next word as its value unless it starts with "-" or "@", and is then an error.

#[derive(Debug, Default)]
pub struct Args {
    pub print: bool,
    // "text", "json" or "rpc".
    pub mode: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub thinking: Option<String>,
    pub continue_recent: bool,
    pub session: Option<String>,
    // -r/--resume [id or name]: that session (as --session finds it), or "" for the session picker.
    pub resume: Option<String>,
    pub no_session: bool,
    // --no-context-files: leave AGENTS.md and CLAUDE.md out of the prompt.
    pub no_context_files: bool,
    // --no-tools (and --no-builtin-tools: piccosta's tools are all built in): no tools at all.
    pub no_tools: bool,
    // --fork <file>: start on a new session holding that one's history.
    pub fork: Option<String>,
    pub session_dir: Option<String>,
    // --name: the session's name.
    pub name: Option<String>,
    // --use <name>: the model scale's choice for that kind of work (scale.rs), unless --model is given.
    pub use_scale: Option<String>,
    // --pair <model> or --trio <model>: the session works with that model (/pair, /trio).
    pub team: Option<crate::host::Team>,
    // --tools a,b: only these tools; --exclude-tools a,b: all but these.
    pub tools: Option<Vec<String>>,
    pub exclude_tools: Vec<String>,
    // --append-system-prompt, each one text or a file's path, in order.
    pub append_prompt: Vec<String>,
    // --yolo (Claude Code's --dangerously-skip-permissions, Codex's
    // --dangerously-bypass-approvals-and-sandbox): run every tool without asking, even where a
    // frontend asks for approvals. The terminal never asks, as in pi.
    pub yolo: bool,
    // --restore: the terminal UI as it was when it last closed, whatever else the line asks for
    // (what /restart starts it again with).
    pub restore: bool,
    pub help: bool,
    pub version: bool,
    pub messages: Vec<String>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

// pi's thinking levels (args.ts VALID_THINKING_LEVELS).
const THINKING_LEVELS: [&str; 7] = crate::provider::THINKING_LEVELS;
// The modes --mode takes.
const MODES: [&str; 4] = ["text", "json", "rpc", "acp"];

// A comma-separated list of names, blanks dropped.
fn names(list: &str) -> Vec<String> {
    list.split(',').map(str::trim).filter(|name| !name.is_empty()).map(String::from).collect()
}

// The model asked for: --model, with --provider in front of a bare id.
pub fn model_name(args: &Args) -> Option<String> {
    match (&args.provider, &args.model) {
        (Some(provider), Some(model)) if !model.contains('/') => Some(format!("{provider}/{model}")),
        (_, model) => model.clone(),
    }
}

pub fn parse(arguments: &[String]) -> Args {
    let mut args = Args::default();
    let mut index = 0;
    // The value after a flag, or None when the flag is last on the line.
    let take = |index: &mut usize| -> Option<String> {
        let value = arguments.get(*index + 1).cloned();
        if value.is_some() {
            *index += 1;
        }
        value
    };
    while index < arguments.len() {
        let arg = arguments[index].as_str();
        match arg {
            "--help" | "-h" => args.help = true,
            "--version" | "-v" => args.version = true,
            "--mode" => args.mode = take(&mut index).or(args.mode.take()),
            "--provider" => args.provider = take(&mut index).or(args.provider.take()),
            // -m is Codex's short form.
            "--model" | "-m" => args.model = take(&mut index).or(args.model.take()),
            "--session" => args.session = take(&mut index).or(args.session.take()),
            // Claude Code's --resume: an id or name when one follows, else the picker.
            "--resume" | "-r" => {
                let named = arguments.get(index + 1).filter(|next| !next.starts_with('-')).cloned();
                index += usize::from(named.is_some());
                args.resume = Some(named.unwrap_or_default());
            }
            "--continue" | "-c" => args.continue_recent = true,
            "--restore" => args.restore = true,
            "--no-session" => args.no_session = true,
            "--yolo" | "--dangerously-skip-permissions" | "--dangerously-bypass-approvals-and-sandbox" => args.yolo = true,
            "--no-context-files" | "-nc" => args.no_context_files = true,
            // pi's flags for what piccosta doesn't load (extensions, skills, prompt templates,
            // themes, MCP) or ask (project trust), accepted so a frontend can launch piccosta the
            // way it launches pi (T3 Code passes them).
            "--no-extensions" | "-ne" | "--no-skills" | "-ns" | "--no-prompt-templates" | "-np" | "--no-themes" | "--no-mcp" | "--approve" | "-a" | "--no-approve" | "-na" | "--offline" | "--verbose" => {}
            "--no-tools" | "-nt" | "--no-builtin-tools" | "-nbt" => args.no_tools = true,
            "--fork" => args.fork = take(&mut index).or(args.fork.take()),
            "--session-dir" => args.session_dir = take(&mut index).or(args.session_dir.take()),
            "--name" | "-n" => args.name = take(&mut index).or(args.name.take()),
            "--use" => args.use_scale = take(&mut index).or(args.use_scale.take()),
            "--pair" | "--trio" => args.team = take(&mut index).map(|list| crate::host::Team { models: names(&list), pair: arg == "--pair" }).or(args.team.take()),
            "--extension" | "-e" => {
                take(&mut index);
            }
            "--tools" | "-t" => args.tools = take(&mut index).map(|list| names(&list)).or(args.tools.take()),
            "--exclude-tools" | "-xt" => args.exclude_tools.extend(take(&mut index).map(|list| names(&list)).unwrap_or_default()),
            "--append-system-prompt" => args.append_prompt.extend(take(&mut index)),
            // What would change the run in a way piccosta can't do stops it, rather than going
            // ahead on something else (a stored login instead of the key given, say).
            "--api-key" | "--system-prompt" | "--models" => {
                take(&mut index);
                args.errors.push(format!("piccosta doesn't support {arg}"));
            }
            // pi's look-and-feel flags: accepted, said so, not used.
            "--skill" | "--prompt-template" | "--theme" | "--use-theme" | "--tui-mode" => {
                take(&mut index);
                args.warnings.push(format!("piccosta doesn't support {arg} yet; ignored"));
            }
            // --effort is Claude Code's name for it.
            "--thinking" | "--effort" => {
                if let Some(level) = take(&mut index) {
                    if THINKING_LEVELS.contains(&level.as_str()) {
                        args.thinking = Some(level);
                    } else {
                        args.warnings.push(format!("Invalid thinking level \"{level}\". Valid values: {}", THINKING_LEVELS.join(", ")));
                    }
                }
            }
            "--print" | "-p" => {
                args.print = true;
                if let Some(next) = arguments.get(index + 1)
                    && !next.starts_with('@')
                    && (!next.starts_with('-') || next.starts_with("---"))
                {
                    args.messages.push(next.clone());
                    index += 1;
                }
            }
            _ if arg.starts_with('@') => args.errors.push(format!("@file arguments aren't supported yet: {arg}")),
            _ if arg.starts_with("--") => {
                let name = arg[2..].split('=').next().unwrap_or_default().to_string();
                if !arg.contains('=')
                    && let Some(next) = arguments.get(index + 1)
                    && !next.starts_with('-')
                    && !next.starts_with('@')
                {
                    index += 1;
                }
                args.errors.push(format!("Unknown option: --{name}"));
            }
            _ if arg.starts_with('-') => args.errors.push(format!("Unknown option: {arg}")),
            _ => args.messages.push(arg.to_string()),
        }
        index += 1;
    }
    // pi's provider/id:level shorthand: a level after the last colon sets the thinking level.
    if let Some((model, level)) = args.model.as_deref().and_then(|model| model.rsplit_once(':'))
        && THINKING_LEVELS.contains(&level)
    {
        args.thinking = args.thinking.take().or(Some(level.to_string()));
        args.model = Some(model.to_string());
    }
    // Tool names are checked when the session opens, after its plugins have been described.
    if let Some(mode) = &args.mode
        && !MODES.contains(&mode.as_str())
    {
        args.errors.push(format!("Invalid mode \"{mode}\". Valid values: {}", MODES.join(", ")));
    }
    args
}

pub const HELP: &str = "piccosta: one agent harness for every model you use

Usage:
  piccosta [options] [messages...]
  piccosta auth status
  piccosta models add <url> [--name n] [--api a] [--key-env VAR | --key-command CMD | --key-stdin]
  piccosta models remove <name>
  piccosta zed [dir]     Zed, with piccosta as its agent
  piccosta t3 [dir]      T3 Code, with piccosta as its Pi provider
  piccosta assistant     The user's assistant service (the terminal UI starts it with manager.autostart)
  piccosta tell <text>   A message for the assistant
  piccosta tui           List running terminal UIs and their open sessions (JSON)
  piccosta open <id or name> [--tui <id or pid>]   Open in the sidebar without changing focus
  piccosta restart [--tui <id or pid>]            Restart after running turns, keeping drafts
                         Without --tui, open/restart use the most recently active terminal UI

Options:
  -p, --print            Answer the message and exit (also when stdin or stdout isn't a terminal)
  --mode <mode>          text, json (the session header, then one event per line), rpc (pi's
                         protocol, for T3 Code) or acp (for Zed and other editors)
  -m, --model <id>       provider/id, or an id one provider has; provider/id:level sets thinking
  --provider <name>      The provider for a bare --model id
  --thinking <level>     off, minimal, low, medium, high, xhigh, max (also --effort)
  --use <kind>           The model the scale picks for that kind of work (critical, general,
                         review, reading, bugs, search, tests, visual, monitoring, exploration;
                         /scale lists them)
  -c, --continue         Continue this directory's most recent session; in the terminal UI,
                         everything as it was when it closed (its panes, drafts, open sessions)
  -r, --resume [id]      Resume a session by id, id prefix or name; without one, pick from a list
  --session <path|id>    Use this session file, or the session with this id or name
  --fork <path|id>       Start a new session holding that one's history
  --no-session           Keep the session in memory only
  --session-dir <dir>    Where session files go
  -n, --name <name>      Name the session
  --pair <model>         Work with that model as a pair: the two answer each other in turn
                         until one says it's done or the round limit, then you speak
  --trio <model>         Work with that model as a trio: after each answer it checks it once,
                         then you speak
  -t, --tools <a,b>      Only these tools; --no-tools for none
  -xt, --exclude-tools <a,b>  Every tool but these
  --append-system-prompt <text|file>  Add to the system prompt (repeatable)
  -nc, --no-context-files     Leave AGENTS.md and CLAUDE.md out
  --yolo                 Run every tool without asking, even in Zed's Ask mode or T3 Code's
                         Supervised (also --dangerously-skip-permissions,
                         --dangerously-bypass-approvals-and-sandbox); the terminal never asks
  -h, --help             This help
  -v, --version          The version
";
