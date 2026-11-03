// The piccosta command: the terminal UI, `-p` and `--mode json` runs, rpc mode, and `piccosta auth
// status`.

use piccosta::agent::Event;
use piccosta::message::{AgentMessage, Content, StopReason};
use piccosta::{auth, cli, config, model};
use std::io::{IsTerminal, Write};
use std::process::ExitCode;
use std::time::Duration;

// How long a piped stdin may stay silent before the run starts without it. pi waits forever, so
// `sleep 60 | pi -p x` hung; the piccosta launcher added this guard (bin/piccosta:160-192).
const STDIN_WAIT_ENV: &str = "PICCOSTA_STDIN_WAIT_S";
const DEFAULT_STDIN_WAIT_SECONDS: u64 = 30;

// The exit code for a command line that can't run (sysexits' EX_USAGE is 64; 2 is what shells and
// argument parsers use, and what pi exits with on a bad flag).
const USAGE_EXIT_CODE: u8 = 2;

// `piccosta auth status`: for each account piccosta knows (auth::ACCOUNTS), whether a credential
// resolves and where it comes from. Keys are never printed. An account with none is listed, not
// a failure: the user has the accounts they have. Output: exit 1 when none has a credential.
async fn auth_status() -> Result<ExitCode, String> {
    let agent_dir = config::agent_dir();
    let paths = auth::CredentialPaths::from_environment(&agent_dir);
    let registry = model::ModelRegistry::load(&agent_dir.join("models.json"))?;
    let mut any = false;
    for account in &auth::ACCOUNTS {
        match auth::credential_for(account.provider, registry.provider_config(account.provider), &paths).await {
            Ok(credential) => {
                any = true;
                println!("ok   {:<12} {}", account.plan, credential.source);
            }
            Err(error) => println!("--   {:<12} {error}", account.plan),
        }
    }
    Ok(if any { ExitCode::SUCCESS } else { ExitCode::FAILURE })
}

// Piped stdin, trimmed, or "" when it's a terminal, empty, or silent past the wait.
async fn read_piped_stdin() -> String {
    if std::io::stdin().is_terminal() {
        return String::new();
    }
    let wait = std::env::var(STDIN_WAIT_ENV).ok().and_then(|seconds| seconds.parse().ok()).unwrap_or(DEFAULT_STDIN_WAIT_SECONDS);
    // The read runs on a plain thread, not tokio's blocking pool. On a pipe nobody writes to or
    // closes, it never returns, and the runtime waits for its blocking tasks at exit: the process
    // would hang after the run, with SIGTERM's listener already gone.
    let (started_sender, started) = tokio::sync::oneshot::channel::<bool>();
    let (rest_sender, rest) = tokio::sync::oneshot::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        use std::io::Read;
        let mut stdin = std::io::stdin().lock();
        let mut first = [0u8; 1];
        let arrived = matches!(stdin.read(&mut first), Ok(1));
        let _ = started_sender.send(arrived);
        if arrived {
            let mut bytes = vec![first[0]];
            let _ = stdin.read_to_end(&mut bytes);
            let _ = rest_sender.send(bytes);
        }
    });
    match tokio::time::timeout(Duration::from_secs(wait), started).await {
        Ok(Ok(true)) => {
            let bytes = rest.await.unwrap_or_default();
            piccosta::js::trim(&String::from_utf8_lossy(&bytes)).to_string()
        }
        Ok(_) => String::new(),
        Err(_) => {
            eprintln!("piccosta: nothing arrived on stdin for {wait} s ({STDIN_WAIT_ENV}); running without it");
            String::new()
        }
    }
}

fn write_line(line: &str) {
    let mut stdout = std::io::stdout().lock();
    if stdout.write_all(line.as_bytes()).and_then(|()| stdout.write_all(b"\n")).and_then(|()| stdout.flush()).is_err() {
        // The reader went away (EPIPE): nothing more can be reported.
        std::process::exit(1);
    }
}

// A print or json run: every message in turn, then (text mode) the last answer on stdout. Output:
// the exit code, or the error that stopped the run before it began.
async fn run_print(args: cli::Args) -> Result<ExitCode, String> {
    let json = args.mode.as_deref() == Some("json");
    let host = piccosta::host::Host::for_args(&args);
    host.show_settings_errors();
    let mut session = host.opening(&args)?;
    // A subagent's new session records its job as its kind, whose tools and readOnly it then runs
    // with, now and whenever it's reopened.
    if let Some(job) = std::env::var(piccosta::tools::subagent::CHILD_ENV).ok().filter(|_| session.entries().is_empty()) {
        piccosta::manager::set_work(&mut session, &job).map_err(|error| format!("could not write the session: {error}"))?;
    }
    let emit: piccosta::agent::EventSink = if json {
        Box::new(|event: &Event| write_line(&serde_json::to_string(event).unwrap_or_default()))
    } else {
        Box::new(|_event: &Event| {})
    };
    let (model, thinking) = host.launch(&args)?;
    let (mut agent, tool_shared) = host.open(session, model.as_deref(), thinking.as_deref(), emit, true)?;
    if let Some(name) = &args.name {
        agent.set_name(name).map_err(|error| format!("could not write the session: {error}"))?;
    }
    if json {
        write_line(&serde_json::to_string(agent.session.header()).unwrap_or_default());
    }

    // SIGTERM and SIGHUP end the run, after killing background jobs with their process groups.
    // SIGINT goes the SIGTERM way (exit 143), as longcmd made it: Node's default left jobs
    // running (longcmd.ts:350-363).
    let shutdown = tool_shared.clone();
    let abort = agent.handle();
    tokio::spawn(async move {
        use tokio::signal::unix::{signal, SignalKind};
        let (Ok(mut term), Ok(mut hangup), Ok(mut interrupt)) =
            (signal(SignalKind::terminate()), signal(SignalKind::hangup()), signal(SignalKind::interrupt()))
        else {
            return;
        };
        let code = tokio::select! {
            _ = term.recv() => 143,
            _ = interrupt.recv() => 143,
            _ = hangup.recv() => 129,
        };
        abort.abort();
        piccosta::provider::claude::shutdown().await;
        shutdown.jobs.shutdown(if code == 129 { "SIGHUP" } else { "SIGTERM" });
        // A foreground command whose tool call hasn't seen the abort yet.
        piccosta::tools::bash::kill_all_running();
        shutdown.web.shutdown();
        std::process::exit(code);
    });

    // The first prompt is piped stdin, then a blank line, then the first message (pi runs the two
    // together, so a file's last line fused with the question); each further message is a prompt
    // of its own. With neither there's nothing to answer, which is an error, not a quiet success.
    let stdin = read_piped_stdin().await;
    let mut messages = args.messages.clone().into_iter();
    let first = [stdin, messages.next().unwrap_or_default()].into_iter().filter(|part| !part.is_empty()).collect::<Vec<_>>().join("\n\n");
    let prompts: Vec<String> = std::iter::once(first).filter(|text| !text.is_empty()).chain(messages).collect();
    if prompts.is_empty() {
        return Err("nothing to answer: give a message, or pipe one in".into());
    }
    // The team (--pair or --trio, recorded with the session, or the one it already has): each
    // answer goes round it as in the terminal UI, and the run ends after the exchange.
    let team = match &args.team {
        Some(team) => piccosta::host::record_team(&host.registry()?, &mut agent.session, team)?,
        None => piccosta::host::team(&agent.session),
    };
    // Then whatever was left in the session's inbox meanwhile (another piccosta's messages for it),
    // until it's empty.
    let mut queue: std::collections::VecDeque<String> = prompts.into();
    while let Some(text) = queue.pop_front().or_else(|| agent.session.file().map(piccosta::manager::take_inbox).and_then(|more| {
        queue.extend(more);
        queue.pop_front()
    })) {
        let before = agent.session.context().messages.len();
        let answered = host.answer(&mut agent, &team, &text, Vec::new()).await;
        // With a team, every reply of the exchange is printed under the model that wrote it, so
        // the conversation reads as it happened; alone, only the last message is (below).
        if !json && !team.models.is_empty() {
            let mut speaker = String::new();
            for message in agent.session.context().messages.into_iter().skip(before) {
                let AgentMessage::Assistant(reply) = message else { continue };
                let model = format!("{}/{}", reply.provider, reply.model);
                if model != speaker {
                    write_line(&format!("\n── {model} ──"));
                    speaker = model;
                }
                reply.content.iter().for_each(|block| if let Content::Text { text, .. } = block { write_line(text) });
            }
        }
        if let Err(error) = answered {
            piccosta::provider::claude::shutdown().await;
            tool_shared.jobs.shutdown("error");
            tool_shared.web.shutdown();
            return Err(error);
        }
    }
    // The last answer's Claude query waits for a next message that won't come here.
    piccosta::provider::claude::shutdown().await;
    tool_shared.jobs.shutdown("quit");
    tool_shared.web.shutdown();
    // A subagent's session is one task: done when its run is, so it closes and leaves its parent's
    // list.
    if std::env::var_os(piccosta::tools::subagent::CHILD_ENV).is_some()
        && let Some(file) = agent.session.file()
    {
        piccosta::session::set_closed(&host.agent_dir, file, true);
    }

    // The last message, when it's the model's (print-mode.ts:139-159). A run that ended in an
    // error exits 1 in json mode too, so a script checking $? sees it (pi's json mode exits 0).
    let last = agent.session.context().messages.pop();
    let Some(AgentMessage::Assistant(answer)) = last else { return Ok(ExitCode::SUCCESS) };
    if matches!(answer.stop_reason, StopReason::Error | StopReason::Aborted) {
        if !json {
            let reason = if answer.stop_reason == StopReason::Error { "error" } else { "aborted" };
            eprintln!("{}", answer.error_message.unwrap_or_else(|| format!("Request {reason}")));
        }
        return Ok(ExitCode::FAILURE);
    }
    if json || !team.models.is_empty() {
        return Ok(ExitCode::SUCCESS);
    }
    for block in &answer.content {
        if let Content::Text { text, .. } = block {
            write_line(text);
        }
    }
    if answer.stop_reason == StopReason::Length {
        eprintln!("Response was truncated: the model reached its output token limit.");
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

#[tokio::main]
async fn main() -> ExitCode {
    // Started as the reaper by another piccosta (tools/reaper.rs): watch for it to go, and nothing else.
    if std::env::var_os(piccosta::tools::reaper::REAPER_ENV).is_some() {
        piccosta::tools::reaper::run();
        return ExitCode::SUCCESS;
    }
    // Each account's model list, fetched again in the background (model.rs); a start with no list
    // kept that reads waits for them, and says what went wrong. The terminal UI shows that too.
    let agent_dir = config::agent_dir();
    let waits = !model::has_lists(&agent_dir);
    let refresh = model::start_refresh(agent_dir.clone());
    if waits {
        for problem in refresh.await.into_iter().filter_map(Result::err) {
            eprintln!("piccosta: {problem}");
        }
    }
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let fail = |error: String| {
        eprintln!("piccosta: {error}");
        ExitCode::FAILURE
    };
    // `piccosta assistant [flags]` runs the assistant service; `piccosta tell <text>` leaves it a
    // message.
    match arguments.first().map(String::as_str) {
        Some("assistant") => return piccosta::manager::run(cli::parse(&arguments[1..])).await.unwrap_or_else(fail),
        // `piccosta zed [dir]`, `piccosta t3 [dir]`: the GUI, running piccosta as its agent.
        Some(gui @ ("zed" | "t3")) => return piccosta::gui::open(gui, &arguments[1..]).map_or_else(fail, |()| ExitCode::SUCCESS),
        // `piccosta cache [hours]`: the prompt-cache misses across sessions (cache.rs).
        Some("cache") => {
            print!("{}", piccosta::cache::report(&config::agent_dir(), arguments.get(1).map(String::as_str)));
            return ExitCode::SUCCESS;
        }
        Some("tell") => {
            let text = arguments[1..].join(" ");
            let Some(file) = piccosta::manager::session_file(&config::agent_dir()) else { return fail("no assistant yet: start it with `piccosta assistant`".into()) };
            return match piccosta::manager::deliver(&file, &text) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => fail(format!("{}: {error}", file.display())),
            };
        }
        _ => {}
    }
    // `auth` is a command, never a prompt: anything but `auth status` gets the usage line.
    if arguments.first().map(String::as_str) == Some("auth") {
        if arguments.iter().map(String::as_str).eq(["auth", "status"]) {
            return auth_status().await.unwrap_or_else(fail);
        }
        eprintln!("Usage: piccosta auth status (log in with the account's own tool: codex login, opencode auth login, claude auth login)");
        return ExitCode::from(USAGE_EXIT_CODE);
    }
    let args = cli::parse(&arguments);
    for warning in &args.warnings {
        eprintln!("Warning: {warning}");
    }
    if !args.errors.is_empty() {
        for error in &args.errors {
            eprintln!("Error: {error}");
        }
        return ExitCode::FAILURE;
    }
    if args.help {
        print!("{}", cli::HELP);
        return ExitCode::SUCCESS;
    }
    if args.version {
        // The pi release whose protocols piccosta speaks comes first: frontends that drive pi
        // check the first version number they see (T3 Code wants at least 0.80.5).
        println!("pi {} compatible (piccosta {})", piccosta::host::PI_VERSION, env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    // pi's mode choice: json, else print when asked or when stdin or stdout isn't a terminal.
    let print = args.print || !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal();
    match args.mode.as_deref() {
        Some("rpc") => piccosta::rpc::run(args).await.unwrap_or_else(fail),
        Some("acp") => piccosta::acp::run(args).await.unwrap_or_else(fail),
        Some("json") => run_print(args).await.unwrap_or_else(fail),
        _ if print => run_print(args).await.unwrap_or_else(fail),
        _ => piccosta::tui::run(args).await.unwrap_or_else(fail),
    }
}
