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

// `piccosta auth status`: for each account piccosta knows (auth::ACCOUNTS), then each other
// provider models.json configures, whether a credential resolves and where it comes from. Keys are
// never printed. A provider with none is listed, not a failure: the user has the accounts they
// have. A provider is ok exactly when credential_for, which its runs use, accepts it. Output: exit
// 1 when none is ok.
async fn auth_status() -> Result<ExitCode, String> {
    let agent_dir = config::agent_dir();
    let paths = auth::CredentialPaths::from_environment(&agent_dir);
    let registry = model::ModelRegistry::load(&agent_dir.join("models.json"))?;
    let accounts = auth::ACCOUNTS.iter().map(|account| (account.provider, account.plan));
    let serves = |provider: &str| registry.provider_config(provider).is_some_and(|config| config.base_url.is_some() || !config.models.is_empty());
    let configured = registry.configured().filter(|provider| auth::account(provider).is_none() && serves(provider)).map(|provider| (provider, provider));
    let mut any = false;
    for (provider, name) in accounts.chain(configured).collect::<Vec<_>>() {
        match auth::credential_for(provider, registry.provider_config(provider), &paths).await {
            Ok(credential) => {
                any = true;
                println!("ok   {name:<12} {}", credential.source);
            }
            Err(error) => println!("--   {name:<12} {error}"),
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

// A print or json run: what's queued in the session, every message in turn, then whatever is sent
// to its runtime while it runs, and (text mode) the last prompt's answer on stdout. Output: the exit code, or
// the error that stopped the run before it began.
async fn run_print(args: cli::Args) -> Result<ExitCode, String> {
    let json = args.mode.as_deref() == Some("json");
    let host = std::sync::Arc::new(piccosta::host::Host::for_args(&args));
    host.show_settings_errors();
    let session = host.opening(&args)?;
    let emit: piccosta::agent::EventSink = if json {
        Box::new(|event: &Event| write_line(&serde_json::to_string(event).unwrap_or_default()))
    } else {
        Box::new(|_event: &Event| {})
    };
    let (model, thinking, note) = host.launch(&args)?;
    if let Some(listing) = host.listing(Some(&session), model.as_deref()) {
        listing.await;
    }
    let (mut agent, tool_shared) = host.open(session, model.as_deref(), thinking.as_deref(), emit, true)?;
    if let Some(note) = note {
        eprintln!("{}", piccosta::scale::fell_back(&note, &format!("{}/{}", agent.model().provider, agent.model().id)));
    }
    if let Some(name) = &args.name {
        agent.set_name(name).map_err(|error| format!("could not write the session: {error}"))?;
    }
    if json {
        write_line(&serde_json::to_string(agent.session.header()).unwrap_or_default());
    }
    // The first prompt is piped stdin, then a blank line, then the first message (pi runs the two
    // together, so a file's last line fused with the question); each further message is a prompt
    // of its own. With neither, and nothing queued in the session, there's nothing to answer, which
    // is an error, not a quiet success.
    let stdin = read_piped_stdin().await;
    let mut messages = args.messages.clone().into_iter();
    let first = [stdin, messages.next().unwrap_or_default()].into_iter().filter(|part| !part.is_empty()).collect::<Vec<_>>().join("\n\n");
    let prompts: Vec<String> = std::iter::once(first).filter(|text| !text.is_empty()).chain(messages).collect();
    if prompts.is_empty() && agent.session.queued().is_empty() {
        return Err("nothing to answer: give a message, or pipe one in".into());
    }
    // The team (--pair or --trio, recorded with the session, or the one it already has): each
    // answer goes round it as in the terminal UI, and the run ends after the exchange.
    let team = match &args.team {
        Some(team) => piccosta::host::record_team(&host.registry()?, &mut agent.session, team)?,
        None => piccosta::host::team(&agent.session),
    };
    let before = agent.session.context().messages.len();
    // The run is the session's runtime, which ends when nothing is left to answer; the first
    // failure to write the session is the run's error.
    let failed = std::rc::Rc::new(std::cell::RefCell::new(None));
    let failure = failed.clone();
    let (commands, task) = piccosta::runtime::spawn(host.clone(), agent, tool_shared, true, true, move |done| {
        if let Err(error) = done.result {
            failure.borrow_mut().get_or_insert(error);
        }
    });
    for prompt in prompts {
        let _ = commands.send(piccosta::runtime::Command::Prompt(prompt, Vec::new()));
    }
    // SIGTERM and SIGHUP close the runtime, which stops the run, keeps what it hadn't read in the
    // session and kills its jobs with their process groups. SIGINT goes the SIGTERM way (exit 143),
    // as longcmd made it: Node's default left jobs running (longcmd.ts:350-363).
    let signalled = std::rc::Rc::new(std::cell::Cell::new(None));
    let (signal_code, closer) = (signalled.clone(), commands.clone());
    tokio::task::spawn_local(async move {
        let code = piccosta::proc::termination(true).await;
        signal_code.set(Some(code));
        let _ = closer.send(piccosta::runtime::Command::Close(None));
        tokio::time::sleep(piccosta::runtime::STOP_WAIT).await;
        std::process::exit(code.into());
    });
    drop(commands);
    let agent = task.await.map_err(|error| error.to_string())?;
    // The last answer's Claude query waits for a next message that won't come here.
    piccosta::provider::claude::shutdown().await;
    if let Some(code) = signalled.get() {
        return Ok(ExitCode::from(code));
    }
    if let Some(error) = failed.borrow_mut().take() {
        return Err(error);
    }
    // With a team, every reply of the run is printed under the model that wrote it, so the
    // conversation reads as it happened; alone, only the last message is (below).
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
    // A subagent's session is one task: done when its run is, so it closes and leaves its parent's
    // list.
    if std::env::var_os(piccosta::tools::subagent::CHILD_ENV).is_some()
        && let Some(file) = agent.session.file()
    {
        piccosta::session::set_closed(&host.agent_dir, file, true);
    }

    // The last message, when it's the model's (print-mode.ts:139-159). A run that ended in an
    // error exits 1 in json mode too, so a script checking $? sees it (pi's json mode exits 0).
    let messages = agent.session.context().messages;
    let Some(AgentMessage::Assistant(answer)) = messages.last() else { return Ok(ExitCode::SUCCESS) };
    if matches!(answer.stop_reason, StopReason::Error | StopReason::Aborted) {
        if !json {
            let reason = if answer.stop_reason == StopReason::Error { "error" } else { "aborted" };
            eprintln!("{}", answer.error_message.clone().unwrap_or_else(|| format!("Request {reason}")));
        }
        return Ok(ExitCode::FAILURE);
    }
    if json || !team.models.is_empty() {
        return Ok(ExitCode::SUCCESS);
    }
    // The answer is every turn that ended since the last prompt, in order: a job still running
    // after the model's answer starts another turn when it ends, and the model's reply to that
    // news (often "nothing to do") would otherwise be all a subagent's parent gets.
    let since_prompt = messages.iter().rposition(|message| matches!(message, AgentMessage::User(_))).map_or(0, |prompt| prompt + 1);
    for message in &messages[since_prompt..] {
        let AgentMessage::Assistant(reply) = message else { continue };
        if reply.stop_reason == StopReason::ToolUse { continue }
        reply.content.iter().for_each(|block| if let Content::Text { text, .. } = block { write_line(text) });
    }
    if answer.stop_reason == StopReason::Length {
        eprintln!("Response was truncated: the model reached its output token limit.");
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    #[cfg(target_os = "linux")]
    if std::env::var_os(piccosta::tools::plugin_host::HOST_ENV).is_some() { piccosta::tools::plugin_host::run(); }
    // An image /restart started: what it inherited closes on exec before anything can spawn.
    #[cfg(unix)]
    if std::env::args_os().any(|argument| argument == "--restore") { piccosta::tools::reaper::inherit(); }
    run()
}

#[tokio::main]
async fn run() -> ExitCode {
    // Started as the watcher by another piccosta (tools/reaper.rs): watch for it to go, and nothing else.
    if std::env::var_os(piccosta::tools::reaper::REAPER_ENV).is_some() {
        piccosta::tools::reaper::run();
        return ExitCode::SUCCESS;
    }
    // Each account's model list, fetched again in the background (model.rs); a start with no list
    // kept that reads waits for them, and says what went wrong. The terminal UI shows that too.
    // --help and --version need none.
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    // Local UI control needs no model startup and must never become a prompt.
    if arguments.first().is_some_and(|arg| matches!(arg.as_str(), "open" | "restart" | "tui")) {
        return match piccosta::tui::control::cli(&arguments) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("piccosta: {error}"); ExitCode::FAILURE }
        };
    }
    // Model setup needs no account login or startup refresh, and must never become a prompt.
    if arguments.first().is_some_and(|arg| arg == "models") {
        return match piccosta::model_add::cli(&arguments[1..]).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("piccosta: {error}"); ExitCode::FAILURE }
        };
    }
    let args = cli::parse(&arguments);
    let agent_dir = config::agent_dir();
    if !(args.help || args.version) {
        let waits = !model::has_lists(&agent_dir);
        let refresh = model::start_refresh(agent_dir.clone());
        if waits {
            for problem in refresh.await.into_iter().filter_map(Result::err) {
                eprintln!("piccosta: {problem}");
            }
        }
    }
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
            let text = piccosta::manager::from_user(&arguments[1..].join(" "));
            let Some(file) = piccosta::manager::session_file(&config::agent_dir()) else { return fail("no assistant yet: start it with `piccosta assistant`".into()) };
            // To its runtime; with the service down it waits in its session.
            return match piccosta::runtime::send(&file, &text, None) {
                // Sent all the same, so it isn't a failure to send again.
                Ok(piccosta::runtime::Sent::Unconfirmed) => {
                    eprintln!("piccosta: sent, not confirmed: the assistant didn't answer in time");
                    ExitCode::SUCCESS
                }
                Ok(_) => ExitCode::SUCCESS,
                Err(error) => fail(error),
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
        Some("json") => tokio::task::LocalSet::new().run_until(run_print(args)).await.unwrap_or_else(fail),
        _ if print => tokio::task::LocalSet::new().run_until(run_print(args)).await.unwrap_or_else(fail),
        _ => piccosta::tui::run(args).await.unwrap_or_else(fail),
    }
}
