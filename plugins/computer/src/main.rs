// The optional computer plugin. Only the JSON-lines transport lives here; desktop actions keep
// their experimental implementation and its upstream notices in computer.rs.
#[cfg(target_os = "linux")]
mod computer;
#[cfg(target_os = "linux")]
mod isolation;

#[cfg(target_os = "linux")]
fn main() {
    if let Err(error) = isolation::run() { eprintln!("Private app sandbox: {error}"); std::process::exit(1); }
    let cwd = std::env::current_dir().unwrap_or_default();
    let settings = piccosta::config::Settings::load(&piccosta::config::agent_dir(), Some(&cwd));
    let _ = *isolation::AVAILABLE;
    if std::env::args().nth(1).as_deref() != Some("describe") && let Err(error) = isolation::desktop() {
        eprintln!("Private desktop sandbox: {error}"); std::process::exit(1);
    }
    tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap().block_on(serve(cwd, settings));
}

#[cfg(target_os = "linux")]
async fn serve(cwd: std::path::PathBuf, settings: piccosta::config::Settings) {
    use computer::{ComputerTool, Settings, ToolContext};
    use piccosta::message::Content;
    use serde_json::{Value, json};
    use std::io::{BufRead, Write};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    let mut definition = serde_json::to_value(ComputerTool::definition()).unwrap();
    definition["kinds"] = json!(["visual"]);
    let mut asks = vec![json!({"when": {"desktop": "main"}, "question": computer::MAIN_QUESTION})];
    if !*isolation::AVAILABLE {
        asks.push(json!({"when": {"action": "open"}, "question": isolation::FALLBACK_QUESTION}));
    }
    definition["asks"] = json!(asks);
    if std::env::args().nth(1).as_deref() == Some("describe") {
        println!("{}", json!({"name": env!("CARGO_PKG_NAME"), "version": env!("CARGO_PKG_VERSION"), "tools": [definition]}));
        return;
    }
    let tool = Arc::new(ComputerTool(Settings::new(&settings)));
    let (input, mut inputs) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            if input.send(line).is_err() { break; }
        }
    });
    let (output, mut outputs) = tokio::sync::mpsc::unbounded_channel();
    let mut active: Option<(String, CancellationToken)> = None;
    let write = |answer: Value| {
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "{answer}").and_then(|()| stdout.flush()).is_ok()
    };
    loop {
        tokio::select! {
            line = inputs.recv() => {
                let Some(Ok(line)) = line else { break };
                let message: Value = match serde_json::from_str(&line) {
                    Ok(message) => message,
                    Err(error) => { eprintln!("piccosta-computer: invalid message: {error}"); continue; }
                };
                if let Some(id) = message.get("cancel").and_then(Value::as_str) {
                    if let Some((call, cancel)) = &active && call == id { cancel.cancel(); }
                    continue;
                }
                let Some(id) = message.get("call").and_then(Value::as_str).map(str::to_string) else { continue };
                if active.is_some() {
                    if !write(json!({"call": id, "content": [Content::text("A computer call is already running")], "isError": true})) { break; }
                    continue;
                }
                let arguments = message.get("arguments").and_then(Value::as_object).cloned();
                let valid = message.get("tool").and_then(Value::as_str) == Some("computer");
                let context = ToolContext {
                    cwd: message.get("cwd").and_then(Value::as_str).map(std::path::PathBuf::from).unwrap_or_else(|| cwd.clone()),
                    approved: message.get("approved").and_then(Value::as_bool).unwrap_or(false),
                    cancel: CancellationToken::new(), model_accepts_images: true,
                    image_limits: piccosta::tools::image::Limits::of(None),
                };
                active = Some((id.clone(), context.cancel.clone()));
                let (tool, output) = (tool.clone(), output.clone());
                tokio::spawn(async move {
                    let result = match (valid, arguments) {
                        (true, Some(arguments)) => tool.execute(arguments, &context).await,
                        _ => Err("Expected the computer tool and an arguments object".into()),
                    };
                    let (content, is_error) = match result {
                        Ok(result) => (result.content, result.is_error),
                        Err(error) => (vec![Content::text(error)], true),
                    };
                    let _ = output.send(json!({"call": id, "content": content, "isError": is_error}));
                });
            }
            Some(answer) = outputs.recv() => {
                active = None;
                if !write(answer) { break; }
            }
            _ = piccosta::proc::termination(true) => break,
        }
    }
    if let Some((_, cancel)) = active { cancel.cancel(); }
    tool.0.release();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("piccosta-computer requires Linux and X11; the piccosta core does not.");
    std::process::exit(1);
}
