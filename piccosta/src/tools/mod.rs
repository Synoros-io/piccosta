// The tools a model can call, and what every call needs from the session.
//
// A tool has a definition (name, description, JSON Schema, sent to the model), prompt lines for the
// system prompt, an optional repair of argument shapes models get wrong, and an async execute. The
// agent loop validates arguments against the schema (validate.rs) before execute sees them. A
// tool's Err is its error message: the loop turns it into an error result for the model, as pi
// does with a thrown error.
//
// The built-in four are pi's read, write, edit and bash, with piccosta's editfix folded into edit
// and longcmd and path hints into bash.

pub mod bash;
pub mod computer;
pub mod edit;
pub mod editfix;
pub mod image;
pub mod jobs;
pub mod mutation;
pub mod output;
pub mod pathhint;
pub mod paths;
pub mod read;
pub mod readonly;
pub mod reaper;
pub mod script;
pub mod subagent;
pub mod truncate;
pub mod validate;
pub mod web;
pub mod write;

use crate::message::{Content, CustomMessage, ToolDefinition};
use serde_json::{Map, Value};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutput, String>> + Send + 'a>>;

// What a tool hands back: content for the model, details for the session and UI.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutput {
    pub content: Vec<Content>,
    pub details: Option<Value>,
    // A result the model should read as a failure (bash's non-zero exit) without the tool
    // having failed to run.
    pub is_error: bool,
}

// Partial output while a tool runs, for the UI (bash streams its output this way).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolUpdate {
    pub content: Vec<Content>,
    pub details: Option<Value>,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn definition(&self) -> ToolDefinition;
    // One line for the system prompt's tool list, and guidelines for its guideline list.
    fn prompt_snippet(&self) -> Option<&'static str> {
        None
    }
    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &[]
    }
    // Whether rpc and ACP run it without asking the user: it only reads, or (the script) each call
    // it makes is asked about on its own.
    fn read_only(&self) -> bool {
        false
    }
    // Repair argument shapes models are known to send, before validation (pi's prepareArguments).
    fn prepare_arguments(&self, arguments: Map<String, Value>) -> Map<String, Value> {
        arguments
    }
    // A question the user has to answer yes to before this call runs, whatever the frontend's
    // approval mode; with nobody to ask (print mode, a subagent), the call is refused (agent.rs).
    fn approval(&self, _arguments: &Map<String, Value>) -> Option<&'static str> {
        None
    }
    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a>;
}

// State every tool call in a session shares.
pub struct ToolShared {
    // One writer per file at a time.
    pub mutations: &'static mutation::MutationQueue,
    // What the model last saw of each file, for editfix's diagnosis.
    pub seen: editfix::SeenFiles,
    // Background bash jobs.
    pub jobs: jobs::Jobs,
    // From settings: `shellPath` and `shellCommandPrefix`.
    pub shell_path: Option<String>,
    pub shell_command_prefix: Option<String>,
    // The agent dir's bin/, put first on PATH for commands.
    pub bin_dir: PathBuf,
    // The web tools' client, settings and stored results.
    pub web: web::WebShared,
    // The computer tool's hidden desktop.
    pub computer: computer::Settings,
    // Whether bash runs in the read-only sandbox: a session for a read-only kind, a read-only job's
    // subagent among them (host.rs).
    pub read_only: bool,
    // Whether the session is of the visual kind, whose bash commands then run on the hidden
    // desktop (host.rs, bash.rs).
    pub visual: bool,
}

impl ToolShared {
    // Input: the settings in effect, the agent dir, where background jobs' finish messages go, and
    // the model registry and credential paths web_search's search model comes from.
    pub fn new(
        settings: &crate::config::Settings,
        agent_dir: &std::path::Path,
        notices: mpsc::UnboundedSender<CustomMessage>,
        registry: &crate::model::ModelRegistry,
        credentials: crate::auth::CredentialPaths,
    ) -> ToolShared {
        ToolShared {
            mutations: {
                static MUTATIONS: std::sync::LazyLock<mutation::MutationQueue> = std::sync::LazyLock::new(mutation::MutationQueue::default);
                &MUTATIONS
            },
            seen: editfix::SeenFiles::default(),
            jobs: jobs::Jobs::new(notices),
            shell_path: settings.shell_path(),
            shell_command_prefix: settings.shell_command_prefix(),
            bin_dir: agent_dir.join("bin"),
            web: web::WebShared::new(settings.web(), agent_dir, registry, credentials),
            computer: computer::Settings::new(settings),
            read_only: false,
            visual: false,
        }
    }
}

// What one tool call knows about the session it runs in.
#[derive(Clone)]
pub struct ToolContext {
    pub cwd: PathBuf,
    pub session_id: String,
    pub session_file: Option<PathBuf>,
    pub provider: String,
    pub model_id: String,
    pub thinking_level: String,
    // Whether the current model takes images; read says so beside an image when it doesn't.
    pub model_accepts_images: bool,
    // What an image read or fetched is fitted to: the model's own limits, or Anthropic's.
    pub image_limits: image::Limits,
    // Set when the user aborts the turn.
    pub cancel: CancellationToken,
    pub updates: Option<mpsc::UnboundedSender<ToolUpdate>>,
    // A script's calls: its commands never move to the background, the script moves as a whole.
    pub stay_in_foreground: bool,
    // The session's approval gate (rpc's, ACP's) and the id of the model's call this is for, so the
    // calls a script makes are asked about as the model's own are.
    pub gate: Option<(String, crate::agent::Gate)>,
    pub shared: Arc<ToolShared>,
}

// Whether the built-in tool called `name` only reads (Tool::read_only); no other tool does.
pub fn read_only(name: &str) -> bool {
    builtin().iter().any(|tool| tool.name() == name && tool.read_only())
}

// The built-in tools, in the order pi lists them, then the web tools, the computer and the script
// tool (with no functions until script::bind gives it the session's list).
pub fn builtin() -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = vec![Arc::new(read::ReadTool), Arc::new(bash::BashTool), Arc::new(edit::EditTool), Arc::new(write::WriteTool)];
    tools.extend(web::tools());
    tools.push(Arc::new(computer::ComputerTool));
    tools.push(Arc::new(script::ScriptTool::default()));
    tools
}
