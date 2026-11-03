// piccosta: one agent harness for every model you use. The library holds everything; the binary
// in main.rs only reads the command line and hands over.

pub mod acp;
pub mod agent;
pub mod auth;
pub mod cache;
pub mod cli;
pub mod clock;
pub mod compaction;
pub mod config;
pub mod gui;
pub mod host;
pub mod js;
pub mod machine;
pub mod manager;
pub mod json_number;
pub mod message;
pub mod model;
pub mod pintask;
pub mod proc;
pub mod prompt;
pub mod provider;
pub mod retry;
pub mod rpc;
pub mod scale;
pub mod session;
pub mod tools;
pub mod tui;
