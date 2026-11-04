// lfm: LFM3-Audio-2B as the two commands of piccosta's voice mode, speech in and text out (`listen`), text in
// and speech out (`speak`), through stdin and stdout, each loading the model itself or using the one an `lfm serve`
// holds for the length of a call.

use lfm::{VERBOSE, audio, codec::Codec, model::Model, note, quant::Format, tensor::Result};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::Shutdown,
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
    process::ExitCode,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

const USAGE: &str = "\
usage: lfm listen [options] [FILE]   transcribe 16 kHz WAV speech (FILE, or stdin) to stdout, once it ends
       lfm speak [options] [TEXT]    speak TEXT as a 24 kHz WAV on stdout, or else each line of stdin as it
                                     comes, into one WAV stream a player can play as it comes
       lfm serve [options] SOCKET    hold the model for listen and speak --server SOCKET until killed
       lfm decode [options] [FILE]   decode speech codes (a frame of codes a line) to a WAV on stdout
options:
  --model DIR     the LFM3-Audio-2B checkpoint directory (default: $LFM_MODEL_DIR); its files are read in
                  place, so they must not be changed while lfm runs
  --threads N     worker threads, at least one (default: $LFM_THREADS, else half the logical CPUs)
  --quant FORMAT  the language model's and depth decoder's weights: bf16 as in the checkpoint, q8, or q4 (the
                  language model's layers in 4 bits, the rest in 8), quantized once into model.q8.safetensors
                  or model.q4.safetensors beside the checkpoint (default: $LFM_QUANT, else bf16)
  --device NAME   where the language model and depth decoder run: cpu, or gpu when lfm is built with its
                  gpu feature (default: $LFM_DEVICE, else cpu)
  --system TEXT   the system prompt: the task, its language and the voice
  --max N         at most N text tokens (listen) or speech frames (speak, decode)
  --max-input N   refuse input over N bytes (default: $LFM_MAX_INPUT, else 64 MiB)
  --max-audio N   refuse speech over N seconds (default: $LFM_MAX_AUDIO, else 300)
  --max-prompt N  refuse prompts over N rows of text and speech (default: $LFM_MAX_PROMPT, else 4096)
  --seed N        the speech sampling seed (default: the clock)
  --greedy        speak deterministically: the likeliest codes, codebooks in order
  --raw           listen to headerless 16-bit mono PCM at 16 kHz (pw-record --container raw) instead of a WAV
  --server SOCKET listen or speak (stdin only) with the model of the lfm serve at SOCKET; the options that
                  shape the model's work are the server's, and a client takes none but --raw
  -o FILE         write the WAV to FILE instead of stdout
  -v              timings on stderr";

// The model card's system prompts (huggingface.co/LiquidAI/LFM3-Audio-2B) and generation budgets: 1024 tokens
// for a transcript, 256 frames (20 seconds) of speech.
const LISTEN_PROMPT: &str = "Perform ASR in English.";
const SPEAK_PROMPT: &str = "Perform TTS in American English using a female voice.";
const LISTEN_MAX_TOKENS: usize = 1024;
const SPEAK_MAX_FRAMES: usize = 256;

// The resource budget, so that a long or endless input is refused instead of filling memory. 64 MiB holds the
// longest speech allowed in any WAV format read (300 s of 16 kHz stereo 32-bit is 38 MB). 300 s, five minutes,
// is well past push-to-talk; the encoder makes 12.5 rows of a second of it, 3750 rows, and 4096 leaves room for
// the prompt's text. Memory grows with the speech's length: the downsampler's first convolution holds 3.3 MB a
// second (256 channels of 64 mels at 50 frames a second, in f32), about 1 GB at 300 s, and the language model's
// feed-forward on 4096 prompt rows about 250 MB.
const MAX_INPUT_BYTES: u64 = 64 << 20;
const MAX_AUDIO_SECONDS: u64 = 300;
const MAX_PROMPT_ROWS: u64 = 4096;
// A spoken line is a sentence or paragraph, not a document. Bound it before BPE's repeated merges,
// including a single word with no breaks. Four KiB leaves room for a long paragraph in UTF-8.
const MAX_LINE_BYTES: u64 = 4096;
// Four connections cover listening, speaking and a replacement of each. Each has at most one job out
// at a time, so a queue as long as the connections never refuses one that was let in (a listen while
// a line is being spoken).
const MAX_CLIENTS: usize = 4;
const QUEUED_JOBS: usize = MAX_CLIENTS;
// Loading on a slow disk can take a minute. Once ready, a peer that stops reading for thirty seconds
// is stalled (a line's generation budget is twenty seconds of speech). Waiting for the next line or
// for the model's answer has no bound here: a call can be quiet for minutes, and piccosta bounds the
// commands themselves (voice.rs LISTEN_FINISH_SECONDS, SPEAK_EXIT_SECONDS) and ends them at hangup.
const STARTUP_WAIT: Duration = Duration::from_secs(120);
const IDLE_WAIT: Duration = Duration::from_secs(30);
const CONNECT_POLL: Duration = Duration::from_millis(50);
// About half a second of 16 kHz mono PCM per read. A progress reply while audio arrives keeps a
// healthy, long push-to-talk from looking like a server that stopped responding before stdin ends.
const AUDIO_CHUNK_BYTES: usize = 16 << 10;
// A reply is an error byte and a little-endian u32 length, then those bytes. Even the longest default
// speech response fits the input budget; refuse a corrupt length before allocating on the client too.
const REPLY_HEADER_BYTES: usize = 5;
static SERVER_SOCKET: OnceLock<std::ffi::CString> = OnceLock::new();

// unlink and _exit are signal-safe. Do this even during model loading or inference: waiting for the
// model thread could run past the call supervisor's grace. The kernel releases the model and devices.
extern "C" fn stop_server(_: libc::c_int) {
    if let Some(socket) = SERVER_SOCKET.get() {
        unsafe {
            libc::unlink(socket.as_ptr());
        }
    }
    unsafe { libc::_exit(0) }
}

struct SocketFile(PathBuf);
impl Drop for SocketFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("lfm: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();
    let mut model = std::env::var_os("LFM_MODEL_DIR").map(PathBuf::from);
    let setting = |name: &str| std::env::var(name).ok().map(|text| (name.to_string(), text));
    let (mut threads, mut quant, mut device) = (setting("LFM_THREADS"), setting("LFM_QUANT"), setting("LFM_DEVICE"));
    let mut limits = [setting("LFM_MAX_INPUT"), setting("LFM_MAX_AUDIO"), setting("LFM_MAX_PROMPT")];
    let (mut system, mut max, mut seed, mut output, mut server, mut rest) = (None, None, None, None, None, Vec::new());
    let (mut greedy, mut raw) = (false, false);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        let number = |text: String| text.parse::<u64>().map_err(|_| format!("{arg} takes a number, not {text}"));
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--threads" => threads = Some((arg.clone(), value()?)),
            "--quant" => quant = Some((arg.clone(), value()?)),
            "--device" => device = Some((arg.clone(), value()?)),
            "--max-input" => limits[0] = Some((arg.clone(), value()?)),
            "--max-audio" => limits[1] = Some((arg.clone(), value()?)),
            "--max-prompt" => limits[2] = Some((arg.clone(), value()?)),
            "--system" => system = Some(value()?),
            "--max" => max = Some(number(value()?)? as usize),
            "--seed" => seed = Some(number(value()?)?),
            "--greedy" => greedy = true,
            "--raw" => raw = true,
            "--server" => server = Some(value()?),
            "-o" => output = Some(value()?),
            "-v" => VERBOSE.store(true, Ordering::Relaxed),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => rest.push(arg),
        }
    }
    match (command.as_str(), server) {
        ("listen" | "speak", Some(_)) if !rest.is_empty() => {
            return Err("--server reads stdin, not FILE or TEXT".into());
        }
        ("listen" | "speak", Some(socket)) => {
            return client(&socket, if raw { "listen --raw" } else { &command });
        }
        // a server says what both commands ask, so it takes neither's prompt or budget
        ("serve", None) if system.is_some() || max.is_some() => {
            return Err("lfm serve listens and speaks, so it takes no --system or --max".into());
        }
        ("listen" | "speak" | "decode" | "serve", None) => {}
        _ => return Err(USAGE.into()),
    }
    // a flag's or variable's whole number, at least one: rayon would read a zero thread count as "its own choice"
    let positive = |setting: Option<(String, String)>, default: u64| match setting {
        None => Ok(default),
        Some((name, text)) => {
            text.parse().ok().filter(|&n| n > 0).ok_or(format!("{name} takes a positive number, not {text}"))
        }
    };
    let [max_input, max_audio, max_prompt] = limits;
    let (max_input, max_audio) = (positive(max_input, MAX_INPUT_BYTES)?, positive(max_audio, MAX_AUDIO_SECONDS)?);
    let max_prompt = positive(max_prompt, MAX_PROMPT_ROWS)? as usize;
    // SMT siblings share a core's multiply units and memory bandwidth, so a thread per physical core does as
    // well as one per logical CPU; half the logical CPUs is that on the usual two-way machine
    let halved = std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1));
    let threads = positive(threads, halved as u64)? as usize;
    let quant = quant.map_or(Ok(None), |(name, text)| Format::parse(&text).map_err(|e| format!("{name}: {e}")))?;
    let (device_setting, device) = device.unwrap_or(("--device".into(), "cpu".into()));
    let on_device = |model: &mut Model| model.set_device(&device).map_err(|e| format!("{device_setting}: {e}"));
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().map_err(|e| e.to_string())?;
    let directory = model.ok_or("no model: pass --model DIR or set LFM_MODEL_DIR")?;
    let start = Instant::now();
    // Every load below maps the checkpoint, which must not change while lfm runs (Weights::open). lfm can't make
    // that so for other processes: it is the operator's precondition, which the usage text states. lfm itself
    // writes the -o file, which write_output refuses when it is, or links to, a mapped file, and a quantized
    // file only under a new name, renamed into place (quant::open).
    if command == "decode" {
        let parse = |line: &str| {
            line.split_whitespace().map(|code| code.parse().map_err(|_| format!("bad code {code}"))).collect()
        };
        let frames = text_input(rest.first(), max_input)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(parse)
            .collect::<Result<Vec<_>>>()?;
        let max_frames = max.unwrap_or(SPEAK_MAX_FRAMES);
        if frames.len() > max_frames {
            return Err(format!("{} frames, over the {max_frames} that --max allows", frames.len()));
        }
        // SAFETY: see above
        let codec = unsafe { Codec::load(&directory)? };
        let (name, mut out) = write_output(output)?;
        return out
            .write_all(&audio::write_wav(&codec.decode(&frames)?, codec.rate))
            .map_err(|e| format!("{name}: {e}"));
    }
    // A server's socket is bound before the model loads, so a client that connects meanwhile waits for it. One
    // left behind by a server that was killed is replaced.
    let listener = match (command.as_str(), rest.first()) {
        ("serve", None) => return Err("lfm serve needs a SOCKET".into()),
        ("serve", Some(socket)) => {
            let _ = std::fs::remove_file(socket);
            let listener = UnixListener::bind(socket).map_err(|e| format!("{socket}: {e}"))?;
            let socket_file = SocketFile(socket.into());
            SERVER_SOCKET
                .set(std::ffi::CString::new(socket.as_bytes()).map_err(|e| e.to_string())?)
                .map_err(|_| "socket already bound")?;
            for signal in [libc::SIGTERM, libc::SIGINT] {
                // SAFETY: the handler only uses signal-safe calls and an initialized, immutable path.
                if unsafe { libc::signal(signal, stop_server as *const () as libc::sighandler_t) } == libc::SIG_ERR {
                    return Err(std::io::Error::last_os_error().to_string());
                }
            }
            Some((listener, socket_file))
        }
        _ => None,
    };
    // Loaded before the speech or the text is read, so the load runs while the user is still talking into a
    // microphone piped in, or the model is still writing what's to be said.
    // SAFETY: see above
    let mut model = unsafe { Model::load(&directory, quant)? };
    model.max_prompt = max_prompt;
    on_device(&mut model)?;
    // SAFETY: see above
    let codec = if command == "listen" { None } else { Some(unsafe { Codec::load(&directory)? }) };
    let rate = codec.as_ref().map_or(0, |codec| codec.rate);
    note(&format!("loaded, {threads} threads"), start);
    let clock =
        || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut sampler = model.sampler(seed.unwrap_or_else(clock));
    if greedy {
        sampler.temperature = 0.0;
    }
    // What the model does, for this command or for a server's clients one at a time: speech to the transcript's
    // line, or text to its speech as 16-bit PCM.
    let mut work = |job: Job| -> Result<Vec<u8>> {
        match job {
            Job::Hear(mut input, raw, streamed) => {
                if raw {
                    // The raw input is complete now, so its synthetic WAV header has a known length.
                    let header = audio::wav_header(model.mel.rate, input.len() as u32);
                    input.splice(0..0, header);
                }
                let (samples, rate) = audio::read_wav(&input, streamed)?;
                if samples.len() as u64 > max_audio.saturating_mul(rate as u64) {
                    return Err(format!("the speech is over {max_audio} s, the budget --max-audio sets"));
                }
                if rate != model.mel.rate {
                    let wanted = model.mel.rate;
                    return Err(format!("the audio is {rate} Hz; resample it to {wanted} Hz (ffmpeg -ar {wanted})"));
                }
                let system = system.as_deref().unwrap_or(LISTEN_PROMPT);
                let text = model.transcribe(&samples, system, max.unwrap_or(LISTEN_MAX_TOKENS))?;
                note(&format!("{:.2} s of speech heard", samples.len() as f64 / rate as f64), start);
                Ok(format!("{text}\n").into_bytes())
            }
            Job::Say(text) => {
                if text.len() as u64 > MAX_LINE_BYTES {
                    return Err(format!("spoken line is over {MAX_LINE_BYTES} bytes"));
                }
                let system = system.as_deref().unwrap_or(SPEAK_PROMPT);
                let frames = model.speak(&text, system, max.unwrap_or(SPEAK_MAX_FRAMES), &mut sampler)?;
                lfm::tensor::dump("frames", &frames.iter().flatten().map(|&code| code as f32).collect::<Vec<_>>());
                let decoding = Instant::now();
                let samples = codec.as_ref().ok_or("no codec")?.decode(&frames)?;
                note("codec", decoding);
                note(&format!("{:.2} s of speech spoken", samples.len() as f64 / rate as f64), start);
                Ok(audio::pcm(&samples))
            }
        }
    };
    match (command.as_str(), listener) {
        ("listen", _) => {
            let input = read_input(rest.first(), max_input)?;
            print!("{}", String::from_utf8_lossy(&work(Job::Hear(input, raw, rest.is_empty()))?));
        }
        ("speak", _) if !rest.is_empty() => {
            let pcm = work(Job::Say(rest.join(" ").trim().into()))?;
            let (name, mut out) = write_output(output)?;
            out.write_all(&[audio::wav_header(rate, pcm.len() as u32), pcm].concat())
                .map_err(|e| format!("{name}: {e}"))?;
        }
        ("speak", _) => {
            let (name, mut out) = write_output(output)?;
            say_lines(
                std::io::stdin().lock(),
                rate,
                max_input,
                |text| work(Job::Say(text.into())),
                |result| match result {
                    Ok(bytes) => out.write_all(&bytes).and_then(|()| out.flush()).map_err(|e| format!("{name}: {e}")),
                    Err(error) => {
                        eprintln!("lfm: {error}");
                        Ok(())
                    }
                },
            )?;
        }
        (_, Some((listener, _socket_file))) => {
            let (jobs, queue) = mpsc::sync_channel::<(Job, Reply)>(QUEUED_JOBS);
            std::thread::spawn(move || {
                let active = Arc::new(AtomicUsize::new(0));
                for mut client in listener.incoming().flatten() {
                    // Until its request line arrives, a peer has IDLE_WAIT (serve clears it then),
                    // so one that never asks can't hold a slot.
                    if client.set_read_timeout(Some(IDLE_WAIT)).and_then(|()| client.set_write_timeout(Some(IDLE_WAIT))).is_err() {
                        continue;
                    }
                    if active.load(Ordering::Relaxed) >= MAX_CLIENTS {
                        let _ = send_reply(&mut client, Err(format!("server busy: at most {MAX_CLIENTS} connections")));
                        continue;
                    }
                    active.fetch_add(1, Ordering::Relaxed);
                    let (jobs, active) = (jobs.clone(), active.clone());
                    std::thread::spawn(move || {
                        if let Err(error) = serve(&mut client, jobs, rate, max_input) {
                            let _ = send_reply(&mut client, Err(error.clone()));
                            eprintln!("lfm: {error}");
                        }
                        active.fetch_sub(1, Ordering::Relaxed);
                    });
                }
            });
            for (job, reply) in queue {
                let _ = reply.send(work(job));
            }
        }
        _ => unreachable!(),
    }
    if VERBOSE.load(Ordering::Relaxed) {
        eprintln!("lfm: memory: {}", lfm::memory());
    }
    Ok(())
}

// A job for the model: speech to hear (a WAV, or with true bare 16-bit samples), or a line of text to say.
enum Job {
    Hear(Vec<u8>, bool, bool),
    Say(String),
}

// where the model thread sends a server's client what came of its job
type Reply = mpsc::Sender<Result<Vec<u8>>>;

// One continuous WAV, with each line bounded before tokenization or allocation of the rest of it.
// Discard an oversized line through its newline, report it, and still say the next one.
fn say_lines(
    input: impl BufRead,
    rate: u32,
    limit: u64,
    mut say: impl FnMut(&str) -> Result<Vec<u8>>,
    mut put: impl FnMut(Result<Vec<u8>>) -> Result<()>,
) -> Result<()> {
    put(Ok(audio::wav_header(rate, u32::MAX)))?;
    let (mut input, mut read) = (input.take(limit.saturating_add(1)), 0);
    loop {
        let mut line = Vec::new();
        let count = (&mut input).take(MAX_LINE_BYTES + 1).read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
        if count == 0 {
            return Ok(());
        }
        read += count as u64;
        let oversized = line.len() as u64 > MAX_LINE_BYTES;
        if oversized && !line.ends_with(b"\n") {
            loop {
                let rest = input.fill_buf().map_err(|e| e.to_string())?;
                let newline = rest.iter().position(|&b| b == b'\n');
                let count = newline.map_or(rest.len(), |at| at + 1);
                input.consume(count);
                read += count as u64;
                if newline.is_some() || count == 0 {
                    break;
                }
            }
        }
        if read > limit {
            return Err(format!("input is over {limit} bytes, the budget --max-input sets"));
        }
        if oversized {
            put(Err(format!("spoken line is over {MAX_LINE_BYTES} bytes")))?;
        } else {
            match String::from_utf8(line) {
                Ok(line) if !line.trim().is_empty() => put(say(line.trim()))?,
                Err(_) => put(Err("the text is not UTF-8".into()))?,
                _ => {}
            }
        }
    }
}

fn send_reply(out: &mut UnixStream, result: Result<Vec<u8>>) -> Result<()> {
    let (error, bytes) = match result {
        Ok(bytes) => (0, bytes),
        Err(error) => (1, error.into_bytes()),
    };
    out.write_all(&[error])
        .and_then(|()| out.write_all(&(bytes.len() as u32).to_le_bytes()))
        .and_then(|()| out.write_all(&bytes))
        .map_err(|e| format!("the client: {e}"))
}

// The initial empty reply means the model is ready. Every connection has one outstanding job,
// a bounded input, and socket deadlines; a full model queue refuses work instead of retaining it.
fn serve(client: &mut UnixStream, jobs: mpsc::SyncSender<(Job, Reply)>, rate: u32, limit: u64) -> Result<()> {
    send_reply(client, Ok(Vec::new()))?;
    let ask = |job| {
        let (reply, answer) = mpsc::channel();
        jobs.try_send((job, reply)).map_err(|e| format!("model queue unavailable: {e}"))?;
        answer.recv().map_err(|e| format!("the model: {e}"))?
    };
    let mut input = BufReader::new(client.try_clone().map_err(|e| e.to_string())?);
    let mut request = String::new();
    (&mut input).take(MAX_LINE_BYTES).read_line(&mut request).map_err(|e| format!("the client: {e}"))?;
    client.set_read_timeout(None).map_err(|e| format!("the client: {e}"))?;
    match request.trim() {
        "speak" => {
            say_lines(input, rate, limit, |text| ask(Job::Say(text.into())), |result| send_reply(client, result))
        }
        hear @ ("listen" | "listen --raw") => {
            let (mut speech, mut chunk) = (Vec::new(), [0; AUDIO_CHUNK_BYTES]);
            loop {
                let left = limit.saturating_sub(speech.len() as u64).saturating_add(1);
                let count = (&mut input).take(left).read(&mut chunk).map_err(|e| format!("the client: {e}"))?;
                if count == 0 {
                    break;
                }
                if speech.len() as u64 + count as u64 > limit {
                    return Err(format!("the client is over {limit} bytes"));
                }
                speech.extend_from_slice(&chunk[..count]);
                send_reply(client, Ok(Vec::new()))?;
            }
            send_reply(client, ask(Job::Hear(speech, hear == "listen --raw", true)))
        }
        other => Err(format!("a client asked for {other:?}, not listen, listen --raw or speak")),
    }
}

fn receive_reply(server: &mut UnixStream) -> Result<Option<Result<Vec<u8>>>> {
    let mut header = [0; REPLY_HEADER_BYTES];
    if server.read(&mut header[..1]).map_err(|e| format!("server read: {e}"))? == 0 {
        return Ok(None);
    }
    server.read_exact(&mut header[1..]).map_err(|e| format!("server header: {e}"))?;
    let size = u32::from_le_bytes(header[1..].try_into().unwrap()) as u64;
    if size > MAX_INPUT_BYTES {
        return Err("server reply exceeds the byte budget".into());
    }
    let mut bytes = vec![0; size as usize];
    server.read_exact(&mut bytes).map_err(|e| format!("server reply: {e}"))?;
    Ok(Some(if header[0] == 0 { Ok(bytes) } else { Err(String::from_utf8_lossy(&bytes).into()) }))
}

// Connect without ever waiting in a full accept backlog. Startup has one deadline, including the
// ready reply; after it, only a server that stops reading has a bound (IDLE_WAIT), since waiting for
// audio while the model is quiet is a normal part of a call.
fn client(socket: &str, request: &str) -> Result<()> {
    let fail = |e: std::io::Error| format!("{socket}: {e}");
    let deadline = Instant::now() + STARTUP_WAIT;
    let mut server: UnixStream = loop {
        let stream = socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None).map_err(fail)?;
        stream.set_nonblocking(true).map_err(fail)?;
        match stream.connect(&socket2::SockAddr::unix(socket).map_err(fail)?) {
            Ok(()) => {
                stream.set_nonblocking(false).map_err(fail)?;
                break stream.into();
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::NotFound
                        | std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::WouldBlock
                ) =>
            {
                if Instant::now() >= deadline {
                    return Err(format!("{socket}: server startup timed out"));
                }
                std::thread::sleep(CONNECT_POLL);
            }
            Err(e) => return Err(fail(e)),
        }
    };
    server
        .set_read_timeout(Some(deadline.saturating_duration_since(Instant::now()).max(CONNECT_POLL)))
        .map_err(fail)?;
    receive_reply(&mut server)?.ok_or("server closed before ready")??;
    server.set_read_timeout(None).and_then(|()| server.set_write_timeout(Some(IDLE_WAIT))).map_err(fail)?;
    server.write_all(format!("{request}\n").as_bytes()).map_err(fail)?;
    let mut sending = server.try_clone().map_err(fail)?;
    std::thread::spawn(move || {
        if let Err(error) = std::io::copy(&mut std::io::stdin().lock(), &mut sending) {
            eprintln!("lfm: server write: {error}");
        }
        let _ = sending.shutdown(Shutdown::Write);
    });
    let (mut out, mut refused) = (std::io::stdout().lock(), false);
    while let Some(reply) = receive_reply(&mut server)? {
        match reply {
            Ok(bytes) => out.write_all(&bytes).and_then(|()| out.flush()).map_err(|e| format!("stdout: {e}"))?,
            Err(error) => {
                eprintln!("lfm: {error}");
                refused = true;
            }
        }
    }
    if refused { Err("server refused input".into()) } else { Ok(()) }
}

// The bytes of FILE, or of stdin when there is none, refused past `limit` (read_limited).
fn read_input(file: Option<&String>, limit: u64) -> Result<Vec<u8>> {
    match file {
        Some(path) => read_limited(path, std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?, limit),
        None => read_limited("stdin", std::io::stdin(), limit),
    }
}

// The bytes of `source`, refused once they pass `limit`: reading stops one byte past it, so an endless stream
// costs no more than that.
fn read_limited(name: &str, source: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    source.take(limit.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| format!("{name}: {e}"))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{name} is over {limit} bytes, the budget --max-input sets"));
    }
    Ok(bytes)
}

fn text_input(file: Option<&String>, limit: u64) -> Result<String> {
    String::from_utf8(read_input(file, limit)?).map_err(|_| "the text is not UTF-8".into())
}

// Where the WAV goes, and the name its errors give: FILE, or stdout when there is none. FILE is opened before it
// is truncated, and refused if it is a file the weights are mapped from, under any name or link, which
// truncating would pull out from under the model.
fn write_output(file: Option<String>) -> Result<(String, Box<dyn Write>)> {
    let Some(path) = file else {
        return Ok(("stdout".into(), Box::new(std::io::stdout().lock())));
    };
    let fail = |e: std::io::Error| format!("{path}: {e}");
    let out = std::fs::OpenOptions::new().write(true).create(true).truncate(false).open(&path).map_err(fail)?;
    if lfm::tensor::is_mapped(&out.metadata().map_err(fail)?) {
        return Err(format!("{path} is a checkpoint file lfm is reading; write the WAV somewhere else"));
    }
    out.set_len(0).map_err(fail)?;
    Ok((path, Box::new(out)))
}
