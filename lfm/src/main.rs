// lfm: LFM3-Audio-2B as the two commands of piccosta's voice mode, speech in and text out (`listen`), text in
// and speech out (`speak`), through stdin and stdout.

use lfm::{VERBOSE, audio, codec::Codec, model::Model, note, quant::Format, tensor::Result};
use std::{
    io::{Read, Write},
    path::PathBuf,
    process::ExitCode,
    sync::atomic::Ordering,
    time::Instant,
};

const USAGE: &str = "\
usage: lfm listen [options] [FILE]   transcribe 16 kHz WAV speech (FILE, or stdin) to stdout
       lfm speak [options] [TEXT]    speak TEXT (or stdin) as a 24 kHz WAV on stdout
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
    let (mut system, mut max, mut seed, mut output, mut greedy, mut rest) = (None, None, None, None, false, Vec::new());
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
            "-o" => output = Some(value()?),
            "-v" => VERBOSE.store(true, Ordering::Relaxed),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => rest.push(arg),
        }
    }
    if !matches!(command.as_str(), "listen" | "speak" | "decode") {
        return Err(USAGE.into());
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
    match command.as_str() {
        "listen" => {
            let (samples, rate) = audio::read_wav(&read_input(rest.first(), max_input)?)?;
            if samples.len() as u64 > max_audio.saturating_mul(rate as u64) {
                return Err(format!("the speech is over {max_audio} s, the budget --max-audio sets"));
            }
            // SAFETY: see above
            let mut model = unsafe { Model::load(&directory, quant)? };
            model.max_prompt = max_prompt;
            on_device(&mut model)?;
            note(&format!("loaded, {threads} threads"), start);
            if rate != model.mel.rate {
                return Err(format!("the audio is {rate} Hz; resample it to {} Hz (ffmpeg -ar {0})", model.mel.rate));
            }
            let system = system.as_deref().unwrap_or(LISTEN_PROMPT);
            println!("{}", model.transcribe(&samples, system, max.unwrap_or(LISTEN_MAX_TOKENS))?);
            note(&format!("{:.2} s of speech heard", samples.len() as f64 / rate as f64), start);
        }
        "speak" => {
            let text = if rest.is_empty() { text_input(None, max_input)? } else { rest.join(" ") };
            // SAFETY: see above
            let (mut model, codec) = unsafe { (Model::load(&directory, quant)?, Codec::load(&directory)?) };
            model.max_prompt = max_prompt;
            on_device(&mut model)?;
            note(&format!("loaded, {threads} threads"), start);
            let clock = || {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
            };
            let mut sampler = model.sampler(seed.unwrap_or_else(clock));
            if greedy {
                sampler.temperature = 0.0;
            }
            let frames = model.speak(
                text.trim(),
                system.as_deref().unwrap_or(SPEAK_PROMPT),
                max.unwrap_or(SPEAK_MAX_FRAMES),
                &mut sampler,
            )?;
            lfm::tensor::dump("frames", &frames.iter().flatten().map(|&code| code as f32).collect::<Vec<_>>());
            let decoding = Instant::now();
            let samples = codec.decode(&frames)?;
            note("codec", decoding);
            write_output(output, &audio::write_wav(&samples, codec.rate))?;
            note(&format!("{:.2} s of speech spoken", samples.len() as f64 / codec.rate as f64), start);
        }
        _ => {
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
            write_output(output, &audio::write_wav(&codec.decode(&frames)?, codec.rate))?;
        }
    }
    if VERBOSE.load(Ordering::Relaxed) {
        eprintln!("lfm: memory: {}", lfm::memory());
    }
    Ok(())
}

// The bytes of FILE, or of stdin when there is none, refused once they pass `limit`: reading stops one byte
// past it, so an endless stream costs no more than that.
fn read_input(file: Option<&String>, limit: u64) -> Result<Vec<u8>> {
    let (name, source): (&str, Box<dyn Read>) = match file {
        Some(path) => (path, Box::new(std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?)),
        None => ("stdin", Box::new(std::io::stdin())),
    };
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

// Writes `bytes` to FILE, or to stdout when there is none. FILE is opened before it is truncated, and refused
// if it is a file the weights are mapped from, under any name or link, which truncating would pull out from
// under the model.
fn write_output(file: Option<String>, bytes: &[u8]) -> Result<()> {
    let Some(path) = file else {
        return std::io::stdout().lock().write_all(bytes).map_err(|e| format!("stdout: {e}"));
    };
    let fail = |e: std::io::Error| format!("{path}: {e}");
    let mut out = std::fs::OpenOptions::new().write(true).create(true).truncate(false).open(&path).map_err(fail)?;
    if lfm::tensor::is_mapped(&out.metadata().map_err(fail)?) {
        return Err(format!("{path} is a checkpoint file lfm is reading; write the WAV somewhere else"));
    }
    out.set_len(0).and_then(|()| out.write_all(bytes)).map_err(fail)
}
