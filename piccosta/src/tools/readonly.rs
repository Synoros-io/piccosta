// The read-only sandbox a read-only subagent job's commands run in (scale.json readOnly,
// tools/subagent.rs): Linux Landlock, which the kernel enforces on the shell and everything it
// starts. A command can read the whole computer and run anything on it, but can't create, change
// or remove a file, can't truncate one, and can't signal a process outside its sandbox, so a job
// told to watch a run can't kill it. Of /dev, only the devices commands write to as a matter of
// course stay writable (DEVICES). A kernel whose Landlock is older than ABI 6, the first with
// all of that, can't run such a command at all: it fails, rather than running with rights the
// user's scale said the job doesn't get.
//
// It guards against a model's careless writes and deletes, not against a model trying to escape:
// a command can still talk to sockets that are already there (tmux's, systemd's) and change a
// file's mode, owner or times, which Landlock doesn't cover.
//
// The numbers are the kernel's (include/uapi/linux/landlock.h); libc has the syscall numbers only.

use std::io;

// The rights to files that change something: write a file, then remove a directory or file and
// make each kind of node (bits 4 to 12), all in ABI 1; then refer (moving a file between
// directories, ABI 2) and truncate (ABI 3).
const WRITE_FILE: u64 = 1 << 1;
const REMOVE_AND_MAKE: u64 = 0b1_1111_1111 << 4;
const REFER: u64 = 1 << 13;
const TRUNCATE: u64 = 1 << 14;
// Signals only to processes in the same sandbox, from ABI 6, the oldest ABI this needs.
const SCOPE_SIGNAL: u64 = 1 << 1;
const NEEDED_ABI: libc::c_long = 6;
// The devices a command may write to: the sinks and sources shells use, and its terminal.
const DEVICES: [&std::ffi::CStr; 6] = [c"/dev/null", c"/dev/zero", c"/dev/full", c"/dev/random", c"/dev/urandom", c"/dev/tty"];
// landlock_create_ruleset's flag that asks for the ABI version instead of making a ruleset.
const CREATE_RULESET_VERSION: libc::c_uint = 1;
// The rule kind that grants rights beneath a directory.
const RULE_PATH_BENEATH: libc::c_int = 1;

#[repr(C)]
struct RulesetAttr {
    handled_access_fs: u64,
    handled_access_net: u64,
    scoped: u64,
}

#[repr(C, packed)]
struct PathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

// Whether this kernel can run a read-only job's commands. Run in the parent, before the fork,
// so the error can say why. Output: an error naming the ABI it has and the one needed.
pub fn check() -> Result<(), String> {
    // SAFETY: the version query passes no pointer.
    let abi = unsafe { libc::syscall(libc::SYS_landlock_create_ruleset, std::ptr::null::<RulesetAttr>(), 0usize, CREATE_RULESET_VERSION) };
    if abi < NEEDED_ABI {
        let found = if abi < 0 { "no Landlock".to_string() } else { format!("Landlock ABI {abi}") };
        return Err(format!("This kernel has {found}, and a read-only job needs ABI {NEEDED_ABI} (Linux 6.12) to keep its commands from truncating files and signalling processes, so they don't run. Give the job \"readOnly\": false in scale.json to run them unsandboxed."));
    }
    Ok(())
}

// Put this process (a forked shell, before it execs) in the read-only sandbox. It only makes raw
// system calls, so it's safe between fork and exec. Output: an error when the kernel refuses the
// ruleset, as one older than NEEDED_ABI does.
pub fn enter() -> io::Result<()> {
    let failed = |result: libc::c_long| if result < 0 { Err(io::Error::last_os_error()) } else { Ok(result) };
    // SAFETY: each call passes the kernel a pointer to a live struct of the size it's told; the
    // descriptors are this process's own.
    unsafe {
        let attr = RulesetAttr { handled_access_fs: WRITE_FILE | REMOVE_AND_MAKE | REFER | TRUNCATE, handled_access_net: 0, scoped: SCOPE_SIGNAL };
        let ruleset = failed(libc::syscall(libc::SYS_landlock_create_ruleset, &attr, size_of::<RulesetAttr>(), 0))? as libc::c_int;
        // Writing to those devices stays allowed; one this machine doesn't have is skipped.
        for device in DEVICES {
            let Ok(file) = failed(libc::open(device.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) as libc::c_long) else { continue };
            let rule = PathBeneathAttr { allowed_access: WRITE_FILE | TRUNCATE, parent_fd: file as libc::c_int };
            failed(libc::syscall(libc::SYS_landlock_add_rule, ruleset, RULE_PATH_BENEATH, &rule, 0))?;
            libc::close(file as libc::c_int);
        }
        // The kernel takes a sandbox only from a process that can't gain rights by exec (setuid).
        failed(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) as libc::c_long)?;
        failed(libc::syscall(libc::SYS_landlock_restrict_self, ruleset, 0))?;
        libc::close(ruleset);
    }
    Ok(())
}
