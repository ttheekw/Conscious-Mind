//! The persistent timeline: every event, appended to one file.
//!
//! This is the harness's long-term record of what happened. Each event is
//! written the moment it occurs, one JSON object per line, and never
//! rewritten. On start the most recent events are read back so the session
//! continues where the last one stopped instead of starting blank.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::event::EventKind;
use crate::learn::now;

/// Events read back on start. Older ones stay in the file.
const MAX_RESTORED: usize = 400;

/// One line of the timeline file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recorded {
    /// Unix seconds when it happened.
    pub wall: u64,
    /// The task it belongs to; none for events about the harness itself,
    /// and for everything recorded before tasks existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<u32>,
    pub kind: EventKind,
}

pub struct Log {
    file: Option<File>,
}

/// Where the timeline lives: beside the routing knowledge.
pub fn default_path() -> Option<PathBuf> {
    Some(
        crate::learn::default_path()?
            .parent()?
            .join("timeline.jsonl"),
    )
}

impl Log {
    /// A log that records nothing, for the demo and tests.
    pub fn disabled() -> Self {
        Self { file: None }
    }

    /// Opens the timeline for appending and returns its most recent events,
    /// oldest first. Lines that cannot be read (a torn final write, an event
    /// kind from another version) are skipped rather than failing the start.
    pub fn open(path: &Path) -> (Self, Vec<Recorded>, Option<String>) {
        let mut recent: VecDeque<Recorded> = VecDeque::new();
        if let Ok(file) = File::open(path) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                if let Ok(event) = serde_json::from_str::<Recorded>(&line) {
                    if recent.len() == MAX_RESTORED {
                        recent.pop_front();
                    }
                    recent.push_back(event);
                }
            }
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(mut file) => {
                // If the last write was cut short, end that line, so the
                // next event starts a line of its own.
                let torn = std::fs::read(path).is_ok_and(|b| b.last().is_some_and(|c| *c != b'\n'));
                if torn {
                    let _ = file.write_all(b"\n");
                }
                (Self { file: Some(file) }, recent.into(), None)
            }
            Err(e) => (
                Self::disabled(),
                recent.into(),
                Some(format!(
                    "{}: {e}; this session will not be remembered",
                    path.display()
                )),
            ),
        }
    }

    /// Appends one event. A failed write loses that event from the record
    /// but never stops the session.
    pub fn append(&mut self, task: Option<u32>, kind: &EventKind) {
        let Some(file) = &mut self.file else {
            return;
        };
        let record = Recorded {
            wall: now(),
            task,
            kind: kind.clone(),
        };
        if let Ok(mut line) = serde_json::to_string(&record) {
            line.push('\n');
            let _ = file.write_all(line.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Severity;

    fn alert(text: &str) -> EventKind {
        EventKind::Alert {
            severity: Severity::Info,
            text: text.into(),
        }
    }

    #[test]
    fn events_survive_a_restart_and_bad_lines_are_skipped() {
        let dir =
            std::env::temp_dir().join(format!("conscious-timeline-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("timeline.jsonl");

        let (mut log, restored, note) = Log::open(&path);
        assert!(restored.is_empty() && note.is_none());
        log.append(None, &alert("first"));
        log.append(
            None,
            &EventKind::Message {
                agent: None,
                text: "hello".into(),
            },
        );
        drop(log);
        // A torn final write, as after a crash.
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"wall\": 12, \"kind\": {\"Alert\"")
            .unwrap();
        drop(file);

        let (mut log, restored, _) = Log::open(&path);
        assert_eq!(restored.len(), 2);
        assert!(matches!(&restored[0].kind, EventKind::Alert { text, .. } if text == "first"));
        assert!(matches!(&restored[1].kind, EventKind::Message { text, .. } if text == "hello"));
        assert!(restored[0].wall > 1_700_000_000);

        // Appending continues the same file.
        log.append(None, &alert("third"));
        drop(log);
        assert_eq!(Log::open(&path).1.len(), 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_the_most_recent_events_are_read_back() {
        let dir =
            std::env::temp_dir().join(format!("conscious-timeline-cap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("timeline.jsonl");
        let (mut log, _, _) = Log::open(&path);
        for i in 0..MAX_RESTORED + 25 {
            log.append(None, &alert(&format!("event {i}")));
        }
        drop(log);
        let (_, restored, _) = Log::open(&path);
        assert_eq!(restored.len(), MAX_RESTORED);
        assert!(matches!(&restored[0].kind, EventKind::Alert { text, .. } if text == "event 25"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
