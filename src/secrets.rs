//! Secrets the user has given the harness: API keys, tokens, passwords.
//!
//! The rule this module exists for: **a secret's value never reaches a
//! model.** It is typed into a window of its own on the screen, goes from
//! there straight to this store, and is not an event, not in the timeline,
//! not in any context and not in a log. A model only ever knows a secret's
//! *name*. It uses one by writing `$NAME` in a shell command; the harness
//! puts the value in that command's environment (`Guard::shell`), asks the
//! user first every time (`Guard::check`), and replaces the value with
//! `[secret NAME]` wherever it turns up in output (`redact`).
//!
//! Kept in `secrets.json` beside the routing knowledge, readable by the
//! user only (mode 0600), as plain text: the same protection as `~/.netrc`
//! or an SSH key without a passphrase, and no more. The file is hidden from
//! sandboxed commands and never named by an allowed one.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// The longest a secret's name and value may be.
const MAX_NAME: usize = 64;
const MAX_VALUE: usize = 8_192;
/// A value shorter than this is not looked for in output: it would match
/// by accident all over ordinary text.
const MIN_REDACTED: usize = 6;

/// A value that must not be printed: `Debug` shows nothing of it. Used for
/// the one hop a secret makes, from the screen to the store.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Hidden(pub String);

impl std::fmt::Debug for Hidden {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Hidden(***)")
    }
}

#[derive(Default)]
pub struct Secrets {
    /// Where the secrets are saved; `None` keeps them in memory only.
    path: Option<PathBuf>,
    values: BTreeMap<String, String>,
}

/// Whether `name` can name a secret: like an environment variable in
/// capitals, since that is how a command is given it.
pub fn valid_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.len() <= MAX_NAME
        && name.starts_with(|c: char| c.is_ascii_uppercase())
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    if ok {
        Ok(())
    } else {
        Err(format!(
            "a secret's name is capitals, digits and underscores, starting with a letter, \
             at most {MAX_NAME} long (like GITHUB_TOKEN)"
        ))
    }
}

impl Secrets {
    /// The secrets saved at `path`, or none if there is no such file yet.
    pub fn open(path: PathBuf) -> Self {
        let values = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self {
            path: Some(path),
            values,
        }
    }

    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Keeps `value` under `name`, replacing what was there.
    pub fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
        valid_name(name)?;
        let value = value.trim();
        if value.is_empty() {
            return Err("the secret is empty".into());
        }
        if value.len() > MAX_VALUE {
            return Err(format!("a secret is at most {MAX_VALUE} characters"));
        }
        self.values.insert(name.to_string(), value.to_string());
        self.save()
    }

    /// Forgets `name`. Says whether there was one.
    pub fn remove(&mut self, name: &str) -> Result<bool, String> {
        let was = self.values.remove(name).is_some();
        if was {
            self.save()?;
        }
        Ok(was)
    }

    pub fn names(&self) -> Vec<String> {
        self.values.keys().cloned().collect()
    }

    pub fn get(&self, name: &str) -> Option<String> {
        self.values.get(name).cloned()
    }

    /// The secrets a shell command refers to, as `$NAME` or `${NAME}`.
    pub fn named_in(&self, command: &str) -> Vec<String> {
        self.values
            .keys()
            .filter(|name| {
                let braced = format!("${{{name}}}");
                if command.contains(&braced) {
                    return true;
                }
                let plain = format!("${name}");
                command.match_indices(&plain).any(|(at, _)| {
                    // `$TOKEN` is not `$TOKEN_2`.
                    !command[at + plain.len()..]
                        .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                })
            })
            .cloned()
            .collect()
    }

    /// `text` with every stored value replaced by `[secret NAME]`.
    pub fn redact(&self, text: &str) -> String {
        let mut known: Vec<(&String, &String)> = self
            .values
            .iter()
            .filter(|(_, v)| v.len() >= MIN_REDACTED)
            .collect();
        if known.is_empty() || !known.iter().any(|(_, v)| text.contains(v.as_str())) {
            return text.to_string();
        }
        // The longest first, so one value inside another is not left half
        // showing.
        known.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
        let mut out = text.to_string();
        for (name, value) in known {
            out = out.replace(value.as_str(), &format!("[secret {name}]"));
        }
        out
    }

    /// Writes the file whole, readable by the user only, beside the old one
    /// and then into its place, so it is never half-written or briefly
    /// readable by others.
    fn save(&self) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let text = serde_json::to_string_pretty(&self.values).map_err(|e| e.to_string())?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let fresh = path.with_extension("json.new");
        let _ = std::fs::remove_file(&fresh);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&fresh)
            .map_err(|e| format!("{}: {e}", fresh.display()))?;
        file.write_all(text.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|e| format!("{}: {e}", fresh.display()))?;
        std::fs::rename(&fresh, path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Where the secrets are kept: beside the routing knowledge.
pub fn path() -> Option<PathBuf> {
    Some(crate::learn::default_path()?.parent()?.join("secrets.json"))
}

/// The one store of this process. In tests it is in memory, so no test
/// reads or writes the real one.
pub fn store() -> &'static Mutex<Secrets> {
    static STORE: OnceLock<Mutex<Secrets>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(if cfg!(test) {
            Secrets::in_memory()
        } else {
            path().map(Secrets::open).unwrap_or_default()
        })
    })
}

/// `text` with every stored secret's value replaced by its name.
pub fn redact(text: &str) -> String {
    store().lock().unwrap().redact(text)
}

/// A stored secret, for the harness's own use (a model backend's key, the
/// Telegram bot's token). Never for anything a model reads.
pub fn get(name: &str) -> Option<String> {
    store().lock().unwrap().get(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_are_named_like_environment_variables() {
        for good in ["GITHUB_TOKEN", "A", "KEY_2"] {
            assert!(valid_name(good).is_ok(), "{good}");
        }
        for bad in [
            "",
            "token",
            "2KEY",
            "MY-KEY",
            "MY KEY",
            "_X",
            &"A".repeat(65),
        ] {
            assert!(valid_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_secret_is_saved_for_the_user_only_and_read_back() {
        let dir = std::env::temp_dir().join(format!("conscious-secrets-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("secrets.json");
        let mut secrets = Secrets::open(file.clone());
        assert!(secrets.names().is_empty());
        secrets.set("DEPLOY_TOKEN", "  tok-12345678  ").unwrap();
        assert!(secrets.set("lower", "x").is_err());
        assert!(secrets.set("EMPTY", "   ").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&file).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        // A later start finds it, trimmed.
        let again = Secrets::open(file.clone());
        assert_eq!(again.names(), ["DEPLOY_TOKEN"]);
        assert_eq!(again.get("DEPLOY_TOKEN").as_deref(), Some("tok-12345678"));
        // Forgetting it takes it out of the file.
        let mut again = again;
        assert!(again.remove("DEPLOY_TOKEN").unwrap());
        assert!(!again.remove("DEPLOY_TOKEN").unwrap());
        assert!(Secrets::open(file).names().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_command_names_the_secrets_it_uses_and_output_never_shows_their_values() {
        let mut secrets = Secrets::in_memory();
        secrets.set("API_TOKEN", "tok-abcdef123456").unwrap();
        secrets.set("API_TOKEN_2", "other-998877").unwrap();
        secrets.set("PIN", "1234").unwrap();
        assert_eq!(
            secrets.named_in("curl -H \"Authorization: Bearer $API_TOKEN\" https://x.test"),
            ["API_TOKEN"]
        );
        assert_eq!(secrets.named_in("echo ${API_TOKEN_2}"), ["API_TOKEN_2"]);
        assert_eq!(secrets.named_in("echo $API_TOKEN_2"), ["API_TOKEN_2"]);
        assert!(secrets.named_in("echo API_TOKEN and $OTHER").is_empty());

        let shown = secrets.redact("sent tok-abcdef123456, then other-998877; pin 1234");
        assert_eq!(
            shown,
            "sent [secret API_TOKEN], then [secret API_TOKEN_2]; pin 1234"
        );
        assert_eq!(secrets.redact("nothing here"), "nothing here");
        // Printing the hop's wrapper shows nothing.
        assert_eq!(
            format!("{:?}", Hidden("tok-abcdef123456".into())),
            "Hidden(***)"
        );
    }
}
