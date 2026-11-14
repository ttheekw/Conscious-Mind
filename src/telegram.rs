//! A Telegram bot as a second way to reach the main conversation.
//!
//! The user makes a bot with Telegram's BotFather and gives its token to
//! the harness as the secret `TELEGRAM_BOT_TOKEN` (typed into the secret
//! window, never into the conversation). From then on what the paired
//! account sends the bot is said to main, and main's replies, and what its
//! branches bring back, are sent to that chat.
//!
//! Fixed rules, compiled, no model involved:
//! - **One owner.** The bot answers to one Telegram account, the one that
//!   sent the pairing code this harness showed on its own screen. Until
//!   then, and from anyone else afterwards, messages are not passed on.
//! - **Private chats only.** A group the bot was added to is ignored.
//! - **Nothing is approved from Telegram.** A call that needs approval, or
//!   a secret that needs typing, waits for the screen; the chat is only
//!   told that something is waiting.
//! - The token is never in an error, an event or a log.
//!
//! This module is the network I/O and the rules (`hear`), both small.
//! `live.rs` owns the bridge to the conversation.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The secret the bot's token is kept under.
pub const TOKEN: &str = "TELEGRAM_BOT_TOKEN";
/// How long one request for new messages waits at Telegram's end.
pub const WAIT_SECS: u32 = 25;
/// Telegram refuses a message longer than 4,096 characters.
const MESSAGE_CHARS: usize = 4_000;

/// One message sent to the bot.
#[derive(Clone, Debug, PartialEq)]
pub struct Incoming {
    pub update: i64,
    pub chat: i64,
    pub from: i64,
    pub private: bool,
    pub text: String,
}

/// Who the bot answers to, and how far its messages have been read. Kept
/// in `telegram.json` beside the routing knowledge.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pairing {
    #[serde(default)]
    pub owner: Option<i64>,
    /// The last update handled, so none is handled twice across restarts.
    #[serde(default)]
    pub after: i64,
}

/// What to do with a message, by the rules above.
#[derive(Clone, Debug, PartialEq)]
pub enum Heard {
    /// The sender gave the pairing code: they are the owner from now on.
    Pair(i64),
    /// The owner said this: it goes to main.
    Say(String),
    /// Answer with this and do nothing else.
    Reply(String),
    /// Not for this bot's owner: no answer at all.
    Nothing,
}

/// Decides what a message is. Pure: the same pairing, code and message
/// always give the same answer.
pub fn hear(pairing: &Pairing, code: &str, message: &Incoming) -> Heard {
    if !message.private {
        return Heard::Nothing;
    }
    let text = message.text.trim();
    match pairing.owner {
        None => {
            if text.strip_prefix("/pair").map(str::trim) == Some(code) && !code.is_empty() {
                Heard::Pair(message.from)
            } else {
                Heard::Reply(
                    "This bot is not paired yet. conscious shows a pairing code on its own \
                     screen when it connects; send it here as: /pair CODE"
                        .into(),
                )
            }
        }
        Some(owner) if owner != message.from => Heard::Nothing,
        Some(_) if text.is_empty() => Heard::Nothing,
        Some(_) if text == "/start" || text == "/help" => Heard::Reply(
            "Paired. What you send here is said to conscious's main conversation, and its \
             replies come back here. Anything that needs your approval waits on its screen."
                .into(),
        ),
        Some(_) => Heard::Say(text.to_string()),
    }
}

/// Reads Telegram's `getUpdates` answer. Anything that is not a text
/// message is left out, but still moves `update` forward through the
/// caller's own bookkeeping (see `last_update`).
pub fn parse_updates(answer: &Value) -> Vec<Incoming> {
    answer["result"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|u| {
            let message = &u["message"];
            Some(Incoming {
                update: u["update_id"].as_i64()?,
                chat: message["chat"]["id"].as_i64()?,
                from: message["from"]["id"].as_i64()?,
                private: message["chat"]["type"].as_str() == Some("private"),
                text: message["text"].as_str()?.to_string(),
            })
        })
        .collect()
}

/// The highest update number in an answer, text message or not.
pub fn last_update(answer: &Value) -> Option<i64> {
    answer["result"]
        .as_array()?
        .iter()
        .filter_map(|u| u["update_id"].as_i64())
        .max()
}

/// A long text as pieces Telegram will take, cut at line ends where it can.
pub fn pieces(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut piece = String::new();
    for line in text.split_inclusive('\n') {
        if piece.chars().count() + line.chars().count() > MESSAGE_CHARS && !piece.is_empty() {
            out.push(std::mem::take(&mut piece));
        }
        // One line longer than a message is cut where it must be.
        let mut rest: Vec<char> = line.chars().collect();
        while rest.len() > MESSAGE_CHARS {
            out.push(rest.drain(..MESSAGE_CHARS).collect());
        }
        piece.extend(rest);
    }
    if !piece.trim().is_empty() {
        out.push(piece);
    }
    out
}

/// A pairing code: six digits, different each time one is needed.
pub fn new_code() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    format!("{:06}", h.finish() % 1_000_000)
}

/// The bot, as the bridge uses it. Behind a trait so the bridge is tested
/// without a network.
pub trait Bot: Send + Sync {
    /// Messages after update `after`, waiting up to `wait` seconds for one.
    /// Returns them with the highest update number seen.
    fn updates(&self, after: i64, wait: u32) -> Result<(Vec<Incoming>, i64), String>;
    fn send(&self, chat: i64, text: &str) -> Result<(), String>;
}

/// The real bot: Telegram's HTTPS API.
pub struct Api {
    token: String,
}

impl Api {
    pub fn new(token: String) -> Self {
        Self { token }
    }

    /// An error in words that never include the token (it is part of the
    /// address every request goes to).
    fn clean(&self, text: String) -> String {
        text.replace(&self.token, "[token]")
    }

    fn call(&self, method: &str, body: Value, wait: u32) -> Result<Value, String> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_global(Some(Duration::from_secs(u64::from(wait) + 20)))
            .build()
            .into();
        let url = format!("https://api.telegram.org/bot{}/{method}", self.token);
        let resp = agent
            .post(&url)
            .header("Content-Type", "application/json")
            .send(body.to_string())
            .map_err(|e| self.clean(format!("Telegram could not be reached: {e}")))?;
        let status = resp.status().as_u16();
        let text = resp.into_body().read_to_string().unwrap_or_default();
        let answer: Value = serde_json::from_str(&text)
            .map_err(|_| format!("Telegram answered {status} with something unreadable"))?;
        if answer["ok"].as_bool() != Some(true) {
            let why = answer["description"].as_str().unwrap_or("no reason given");
            return Err(self.clean(match status {
                401 | 404 => format!(
                    "Telegram does not accept the bot token ({why}). Enter the right one \
                     as the secret {TOKEN}."
                ),
                _ => format!("Telegram answered {status}: {why}"),
            }));
        }
        Ok(answer)
    }
}

impl Bot for Api {
    fn updates(&self, after: i64, wait: u32) -> Result<(Vec<Incoming>, i64), String> {
        let answer = self.call(
            "getUpdates",
            json!({"offset": after + 1, "timeout": wait, "allowed_updates": ["message"]}),
            wait,
        )?;
        Ok((
            parse_updates(&answer),
            last_update(&answer).unwrap_or(after),
        ))
    }

    fn send(&self, chat: i64, text: &str) -> Result<(), String> {
        for piece in pieces(text) {
            self.call("sendMessage", json!({"chat_id": chat, "text": piece}), 0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn said(from: i64, private: bool, text: &str) -> Incoming {
        Incoming {
            update: 1,
            chat: from,
            from,
            private,
            text: text.into(),
        }
    }

    #[test]
    fn only_the_account_that_gave_the_code_is_heard() {
        let mut pairing = Pairing::default();
        // Before pairing nothing is passed on, whoever writes.
        assert!(matches!(
            hear(&pairing, "482913", &said(7, true, "delete everything")),
            Heard::Reply(_)
        ));
        assert!(matches!(
            hear(&pairing, "482913", &said(7, true, "/pair 000000")),
            Heard::Reply(_)
        ));
        // No code set means nobody can pair.
        assert!(matches!(
            hear(&pairing, "", &said(7, true, "/pair")),
            Heard::Reply(_)
        ));
        assert_eq!(
            hear(&pairing, "482913", &said(7, true, " /pair 482913 ")),
            Heard::Pair(7)
        );
        pairing.owner = Some(7);
        // The owner is heard; anyone else gets no answer at all, and
        // cannot pair again even with the code.
        assert_eq!(
            hear(&pairing, "482913", &said(7, true, "how is the build?")),
            Heard::Say("how is the build?".into())
        );
        assert_eq!(
            hear(&pairing, "482913", &said(8, true, "how is the build?")),
            Heard::Nothing
        );
        assert_eq!(
            hear(&pairing, "482913", &said(8, true, "/pair 482913")),
            Heard::Nothing
        );
        // A group chat is ignored, even from the owner.
        assert_eq!(
            hear(&pairing, "482913", &said(7, false, "how is the build?")),
            Heard::Nothing
        );
        assert!(matches!(
            hear(&pairing, "482913", &said(7, true, "/start")),
            Heard::Reply(_)
        ));
    }

    #[test]
    fn updates_are_read_and_what_is_not_a_text_message_is_skipped() {
        let answer = json!({"ok": true, "result": [
            {"update_id": 10, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7}, "text": "hello"}},
            {"update_id": 11, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7}, "photo": []}},
            {"update_id": 12, "message": {
                "chat": {"id": -5, "type": "group"}, "from": {"id": 9}, "text": "hi all"}},
        ]});
        let got = parse_updates(&answer);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], said(7, true, "hello").with_update(10));
        assert!(!got[1].private);
        // The photo still counts as read, so it is not asked for again.
        assert_eq!(last_update(&answer), Some(12));
        assert_eq!(last_update(&json!({"ok": true, "result": []})), None);
    }

    impl Incoming {
        fn with_update(mut self, update: i64) -> Self {
            self.update = update;
            self
        }
    }

    #[test]
    fn long_replies_are_sent_in_pieces_and_codes_are_six_digits() {
        assert_eq!(pieces("short"), ["short"]);
        assert!(pieces("  \n").is_empty());
        let long = format!("{}\n{}\n", "a".repeat(3_000), "b".repeat(3_000));
        let got = pieces(&long);
        assert_eq!(got.len(), 2);
        assert!(got[0].starts_with('a') && got[1].starts_with('b'));
        let one_line = "c".repeat(9_000);
        let got = pieces(&one_line);
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|p| p.chars().count() <= MESSAGE_CHARS));
        assert_eq!(got.concat(), one_line);

        let code = new_code();
        assert_eq!(code.len(), 6);
        assert!(code.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn the_token_is_never_part_of_an_error() {
        let api = Api::new("123456:secret-token-value".into());
        let shown = api.clean(
            "https://api.telegram.org/bot123456:secret-token-value/getUpdates: timed out".into(),
        );
        assert!(!shown.contains("secret-token-value"));
        assert!(shown.contains("[token]"));
    }
}
