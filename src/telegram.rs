//! A Telegram bot as a second way to reach the main conversation.
//!
//! The user makes a bot with Telegram's BotFather and gives its token to
//! the harness as the secret `TELEGRAM_BOT_TOKEN` (typed into the secret
//! window, never into the conversation). From then on what the paired
//! account sends the bot is said to main, and main's replies, and what its
//! branches bring back, are sent to that chat.
//!
//! How the two sides talk:
//! - **Messages.** Text the owner sends is said to main. Main's reply comes
//!   back as a message; so does each branch's result when it merges.
//! - **Approvals.** A call that needs the user's approval is sent with two
//!   buttons, Approve and Deny. Pressing one decides it, exactly as `a` /
//!   `d` at the screen does, and the message then says what was decided
//!   (also when it was decided at the screen instead).
//! - **Photos, video, audio, voice notes and files** the owner sends are
//!   fetched (up to the 20 MB Telegram lets a bot fetch), saved in the
//!   project folder, and main is told where. The models used are given
//!   text only, so the agent works on such a file with its tools; it does
//!   not see or hear it.
//!
//! Fixed rules, compiled, no model involved:
//! - **One owner.** The bot answers to one Telegram account, the one that
//!   sent the pairing code this harness showed on its own screen. Until
//!   then, and from anyone else afterwards, nothing is passed on and no
//!   button counts.
//! - **Private chats only.** A group the bot was added to is ignored.
//! - **Some things are decided at the screen only** (`SCREEN_ONLY`):
//!   widening the agent's own access, and adopting a change to the
//!   harness's own guardrails. And a secret is never entered from
//!   Telegram. The chat is told that something is waiting.
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
/// The largest file Telegram lets a bot fetch.
pub const MAX_FILE: u64 = 20 * 1024 * 1024;
/// Calls that are approved at the screen and nowhere else: they change
/// what the agent itself may do.
pub const SCREEN_ONLY: [&str; 2] = ["access", "adopt self-update"];

/// A photo, video, sound or other file sent to the bot.
#[derive(Clone, Debug, PartialEq)]
pub struct Media {
    /// `photo`, `video`, `audio`, `voice note`, `video note`, `animation`
    /// or `file`.
    pub kind: String,
    /// Telegram's id for fetching it.
    pub file: String,
    /// The name it was sent with, if it had one.
    pub name: String,
    pub size: u64,
}

/// A button under one of the bot's messages, pressed.
#[derive(Clone, Debug, PartialEq)]
pub struct Button {
    /// Telegram's id for this press, to acknowledge it.
    pub id: String,
    pub data: String,
}

/// One thing sent to the bot: a message, with or without a file, or a
/// button pressed.
#[derive(Clone, Debug, PartialEq)]
pub struct Incoming {
    pub update: i64,
    pub chat: i64,
    pub from: i64,
    pub private: bool,
    /// The text, or a file's caption.
    pub text: String,
    pub media: Option<Media>,
    pub button: Option<Button>,
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

/// What to do with something sent to the bot, by the rules above.
#[derive(Clone, Debug, PartialEq)]
pub enum Heard {
    /// The sender gave the pairing code: they are the owner from now on.
    Pair(i64),
    /// The owner said this: it goes to main.
    Say(String),
    /// The owner sent a file, perhaps with a caption.
    File { media: Media, caption: String },
    /// The owner pressed Approve or Deny on a call.
    Decide {
        tool: u32,
        approve: bool,
        button: String,
    },
    /// Answer with this and do nothing else.
    Reply(String),
    /// Not for this bot's owner: no answer at all.
    Nothing,
}

/// What a button carries: which call, and the verdict.
pub fn decision(tool: u32, approve: bool) -> String {
    format!("{}:{tool}", if approve { "ok" } else { "no" })
}

fn read_decision(data: &str) -> Option<(u32, bool)> {
    let (verdict, tool) = data.split_once(':')?;
    let approve = match verdict {
        "ok" => true,
        "no" => false,
        _ => return None,
    };
    Some((tool.parse().ok()?, approve))
}

/// Decides what something sent to the bot is. Pure: the same pairing, code
/// and message always give the same answer.
pub fn hear(pairing: &Pairing, code: &str, message: &Incoming) -> Heard {
    if !message.private {
        return Heard::Nothing;
    }
    let text = message.text.trim();
    let Some(owner) = pairing.owner else {
        // Nobody's yet: only the code does anything.
        if message.button.is_some() {
            return Heard::Nothing;
        }
        return if text.strip_prefix("/pair").map(str::trim) == Some(code) && !code.is_empty() {
            Heard::Pair(message.from)
        } else {
            Heard::Reply(
                "This bot is not paired yet. conscious shows a pairing code on its own \
                 screen when it connects; send it here as: /pair CODE"
                    .into(),
            )
        };
    };
    if owner != message.from {
        return Heard::Nothing;
    }
    if let Some(button) = &message.button {
        return match read_decision(&button.data) {
            Some((tool, approve)) => Heard::Decide {
                tool,
                approve,
                button: button.id.clone(),
            },
            None => Heard::Nothing,
        };
    }
    if let Some(media) = &message.media {
        return Heard::File {
            media: media.clone(),
            caption: text.to_string(),
        };
    }
    match text {
        "" => Heard::Nothing,
        "/start" | "/help" => Heard::Reply(
            "Paired. What you send here is said to conscious's main conversation, and its \
             replies come back here. A call that needs your approval comes with Approve and \
             Deny buttons. Photos, video, audio and files you send are saved in the project \
             folder for it to work on."
                .into(),
        ),
        _ => Heard::Say(text.to_string()),
    }
}

/// The file in a message, if it has one.
fn media_of(message: &Value) -> Option<Media> {
    // A photo comes in several sizes; the last is the largest.
    if let Some(photo) = message["photo"].as_array().and_then(|sizes| sizes.last()) {
        return Some(Media {
            kind: "photo".into(),
            file: photo["file_id"].as_str()?.to_string(),
            name: String::new(),
            size: photo["file_size"].as_u64().unwrap_or(0),
        });
    }
    [
        ("video", "video"),
        ("audio", "audio"),
        ("voice", "voice note"),
        ("video_note", "video note"),
        ("animation", "animation"),
        ("document", "file"),
    ]
    .iter()
    .find_map(|(field, kind)| {
        let part = message.get(*field)?;
        Some(Media {
            kind: (*kind).to_string(),
            file: part["file_id"].as_str()?.to_string(),
            name: part["file_name"].as_str().unwrap_or_default().to_string(),
            size: part["file_size"].as_u64().unwrap_or(0),
        })
    })
}

/// Reads Telegram's `getUpdates` answer: messages with text or a file,
/// and buttons pressed. Anything else is left out; it still counts as read
/// (see `last_update`).
pub fn parse_updates(answer: &Value) -> Vec<Incoming> {
    answer["result"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|u| {
            let update = u["update_id"].as_i64()?;
            if let Some(press) = u.get("callback_query") {
                let chat = &press["message"]["chat"];
                return Some(Incoming {
                    update,
                    chat: chat["id"].as_i64()?,
                    from: press["from"]["id"].as_i64()?,
                    private: chat["type"].as_str() == Some("private"),
                    text: String::new(),
                    media: None,
                    button: Some(Button {
                        id: press["id"].as_str()?.to_string(),
                        data: press["data"].as_str()?.to_string(),
                    }),
                });
            }
            let message = &u["message"];
            let media = media_of(message);
            let text = message["text"]
                .as_str()
                .or_else(|| message["caption"].as_str())
                .unwrap_or_default();
            if text.is_empty() && media.is_none() {
                return None;
            }
            Some(Incoming {
                update,
                chat: message["chat"]["id"].as_i64()?,
                from: message["from"]["id"].as_i64()?,
                private: message["chat"]["type"].as_str() == Some("private"),
                text: text.to_string(),
                media,
                button: None,
            })
        })
        .collect()
}

/// The highest update number in an answer, whatever the updates were.
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

/// The name a fetched file is saved under: when it arrived, then its own
/// name (or its kind and Telegram's extension), with nothing in it that
/// could lead out of the folder it is saved in.
pub fn file_name(media: &Media, telegram_path: &str, now: u64) -> String {
    let own = if media.name.is_empty() {
        let ext = telegram_path
            .rsplit_once('.')
            .map(|(_, ext)| ext)
            .filter(|ext| ext.len() <= 5 && ext.chars().all(|c| c.is_ascii_alphanumeric()))
            .unwrap_or("bin");
        format!("{}.{ext}", media.kind.replace(' ', "-"))
    } else {
        media.name.clone()
    };
    // Only the last part of a name that looks like a path, and no `..`.
    let last = own.rsplit(['/', '\\']).next().unwrap_or_default();
    let safe: String = last
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe.replace("..", "_");
    match safe.trim_start_matches('.') {
        "" => format!("{now}-file.bin"),
        name => format!("{now}-{name}"),
    }
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
    /// What was sent after update `after`, waiting up to `wait` seconds
    /// for something. Returns it with the highest update number seen.
    fn updates(&self, after: i64, wait: u32) -> Result<(Vec<Incoming>, i64), String>;
    fn send(&self, chat: i64, text: &str) -> Result<(), String>;
    /// Sends `text` with buttons under it, each (label, what it carries).
    /// Returns the message's id.
    fn ask(&self, chat: i64, text: &str, buttons: &[(String, String)]) -> Result<i64, String>;
    /// Replaces a message sent with `ask` by `text`, without its buttons.
    fn settle(&self, chat: i64, message: i64, text: &str) -> Result<(), String>;
    /// Acknowledges a pressed button with a short note.
    fn pressed(&self, button: &str, note: &str) -> Result<(), String>;
    /// Fetches a file: its bytes, and the path Telegram keeps it under.
    fn fetch(&self, file: &str) -> Result<(Vec<u8>, String), String>;
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

    fn agent(wait: u32) -> ureq::Agent {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_global(Some(Duration::from_secs(u64::from(wait) + 60)))
            .build()
            .into()
    }

    fn call(&self, method: &str, body: Value, wait: u32) -> Result<Value, String> {
        let url = format!("https://api.telegram.org/bot{}/{method}", self.token);
        let resp = Self::agent(wait)
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
            json!({
                "offset": after + 1,
                "timeout": wait,
                "allowed_updates": ["message", "callback_query"],
            }),
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

    fn ask(&self, chat: i64, text: &str, buttons: &[(String, String)]) -> Result<i64, String> {
        let row: Vec<Value> = buttons
            .iter()
            .map(|(label, data)| json!({"text": label, "callback_data": data}))
            .collect();
        let text: String = text.chars().take(MESSAGE_CHARS).collect();
        let answer = self.call(
            "sendMessage",
            json!({
                "chat_id": chat,
                "text": text,
                "reply_markup": {"inline_keyboard": [row]},
            }),
            0,
        )?;
        answer["result"]["message_id"]
            .as_i64()
            .ok_or_else(|| "Telegram did not say which message it sent".to_string())
    }

    fn settle(&self, chat: i64, message: i64, text: &str) -> Result<(), String> {
        let text: String = text.chars().take(MESSAGE_CHARS).collect();
        self.call(
            "editMessageText",
            json!({"chat_id": chat, "message_id": message, "text": text}),
            0,
        )
        .map(|_| ())
    }

    fn pressed(&self, button: &str, note: &str) -> Result<(), String> {
        self.call(
            "answerCallbackQuery",
            json!({"callback_query_id": button, "text": note}),
            0,
        )
        .map(|_| ())
    }

    fn fetch(&self, file: &str) -> Result<(Vec<u8>, String), String> {
        let answer = self.call("getFile", json!({"file_id": file}), 0)?;
        let path = answer["result"]["file_path"]
            .as_str()
            .ok_or("Telegram did not say where the file is")?
            .to_string();
        if answer["result"]["file_size"].as_u64().unwrap_or(0) > MAX_FILE {
            return Err("the file is larger than the 20 MB a bot may fetch".into());
        }
        let url = format!("https://api.telegram.org/file/bot{}/{path}", self.token);
        let resp = Self::agent(120)
            .get(&url)
            .call()
            .map_err(|e| self.clean(format!("the file could not be fetched: {e}")))?;
        let status = resp.status().as_u16();
        if status != 200 {
            return Err(format!("Telegram answered {status} for the file"));
        }
        let bytes = resp
            .into_body()
            .with_config()
            .limit(MAX_FILE)
            .read_to_vec()
            .map_err(|e| self.clean(format!("the file could not be read: {e}")))?;
        Ok((bytes, path))
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
            media: None,
            button: None,
        }
    }

    fn pressed(from: i64, data: &str) -> Incoming {
        Incoming {
            button: Some(Button {
                id: "press-1".into(),
                data: data.into(),
            }),
            ..said(from, true, "")
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
        // A button pressed before pairing decides nothing.
        assert_eq!(
            hear(&pairing, "482913", &pressed(7, "ok:3")),
            Heard::Nothing
        );
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
    fn the_owners_buttons_decide_a_call_and_nobody_elses_do() {
        let pairing = Pairing {
            owner: Some(7),
            after: 0,
        };
        assert_eq!(decision(12, true), "ok:12");
        assert_eq!(
            hear(&pairing, "", &pressed(7, &decision(12, true))),
            Heard::Decide {
                tool: 12,
                approve: true,
                button: "press-1".into()
            }
        );
        assert_eq!(
            hear(&pairing, "", &pressed(7, &decision(12, false))),
            Heard::Decide {
                tool: 12,
                approve: false,
                button: "press-1".into()
            }
        );
        // Someone else pressing it, or a button that says something else.
        assert_eq!(hear(&pairing, "", &pressed(8, "ok:12")), Heard::Nothing);
        for odd in ["ok:", "yes:12", "ok:twelve", "12"] {
            assert_eq!(
                hear(&pairing, "", &pressed(7, odd)),
                Heard::Nothing,
                "{odd}"
            );
        }
    }

    #[test]
    fn updates_are_read_with_their_files_and_buttons() {
        let answer = json!({"ok": true, "result": [
            {"update_id": 10, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7}, "text": "hello"}},
            {"update_id": 11, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7},
                "caption": "the whiteboard",
                "photo": [
                    {"file_id": "small", "file_size": 900},
                    {"file_id": "large", "file_size": 240000}]}},
            {"update_id": 12, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7},
                "voice": {"file_id": "v1", "file_size": 52000}}},
            {"update_id": 13, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7},
                "document": {"file_id": "d1", "file_name": "report.pdf", "file_size": 81000}}},
            {"update_id": 14, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7},
                "video": {"file_id": "vid", "file_name": "clip.mp4", "file_size": 3000000}}},
            {"update_id": 15, "message": {
                "chat": {"id": 7, "type": "private"}, "from": {"id": 7}, "sticker": {}}},
            {"update_id": 16, "callback_query": {
                "id": "press-9", "from": {"id": 7}, "data": "ok:4",
                "message": {"message_id": 55, "chat": {"id": 7, "type": "private"}}}},
            {"update_id": 17, "message": {
                "chat": {"id": -5, "type": "group"}, "from": {"id": 9}, "text": "hi all"}},
        ]});
        let got = parse_updates(&answer);
        assert_eq!(got.len(), 7, "{got:?}");
        assert_eq!(got[0].text, "hello");
        // The largest size of a photo, with its caption.
        let photo = got[1].media.clone().unwrap();
        assert_eq!(
            (photo.kind.as_str(), photo.file.as_str()),
            ("photo", "large")
        );
        assert_eq!(got[1].text, "the whiteboard");
        assert_eq!(got[2].media.as_ref().unwrap().kind, "voice note");
        let file = got[3].media.clone().unwrap();
        assert_eq!(
            (file.kind.as_str(), file.name.as_str()),
            ("file", "report.pdf")
        );
        assert_eq!(got[4].media.as_ref().unwrap().kind, "video");
        assert_eq!(got[5].button.as_ref().unwrap().data, "ok:4");
        assert!(!got[6].private);
        // The sticker still counts as read, so it is not asked for again.
        assert_eq!(last_update(&answer), Some(17));
        assert_eq!(last_update(&json!({"ok": true, "result": []})), None);

        // A file from the owner is for main; from anyone else, for nobody.
        let pairing = Pairing {
            owner: Some(7),
            after: 0,
        };
        assert_eq!(
            hear(&pairing, "", &got[1]),
            Heard::File {
                media: photo,
                caption: "the whiteboard".into()
            }
        );
        let mut stranger = got[1].clone();
        stranger.from = 8;
        assert_eq!(hear(&pairing, "", &stranger), Heard::Nothing);
    }

    #[test]
    fn a_fetched_file_is_named_safely() {
        let media = |kind: &str, name: &str| Media {
            kind: kind.into(),
            file: "f".into(),
            name: name.into(),
            size: 1,
        };
        assert_eq!(
            file_name(&media("photo", ""), "photos/file_12.jpg", 1_794_000_000),
            "1794000000-photo.jpg"
        );
        assert_eq!(
            file_name(&media("voice note", ""), "voice/file_3.oga", 5),
            "5-voice-note.oga"
        );
        assert_eq!(
            file_name(
                &media("file", "Q3 report (final).pdf"),
                "documents/x.pdf",
                5
            ),
            "5-Q3_report__final_.pdf"
        );
        // Nothing in a name leads out of the folder it is saved in.
        let odd = file_name(&media("file", "../../.ssh/authorized_keys"), "d/x", 5);
        assert!(!odd.contains('/') && !odd.contains(".."), "{odd}");
        assert_eq!(
            file_name(&media("file", ""), "no-extension", 5),
            "5-file.bin"
        );
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
