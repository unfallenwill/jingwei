//! The TUI frontend: the only layer allowed to touch a terminal. Everything
//! above this line is pure (model/update/view); this module owns raw mode,
//! the alternate screen, the event stream, and the agent task — and does
//! nothing else. If it has logic, that logic belongs in update.rs.

pub mod model;
pub mod update;
pub mod view;

use crate::display::{self, Msg, Sev};
use crate::{agent_turn, home_dir, CancelToken, Config, Error};
use model::App;
use std::io::{self, IsTerminal};
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use update::{update as step, Action, Ev};

/// Open the TUI: terminal setup, the event loop, guaranteed restore.
pub async fn run(cfg: &Config) -> crate::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    display::install_chan(tx);
    display::disp(Msg::Banner(format!(
        "jingwei — 精卫填海，一石一石 · {} · {} · {}",
        cfg.protocol_label(), cfg.model, cfg.base_url)));
    display::disp(Msg::Banner(
        "type a task · Ctrl-O unfolds everything · Ctrl-C interrupts (twice exits) · Ctrl-D rests".into()));

    let mut app = App::new();
    app.input.history = load_history();

    let history = Arc::new(AsyncMutex::new(Vec::<serde_json::Value>::new()));
    let mut agent: Option<(Arc<CancelToken>, tokio::task::JoinHandle<()>)> = None;

    crossterm::terminal::enable_raw_mode().map_err(|e| Error::Msg(format!("raw mode: {e}")))?;
    crossterm::execute!(
        io::stdout(),
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableBracketedPaste
    )?;
    let mut term = ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(io::stdout()))
        .map_err(|e| Error::Msg(format!("terminal: {e}")))?;

    let mut events = key_channel();
    let mut tick = tokio::time::interval(model::TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut done = false;
    while !done {
        draw(&mut term, &app)?;

        tokio::select! {
            maybe = events.recv() => {
                match maybe {
                    Some(ev) => match ev {
                        crossterm::event::Event::Key(k) => {
                            handle(step(&mut app, Ev::Key(k)), cfg, &history, &mut agent);
                        }
                        crossterm::event::Event::Paste(p) => {
                            handle(step(&mut app, Ev::Paste(p)), cfg, &history, &mut agent);
                        }
                        crossterm::event::Event::Resize(..) => {
                            let _ = term.autoresize();
                        }
                        _ => {}
                    },
                    None => done = true, // stdin gone
                }
            }
            maybe = rx.recv() => {
                match maybe {
                    Some(Msg::TaskEnd) => {
                        agent = None;
                        step(&mut app, Ev::Msg(Msg::TaskEnd));
                    }
                    Some(m) => {
                        step(&mut app, Ev::Msg(m));
                    }
                    None => {}
                }
            }
            _ = tick.tick() => {
                step(&mut app, Ev::Tick);
            }
        }
        done = done || app.quit;
    }

    // Restore the terminal no matter how we left the loop.
    let _ = term.show_cursor();
    let _ = crossterm::terminal::disable_raw_mode();
    let _ = crossterm::execute!(io::stdout(), crossterm::event::DisableBracketedPaste, crossterm::terminal::LeaveAlternateScreen);
    save_history(&app.input.history);
    Ok(())
}

/// One frame: view is pure; this only blits it and places the cursor.
fn draw(term: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>, app: &App) -> io::Result<()> {
    term.draw(|f| {
        let area = f.area();
        let screen = view::view(app, area.width, area.height);
        let text = ratatui::text::Text::from(screen.lines);
        f.render_widget(ratatui::widgets::Paragraph::new(text), area);
        if let Some(col) = screen.cursor {
            let y = area.height.saturating_sub(2); // the input row
            let x = (col as u16).min(area.width.saturating_sub(1));
            f.set_cursor_position(ratatui::layout::Position { x, y });
        }
    })?;
    Ok(())
}

/// Perform an update's side effect: submit spawns the agent coroutine,
/// cancel fires its token.
fn handle(
    action: Action,
    cfg: &Config,
    history: &Arc<AsyncMutex<Vec<serde_json::Value>>>,
    agent: &mut Option<(Arc<CancelToken>, tokio::task::JoinHandle<()>)>,
) {
    match action {
        Action::None => {}
        Action::Cancel => {
            if let Some((token, _)) = agent {
                token.cancel();
            }
        }
        Action::Submit(line) => {
            let cfg = cfg.clone();
            let history = history.clone();
            let token = Arc::new(CancelToken::new());
            let tok = token.clone();
            display::disp(Msg::TaskBegin(line.clone()));
            let job = tokio::spawn(async move {
                let mut h = history.lock().await;
                h.push(serde_json::json!({"role": "user", "content": line}));
                match agent_turn(&cfg, &mut h, &tok).await {
                    Err(Error::Interrupted) => {}
                    Err(e) => display::disp(Msg::Note { sev: Sev::Err, text: format!(" error: {e} ") }),
                    Ok(()) => {}
                }
                display::disp(Msg::TaskEnd);
            });
            *agent = Some((token, job));
        }
    }
}

/// Should the REPL open the TUI? A terminal on stdout and no opt-out.
pub fn wanted() -> bool {
    io::stdout().is_terminal() && std::env::var_os("JINGWEI_NO_TUI").is_none()
}

fn load_history() -> Vec<String> {
    home_dir()
        .map(|d| d.join(".jingwei_history"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect())
        .unwrap_or_default()
}

fn save_history(history: &[String]) {
    if let Some(dir) = home_dir() {
        let _ = std::fs::write(dir.join(".jingwei_history"), history.join("\n") + "\n");
    }
}

/// Terminal events on their own thread: crossterm's blocking `read()` is a
/// syscall, and syscalls don't suspend coroutines — so the reader gets a
/// thread and the event loop receives through a channel, exactly like the
/// SSE lines of a streamed response. Losing the receiver just ends the
/// thread; quitting never waits on a keystroke.
fn key_channel() -> tokio::sync::mpsc::UnboundedReceiver<crossterm::event::Event> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || loop {
        match crossterm::event::read() {
            Ok(ev) => {
                if tx.send(ev).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    });
    rx
}
