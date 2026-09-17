// Pure string formatting helpers used wherever a display-width ruler is
// needed (--list columns, terminal pads). No rendering, no colors, no
// `Msg` types — those are `display`'s job. This module exists so session
// can build its `--list` table without importing the whole presentation
// layer.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Display width of a string, per Unicode (east-asian wide = 2, combining
/// marks = 0). Both frontends measure with the same ruler.
pub fn disp_width(s: &str) -> usize {
    s.width()
}

/// Cut to `max` display columns, marking the cut with an ellipsis. Walks
/// graphemes so a combining mark is never severed from its base.
pub fn truncate_cols(s: &str, max: usize) -> String {
    let mut out = String::new();
    let mut w = 0;
    for g in s.graphemes(true) {
        let gw = disp_width(g).max(1);
        if w + gw > max.saturating_sub(2) {
            out.push('…');
            return out;
        }
        out.push_str(g);
        w += gw;
    }
    out
}
