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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_character_costs_two_columns() {
        assert_eq!(disp_width("ab"), 2, "ascii is one column each");
        assert_eq!(disp_width("精卫"), 4, "east-asian wide is two columns each");
        assert_eq!(disp_width("e\u{301}"), 1, "a combining mark rides along with its base");
    }

    #[test]
    fn a_long_string_is_cut_to_fit_and_marked() {
        assert_eq!(truncate_cols("abcdef", 10), "abcdef", "a string that fits is untouched");
        assert_eq!(truncate_cols("abcdefghij", 5), "abc…", "a cut string is marked with an ellipsis");
        assert_eq!(truncate_cols("精卫填海", 5), "精…", "wide characters are counted whole");
    }
}
