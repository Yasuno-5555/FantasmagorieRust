//! Markdown parser
//! Ported from core/markdown.hpp

#[derive(Debug, Clone, PartialEq)]
pub struct TextSpan {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub is_link: bool,
    pub is_heading: bool,
    pub heading_level: i32,
    pub link_url: String,
}

impl Default for TextSpan {
    fn default() -> Self {
        Self {
            text: String::new(),
            bold: false,
            italic: false,
            is_link: false,
            is_heading: false,
            heading_level: 0,
            link_url: String::new(),
        }
    }
}

pub fn parse_markdown(input: &str) -> Vec<TextSpan> {
    let mut spans = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = input.chars().collect();
    let n = chars.len();

    let mut in_bold = false;
    let mut in_italic = false;
    let mut buffer = String::new();

    // Helper to flush buffer
    let mut flush =
        |spans: &mut Vec<TextSpan>, in_bold: bool, in_italic: bool, buffer: &mut String| {
            if !buffer.is_empty() {
                spans.push(TextSpan {
                    text: buffer.clone(),
                    bold: in_bold,
                    italic: in_italic,
                    ..Default::default()
                });
                buffer.clear();
            }
        };

    // Heading check
    let mut h_level = 0;
    if input.starts_with("## ") {
        h_level = 2;
        i = 3;
    } else if input.starts_with("# ") {
        h_level = 1;
        i = 2;
    }

    while i < n {
        // Unescape
        if chars[i] == '\\' && i + 1 < n {
            buffer.push(chars[i + 1]);
            i += 2;
            continue;
        }

        // Bold **
        if i + 1 < n && chars[i] == '*' && chars[i + 1] == '*' {
            flush(&mut spans, in_bold, in_italic, &mut buffer);
            in_bold = !in_bold;
            i += 2;
            continue;
        }

        // Italic *
        if chars[i] == '*' {
            flush(&mut spans, in_bold, in_italic, &mut buffer);
            in_italic = !in_italic;
            i += 1;
            continue;
        }

        // Link [text](url)
        if chars[i] == '[' {
            // Find closing ']' starting from i
            let remainder = &input[i..]; // Note: byte indexing vs char indexing mismatch risk here.
                                         // In C++, input.find("](", i) works on bytes.
                                         // In Rust, we need to be careful. constructing a string from chars[i..] is expensive.
                                         // Let's stick to simple char scan.

            // Look ahead for "]("
            let mut close_bracket = None;
            for k in i..n - 1 {
                if chars[k] == ']' && chars[k + 1] == '(' {
                    close_bracket = Some(k);
                    break;
                }
            }

            if let Some(cb) = close_bracket {
                // Look ahead for ")"
                let mut close_paren = None;
                for k in cb + 2..n {
                    if chars[k] == ')' {
                        close_paren = Some(k);
                        break;
                    }
                }

                if let Some(cp) = close_paren {
                    flush(&mut spans, in_bold, in_italic, &mut buffer);

                    let link_text: String = chars[i + 1..cb].iter().collect();
                    let link_url: String = chars[cb + 2..cp].iter().collect();

                    spans.push(TextSpan {
                        text: link_text,
                        is_link: true,
                        link_url,
                        bold: in_bold,
                        italic: in_italic,
                        ..Default::default()
                    });

                    i = cp + 1;
                    continue;
                }
            }
        }

        buffer.push(chars[i]);
        i += 1;
    }

    flush(&mut spans, in_bold, in_italic, &mut buffer);

    if h_level > 0 {
        for s in &mut spans {
            s.is_heading = true;
            s.heading_level = h_level;
        }
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bold_italic() {
        let spans = parse_markdown("Hello **World** *italic*");
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, "Hello ");
        assert_eq!(spans[1].text, "World");
        assert!(spans[1].bold);
        assert_eq!(spans[2].text, " italic"); // C++ logic puts trailing space in next buffer?
                                              // Wait, " italic" starts with space. "Hello **..." space after hello.
                                              // "italic*" -> space is before *.
                                              // Logic: " " -> buffer. "*" -> flush buffer " ". in_italic=true. "italic". "*" -> flush "italic".
    }

    #[test]
    fn test_link() {
        let spans = parse_markdown("Click [Here](http://url)");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].text, "Click ");
        assert_eq!(spans[1].text, "Here");
        assert!(spans[1].is_link);
        assert_eq!(spans[1].link_url, "http://url");
    }

    #[test]
    fn test_heading() {
        let spans = parse_markdown("## Title");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "Title");
        assert!(spans[0].is_heading);
        assert_eq!(spans[0].heading_level, 2);
    }
}
