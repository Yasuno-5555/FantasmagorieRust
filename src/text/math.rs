use crate::core::Vec2;
use crate::text::FontManager;

#[derive(Debug, Clone)]
pub enum MathNode {
    Text(String),
    Sup(Box<MathNode>, Box<MathNode>),  // Base, Exp
    Sub(Box<MathNode>, Box<MathNode>),  // Base, Sub
    Frac(Box<MathNode>, Box<MathNode>), // Num, Den
    Sqrt(Box<MathNode>),
    Row(Vec<MathNode>),
}

#[derive(Debug, Clone, Copy)]
pub struct LayoutRect {
    pub offset: Vec2, // Relative to parent
    pub size: Vec2,
    pub advance: f32, // X advance
}

pub struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn consume(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn consume_char(&mut self, target: char) -> bool {
        if let Some(c) = self.peek() {
            if c == target {
                self.consume();
                return true;
            }
        }
        false
    }

    // Parse braced group { ... } or single char
    fn parse_group(&mut self) -> MathNode {
        if self.consume_char('{') {
            let node = self.parse_expr();
            self.consume_char('}');
            node
        } else {
            // Read until special char or end
            let start = self.pos;
            while let Some(c) = self.peek() {
                if matches!(c, '^' | '_' | '{' | '}' | '\\') {
                    break;
                }
                self.consume();
                // If it's just a standard char, return immediately as a single unit?
                // Actually to support multi-char text like "sin", we should read word
                // But for "x^2", x is one node.
                // Simplified: Text is sequence of non-specials?
                // Or: Single char is standard for sup/sub unless braced.
                // Let's adopt LaTeX rule: non-braced argument is single token.
                // So if we are here, we read ONE char if it's not special.

                // EXCEPT if we are parsing normal text "abc".
                // Let's assume Text node captures a run.
            }
            if self.pos > start {
                // Return accumulated text
                let s = &self.input[start..self.pos];
                // Wait, if input is "abc^2", we want Row(Text(abc), Sup(...))?
                // No, standard TeX: "abc" is distinct marks.
                // We'll return Text containing the string.
                // BUT, if we encounter special chars next, this func returns, and caller loops.
                return MathNode::Text(s.to_string());
            } else if let Some(c) = self.consume() {
                if c == '\\' {
                    return self.parse_command();
                }
                return MathNode::Text(c.to_string());
            }
            MathNode::Text("".to_string())
        }
    }

    fn parse_command(&mut self) -> MathNode {
        // Read command name
        let start = self.pos;
        while let Some(c) = self.peek() {
            if !c.is_alphabetic() {
                break;
            }
            self.consume();
        }
        let cmd = &self.input[start..self.pos];
        match cmd {
            "frac" => {
                let num = self.parse_group(); // Standard LaTeX: \frac{num}{den}
                let den = self.parse_group();
                MathNode::Frac(Box::new(num), Box::new(den))
            }
            "sqrt" => {
                let inner = self.parse_group();
                MathNode::Sqrt(Box::new(inner))
            }
            _ => MathNode::Text(format!("\\{}", cmd)), // Unknown command as text
        }
    }

    fn parse_expr(&mut self) -> MathNode {
        let mut nodes = Vec::new();
        while let Some(c) = self.peek() {
            if c == '}' {
                break;
            }

            if c == '^' {
                self.consume();
                let exp = self.parse_group();
                if let Some(last) = nodes.pop() {
                    nodes.push(MathNode::Sup(Box::new(last), Box::new(exp)));
                } else {
                    nodes.push(MathNode::Text("^".to_string())); // Error handling
                    nodes.push(exp);
                }
            } else if c == '_' {
                self.consume();
                let sub = self.parse_group();
                if let Some(last) = nodes.pop() {
                    nodes.push(MathNode::Sub(Box::new(last), Box::new(sub)));
                } else {
                    nodes.push(MathNode::Text("_".to_string()));
                    nodes.push(sub);
                }
            } else {
                let node = self.parse_group();
                if let MathNode::Text(ref t) = node {
                    if t.is_empty() {
                        continue;
                    } // End of stream
                }
                nodes.push(node);
            }
        }

        if nodes.len() == 1 {
            nodes.pop().unwrap()
        } else {
            MathNode::Row(nodes)
        }
    }

    pub fn parse(&mut self) -> MathNode {
        let res = self.parse_expr();
        res
    }
}

// Layout Logic

pub fn measure(node: &MathNode, fm: &mut FontManager, base_size: f32) -> Vec2 {
    match node {
        MathNode::Text(t) => {
            // Measure text width
            // This assumes single font for now.
            // Just sum advances.
            // FontManager doesn't expose quick string width helper publicly?
            // "get_glyph" loop needed.
            // Let's assume a helper exists or do it here.
            let mut w = 0.0;
            // Hack: we need init_fonts check, but usually done.
            if fm.fonts.is_empty() {
                fm.init_fonts();
            } // Ensure
            for c in t.chars() {
                if let Some(g) = fm.get_glyph(0, c, base_size) {
                    w += g.advance;
                }
            }
            // Height is rough. Using font size typically or max bearing?
            // Simple box: width, base_size
            Vec2::new(w, base_size)
        }
        MathNode::Row(nodes) => {
            let mut w = 0.0;
            let mut h: f32 = 0.0;
            for n in nodes {
                let s = measure(n, fm, base_size);
                w += s.x;
                h = h.max(s.y);
            }
            Vec2::new(w, h)
        }
        MathNode::Sup(base, exp) => {
            let bs = measure(base, fm, base_size);
            let es = measure(exp, fm, base_size * 0.7);
            Vec2::new(bs.x + es.x, bs.y + es.y * 0.5) // Rough
        }
        MathNode::Sub(base, sub) => {
            let bs = measure(base, fm, base_size);
            let ss = measure(sub, fm, base_size * 0.7);
            Vec2::new(bs.x + ss.x, bs.y + ss.y * 0.5)
        }
        MathNode::Frac(num, den) => {
            let ns = measure(num, fm, base_size * 0.9);
            let ds = measure(den, fm, base_size * 0.9);
            Vec2::new(ns.x.max(ds.x) + 4.0, ns.y + ds.y + 4.0) // 4.0 padding
        }
        MathNode::Sqrt(inner) => {
            let is = measure(inner, fm, base_size);
            Vec2::new(is.x + 10.0, is.y + 4.0)
        }
    }
}
