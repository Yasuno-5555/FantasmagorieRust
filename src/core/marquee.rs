//! Marquee selection system
//!
//! Provides rectangular selection (marquee/lasso) functionality for selecting
//! multiple items in a canvas or list.

use crate::core::Vec2;

/// State of a marquee selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarqueeState {
    /// No selection in progress
    Idle,
    /// Selection is being dragged
    Selecting { start: Vec2, current: Vec2 },
    /// Selection just completed
    Completed { start: Vec2, end: Vec2 },
}

/// Marquee selection handler
pub struct MarqueeSelection {
    state: MarqueeState,
    /// Minimum drag distance to start selection
    threshold: f32,
    /// Whether to add to existing selection (shift key)
    additive: bool,
}

/// Rectangle for hit testing
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Create from two corner points
    pub fn from_corners(p1: Vec2, p2: Vec2) -> Self {
        let min_x = p1.x.min(p2.x);
        let min_y = p1.y.min(p2.y);
        let max_x = p1.x.max(p2.x);
        let max_y = p1.y.max(p2.y);
        Self {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    /// Check if this rect contains a point
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.x && point.x <= self.x + self.width &&
        point.y >= self.y && point.y <= self.y + self.height
    }

    /// Check if this rect intersects another rect
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width &&
        self.x + self.width > other.x &&
        self.y < other.y + other.height &&
        self.y + self.height > other.y
    }

    /// Check if this rect fully contains another rect
    pub fn contains_rect(&self, other: &Rect) -> bool {
        other.x >= self.x &&
        other.y >= self.y &&
        other.x + other.width <= self.x + self.width &&
        other.y + other.height <= self.y + self.height
    }

    /// Get the center point
    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    /// Get the area
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// Item that can be selected by marquee
pub trait Selectable {
    /// Get the bounding rectangle of this item
    fn bounds(&self) -> Rect;
    
    /// Get unique identifier
    fn id(&self) -> usize;
}

impl MarqueeSelection {
    /// Create a new marquee selection handler
    pub fn new() -> Self {
        Self {
            state: MarqueeState::Idle,
            threshold: 5.0,
            additive: false,
        }
    }

    /// Set the minimum drag distance to start selection
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set additive mode (shift-select behavior)
    pub fn set_additive(&mut self, additive: bool) {
        self.additive = additive;
    }

    /// Start a potential selection
    pub fn begin(&mut self, pos: Vec2) {
        self.state = MarqueeState::Selecting { start: pos, current: pos };
    }

    /// Update the selection as the mouse moves
    pub fn update(&mut self, pos: Vec2) -> bool {
        if let MarqueeState::Selecting { start, .. } = self.state {
            let distance = (pos - start).length();
            if distance >= self.threshold {
                self.state = MarqueeState::Selecting { start, current: pos };
                return true;
            }
        }
        false
    }

    /// Complete the selection
    pub fn end(&mut self, pos: Vec2) -> Option<Rect> {
        if let MarqueeState::Selecting { start, .. } = self.state {
            let distance = (pos - start).length();
            if distance >= self.threshold {
                self.state = MarqueeState::Completed { start, end: pos };
                return Some(Rect::from_corners(start, pos));
            }
        }
        self.state = MarqueeState::Idle;
        None
    }

    /// Cancel the selection
    pub fn cancel(&mut self) {
        self.state = MarqueeState::Idle;
    }

    /// Get current selection rectangle (if selecting)
    pub fn current_rect(&self) -> Option<Rect> {
        match self.state {
            MarqueeState::Selecting { start, current } => Some(Rect::from_corners(start, current)),
            MarqueeState::Completed { start, end } => Some(Rect::from_corners(start, end)),
            MarqueeState::Idle => None,
        }
    }

    /// Check if selection is active
    pub fn is_selecting(&self) -> bool {
        matches!(self.state, MarqueeState::Selecting { .. })
    }

    /// Test which items are selected
    pub fn test_items<T: Selectable>(&self, items: &[T]) -> Vec<usize> {
        let rect = match self.current_rect() {
            Some(r) => r,
            None => return Vec::new(),
        };

        items.iter()
            .filter(|item| rect.intersects(&item.bounds()))
            .map(|item| item.id())
            .collect()
    }

    /// Test which items are fully contained in the selection
    pub fn test_items_contained<T: Selectable>(&self, items: &[T]) -> Vec<usize> {
        let rect = match self.current_rect() {
            Some(r) => r,
            None => return Vec::new(),
        };

        items.iter()
            .filter(|item| rect.contains_rect(&item.bounds()))
            .map(|item| item.id())
            .collect()
    }

    /// Get whether this is an additive selection
    pub fn is_additive(&self) -> bool {
        self.additive
    }

    /// Get current state
    pub fn state(&self) -> MarqueeState {
        self.state
    }
}

impl Default for MarqueeSelection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestItem {
        id: usize,
        rect: Rect,
    }

    impl Selectable for TestItem {
        fn bounds(&self) -> Rect { self.rect }
        fn id(&self) -> usize { self.id }
    }

    #[test]
    fn test_rect_intersection() {
        let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);
        let r3 = Rect::new(200.0, 200.0, 50.0, 50.0);

        assert!(r1.intersects(&r2));
        assert!(!r1.intersects(&r3));
    }

    #[test]
    fn test_marquee_selection() {
        let mut marquee = MarqueeSelection::new().with_threshold(5.0);
        
        let items = vec![
            TestItem { id: 0, rect: Rect::new(10.0, 10.0, 20.0, 20.0) },
            TestItem { id: 1, rect: Rect::new(50.0, 50.0, 20.0, 20.0) },
            TestItem { id: 2, rect: Rect::new(100.0, 100.0, 20.0, 20.0) },
        ];

        marquee.begin(Vec2::new(0.0, 0.0));
        marquee.update(Vec2::new(60.0, 60.0));
        
        let selected = marquee.test_items(&items);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        assert!(!selected.contains(&2));
    }
}
