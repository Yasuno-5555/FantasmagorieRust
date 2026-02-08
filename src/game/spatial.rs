use crate::core::Rectangle;
use super::world::EntityId;

const MAX_ENTITIES: usize = 16;
const MAX_LEVELS: usize = 5;

pub struct Quadtree {
    level: usize,
    bounds: Rectangle,
    entities: Vec<(EntityId, Rectangle)>,
    nodes: [Option<Box<Quadtree>>; 4],
}

impl Quadtree {
    pub fn new(level: usize, bounds: Rectangle) -> Self {
        Self {
            level,
            bounds,
            entities: Vec::new(),
            nodes: [None, None, None, None],
        }
    }

    pub fn clear(&mut self) {
        self.entities.clear();
        for node in &mut self.nodes {
            if let Some(mut n) = node.take() {
                n.clear();
            }
        }
    }

    fn split(&mut self) {
        let sub_w = self.bounds.w * 0.5;
        let sub_h = self.bounds.h * 0.5;
        let x = self.bounds.x;
        let y = self.bounds.y;

        self.nodes[0] = Some(Box::new(Quadtree::new(self.level + 1, Rectangle::new(x + sub_w, y, sub_w, sub_h))));
        self.nodes[1] = Some(Box::new(Quadtree::new(self.level + 1, Rectangle::new(x, y, sub_w, sub_h))));
        self.nodes[2] = Some(Box::new(Quadtree::new(self.level + 1, Rectangle::new(x, y + sub_h, sub_w, sub_h))));
        self.nodes[3] = Some(Box::new(Quadtree::new(self.level + 1, Rectangle::new(x + sub_w, y + sub_h, sub_w, sub_h))));
    }

    fn get_indices(&self, rect: &Rectangle) -> Vec<usize> {
        let mut indices = Vec::new();
        let vertical_midpoint = self.bounds.x + (self.bounds.w * 0.5);
        let horizontal_midpoint = self.bounds.y + (self.bounds.h * 0.5);

        let top_quadrant = rect.y < horizontal_midpoint && rect.y + rect.h < horizontal_midpoint;
        let bottom_quadrant = rect.y > horizontal_midpoint;
        
        // Right
        if rect.x > vertical_midpoint {
            if top_quadrant { indices.push(0); }
            else if bottom_quadrant { indices.push(3); }
            else { indices.push(0); indices.push(3); }
        }
        // Left
        else if rect.x + rect.w < vertical_midpoint {
            if top_quadrant { indices.push(1); }
            else if bottom_quadrant { indices.push(2); }
            else { indices.push(1); indices.push(2); }
        }
        // Spans center
        else {
            if top_quadrant { indices.push(0); indices.push(1); }
            else if bottom_quadrant { indices.push(2); indices.push(3); }
            else { indices.push(0); indices.push(1); indices.push(2); indices.push(3); }
        }

        indices
    }

    pub fn insert(&mut self, entity: EntityId, rect: Rectangle) {
        if let Some(_) = self.nodes[0] {
            let indices = self.get_indices(&rect);
            for idx in indices {
                self.nodes[idx].as_mut().unwrap().insert(entity, rect);
            }
            return;
        }

        self.entities.push((entity, rect));

        if self.entities.len() > MAX_ENTITIES && self.level < MAX_LEVELS {
            if self.nodes[0].is_none() {
                self.split();
            }

            let mut i = 0;
            while i < self.entities.len() {
                let (e, r) = self.entities[i];
                let indices = self.get_indices(&r);
                if !indices.is_empty() {
                    // If it fits into quadrants, move it down
                    // Note: for simplicity, we move it to the first quadrant it fits if it spans, 
                    // or we could keep it here. Traditional Quadtree keeps it in the node it spans.
                    // Here we'll just push it down to ALL quadrants it touches if we want full coverage.
                    for idx in indices {
                        self.nodes[idx].as_mut().unwrap().insert(e, r);
                    }
                    self.entities.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    pub fn retrieve(&self, rect: &Rectangle, return_entities: &mut Vec<EntityId>) {
        if let Some(_) = self.nodes[0] {
            let indices = self.get_indices(rect);
            for idx in indices {
                self.nodes[idx].as_ref().unwrap().retrieve(rect, return_entities);
            }
        }

        for (e, _) in &self.entities {
            return_entities.push(*e);
        }
    }
}
