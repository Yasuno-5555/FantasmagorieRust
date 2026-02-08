use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use crate::game::EntityId;

/// Types of signals that can be dispatched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalData {
    Collision { other: EntityId, normal: crate::core::Vec2 },
    Action { action: String, value: f32 },
    Custom(String),
}

/// A signal instance with source and destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub source: Option<EntityId>,
    pub target: Option<EntityId>, // None means broadcast
    pub data: SignalData,
}

/// The SignalBus manages the queue of signals.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalBus {
    queue: VecDeque<Signal>,
}

impl SignalBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit(&mut self, signal: Signal) {
        self.queue.push_back(signal);
    }

    pub fn broadcast(&mut self, source: Option<EntityId>, data: SignalData) {
        self.queue.push_back(Signal {
            source,
            target: None,
            data,
        });
    }

    pub fn send(&mut self, source: Option<EntityId>, target: EntityId, data: SignalData) {
        self.queue.push_back(Signal {
            source,
            target: Some(target),
            data,
        });
    }

    pub fn clear(&mut self) {
        self.queue.clear();
    }

    pub fn poll(&mut self) -> Option<Signal> {
        self.queue.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}
