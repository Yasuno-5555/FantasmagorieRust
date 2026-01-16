//! Node-Wire interaction system
//!
//! Provides connection handling for node-based editors:
//! - Port hit testing
//! - Ghost wire rendering
//! - Connection validation
//! - Wire dragging

use crate::core::Vec2;

/// Port type (input or output)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortType {
    Input,
    Output,
}

/// Port identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortId {
    pub node_id: usize,
    pub port_index: usize,
    pub port_type: PortType,
}

impl PortId {
    pub fn new(node_id: usize, port_index: usize, port_type: PortType) -> Self {
        Self { node_id, port_index, port_type }
    }

    pub fn input(node_id: usize, port_index: usize) -> Self {
        Self::new(node_id, port_index, PortType::Input)
    }

    pub fn output(node_id: usize, port_index: usize) -> Self {
        Self::new(node_id, port_index, PortType::Output)
    }
}

/// Port position and metadata
#[derive(Debug, Clone)]
pub struct Port {
    pub id: PortId,
    pub position: Vec2,
    pub data_type: String,
    pub name: String,
    pub connected: bool,
}

impl Port {
    pub fn new(id: PortId, position: Vec2, data_type: &str, name: &str) -> Self {
        Self {
            id,
            position,
            data_type: data_type.to_string(),
            name: name.to_string(),
            connected: false,
        }
    }
}

/// Connection between two ports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Connection {
    pub from: PortId,
    pub to: PortId,
}

impl Connection {
    pub fn new(from: PortId, to: PortId) -> Self {
        Self { from, to }
    }
}

/// State of wire interaction
#[derive(Debug, Clone)]
pub enum WireState {
    /// No wire interaction
    Idle,
    /// Dragging from a port
    Dragging {
        start_port: PortId,
        start_pos: Vec2,
        end_pos: Vec2,
    },
    /// Hovering over a valid target port
    Targeting {
        start_port: PortId,
        start_pos: Vec2,
        target_port: PortId,
        target_pos: Vec2,
    },
}

/// Wire interaction handler
pub struct WireInteraction {
    state: WireState,
    /// Hit test radius for ports
    port_radius: f32,
    /// Current hover port
    hover_port: Option<PortId>,
}

/// Result of a connection attempt
#[derive(Debug, Clone)]
pub enum ConnectionResult {
    /// Connection is valid and can be made
    Valid(Connection),
    /// Connection is invalid (e.g., same node, type mismatch)
    Invalid { reason: String },
    /// No connection attempted
    None,
}

impl WireInteraction {
    /// Create a new wire interaction handler
    pub fn new() -> Self {
        Self {
            state: WireState::Idle,
            port_radius: 8.0,
            hover_port: None,
        }
    }

    /// Set the port hit test radius
    pub fn with_port_radius(mut self, radius: f32) -> Self {
        self.port_radius = radius;
        self
    }

    /// Start dragging a wire from a port
    pub fn start_drag(&mut self, port: PortId, pos: Vec2) {
        self.state = WireState::Dragging {
            start_port: port,
            start_pos: pos,
            end_pos: pos,
        };
    }

    /// Update the wire drag position
    pub fn update_drag(&mut self, pos: Vec2, ports: &[Port]) {
        match &self.state {
            WireState::Dragging { start_port, start_pos, .. } |
            WireState::Targeting { start_port, start_pos, .. } => {
                let start_port = *start_port;
                let start_pos = *start_pos;
                
                // Check for hover over valid target port
                if let Some(target) = self.find_port_at(pos, ports) {
                    if self.can_connect(&start_port, &target.id) {
                        self.state = WireState::Targeting {
                            start_port,
                            start_pos,
                            target_port: target.id,
                            target_pos: target.position,
                        };
                        self.hover_port = Some(target.id);
                        return;
                    }
                }
                
                self.hover_port = None;
                self.state = WireState::Dragging {
                    start_port,
                    start_pos,
                    end_pos: pos,
                };
            }
            WireState::Idle => {}
        }
    }

    /// End the wire drag and potentially create a connection
    pub fn end_drag(&mut self, pos: Vec2, ports: &[Port]) -> ConnectionResult {
        let result = match &self.state {
            WireState::Targeting { start_port, target_port, .. } => {
                let from = *start_port;
                let to = *target_port;
                
                // Ensure output -> input direction
                let (from, to) = if from.port_type == PortType::Output {
                    (from, to)
                } else {
                    (to, from)
                };
                
                ConnectionResult::Valid(Connection::new(from, to))
            }
            WireState::Dragging { start_port, .. } => {
                // Check if we're over a port
                if let Some(target) = self.find_port_at(pos, ports) {
                    if self.can_connect(start_port, &target.id) {
                        let from = *start_port;
                        let to = target.id;
                        
                        let (from, to) = if from.port_type == PortType::Output {
                            (from, to)
                        } else {
                            (to, from)
                        };
                        
                        ConnectionResult::Valid(Connection::new(from, to))
                    } else {
                        ConnectionResult::Invalid {
                            reason: "Incompatible ports".to_string(),
                        }
                    }
                } else {
                    ConnectionResult::None
                }
            }
            WireState::Idle => ConnectionResult::None,
        };
        
        self.state = WireState::Idle;
        self.hover_port = None;
        result
    }

    /// Cancel the current wire drag
    pub fn cancel(&mut self) {
        self.state = WireState::Idle;
        self.hover_port = None;
    }

    /// Find a port at the given position
    pub fn find_port_at<'a>(&self, pos: Vec2, ports: &'a [Port]) -> Option<&'a Port> {
        ports.iter().find(|port| {
            let dist = (port.position - pos).length();
            dist <= self.port_radius
        })
    }

    /// Check if two ports can be connected
    pub fn can_connect(&self, from: &PortId, to: &PortId) -> bool {
        // Can't connect to same node
        if from.node_id == to.node_id {
            return false;
        }
        
        // Must be different port types (input <-> output)
        if from.port_type == to.port_type {
            return false;
        }
        
        // TODO: Add data type compatibility check
        true
    }

    /// Get the current wire state
    pub fn state(&self) -> &WireState {
        &self.state
    }

    /// Check if a wire is being dragged
    pub fn is_dragging(&self) -> bool {
        !matches!(self.state, WireState::Idle)
    }

    /// Get the current hover port
    pub fn hover_port(&self) -> Option<PortId> {
        self.hover_port
    }

    /// Get wire endpoints for rendering (if dragging)
    pub fn get_wire_endpoints(&self) -> Option<(Vec2, Vec2)> {
        match &self.state {
            WireState::Dragging { start_pos, end_pos, .. } => Some((*start_pos, *end_pos)),
            WireState::Targeting { start_pos, target_pos, .. } => Some((*start_pos, *target_pos)),
            WireState::Idle => None,
        }
    }

    /// Check if currently targeting a valid port
    pub fn is_targeting(&self) -> bool {
        matches!(self.state, WireState::Targeting { .. })
    }
}

impl Default for WireInteraction {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate control points for a bezier wire
pub fn wire_control_points(start: Vec2, end: Vec2) -> (Vec2, Vec2) {
    let tangent_length = ((end.x - start.x).abs() * 0.5).max(50.0);
    
    let cp1 = Vec2::new(start.x + tangent_length, start.y);
    let cp2 = Vec2::new(end.x - tangent_length, end.y);
    
    (cp1, cp2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_connect() {
        let interaction = WireInteraction::new();
        
        let output = PortId::output(0, 0);
        let input = PortId::input(1, 0);
        let same_node_input = PortId::input(0, 1);
        let another_output = PortId::output(1, 0);
        
        assert!(interaction.can_connect(&output, &input));
        assert!(!interaction.can_connect(&output, &same_node_input)); // Same node
        assert!(!interaction.can_connect(&output, &another_output)); // Both outputs
    }

    #[test]
    fn test_wire_control_points() {
        let start = Vec2::new(0.0, 100.0);
        let end = Vec2::new(200.0, 100.0);
        
        let (cp1, cp2) = wire_control_points(start, end);
        
        assert!(cp1.x > start.x);
        assert!(cp2.x < end.x);
        assert_eq!(cp1.y, start.y);
        assert_eq!(cp2.y, end.y);
    }
}
