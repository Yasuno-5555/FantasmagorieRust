//! Mobile-first features
//!
//! Platform-specific features for mobile devices:
//! - IME (Input Method Editor) positioning
//! - Haptic feedback triggers
//! - Touch accessibility
//! - Safe area insets

use crate::core::Vec2;

/// IME (Input Method Editor) hints for text input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImeHint {
    /// Default text input
    Text,
    /// Email address input
    Email,
    /// URL input
    Url,
    /// Phone number input
    Phone,
    /// Numeric input
    Number,
    /// Decimal number input
    Decimal,
    /// Password input (hidden)
    Password,
    /// Search input
    Search,
    /// Multi-line text
    Multiline,
}

/// IME action button type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImeAction {
    /// Default/None
    None,
    /// Go button (form submit)
    Go,
    /// Search button
    Search,
    /// Send button
    Send,
    /// Next button (move to next field)
    Next,
    /// Done button (close keyboard)
    Done,
    /// Previous button
    Previous,
}

/// IME position request
#[derive(Debug, Clone, Copy)]
pub struct ImePosition {
    /// Position of the text cursor
    pub cursor: Vec2,
    /// Size of the input field
    pub field_size: Vec2,
    /// Preferred keyboard height (0 = auto)
    pub preferred_height: f32,
}

impl ImePosition {
    pub fn new(cursor: Vec2, field_size: Vec2) -> Self {
        Self {
            cursor,
            field_size,
            preferred_height: 0.0,
        }
    }
}

/// Haptic feedback types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HapticType {
    /// Light tap feedback
    Light,
    /// Medium tap feedback  
    Medium,
    /// Heavy tap feedback
    Heavy,
    /// Selection changed
    Selection,
    /// Impact feedback
    Impact,
    /// Error/Warning feedback
    Error,
    /// Success feedback
    Success,
    /// Continuous vibration (duration in ms)
    Vibrate(u32),
}

/// Safe area insets (for notches, home indicators, etc.)
#[derive(Debug, Clone, Copy, Default)]
pub struct SafeAreaInsets {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl SafeAreaInsets {
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    /// Get horizontal insets total
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    /// Get vertical insets total
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

/// Mobile platform abstraction
pub trait MobilePlatform {
    /// Show the soft keyboard
    fn show_keyboard(&self, hint: ImeHint, action: ImeAction, position: ImePosition);

    /// Hide the soft keyboard
    fn hide_keyboard(&self);

    /// Check if keyboard is visible
    fn is_keyboard_visible(&self) -> bool;

    /// Get the current keyboard height
    fn keyboard_height(&self) -> f32;

    /// Trigger haptic feedback
    fn haptic(&self, haptic_type: HapticType);

    /// Get safe area insets
    fn safe_area_insets(&self) -> SafeAreaInsets;

    /// Get device scale factor (for retina displays)
    fn scale_factor(&self) -> f32;

    /// Check if running on a touch device
    fn is_touch_device(&self) -> bool;
}

/// Desktop fallback implementation (no-op for most features)
pub struct DesktopPlatform {
    scale_factor: f32,
}

impl DesktopPlatform {
    pub fn new() -> Self {
        Self { scale_factor: 1.0 }
    }

    pub fn with_scale_factor(mut self, scale: f32) -> Self {
        self.scale_factor = scale;
        self
    }
}

impl Default for DesktopPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl MobilePlatform for DesktopPlatform {
    fn show_keyboard(&self, _hint: ImeHint, _action: ImeAction, _position: ImePosition) {
        // Desktop: keyboard is always available
    }

    fn hide_keyboard(&self) {
        // No-op on desktop
    }

    fn is_keyboard_visible(&self) -> bool {
        true // Physical keyboard is always "visible"
    }

    fn keyboard_height(&self) -> f32 {
        0.0 // No soft keyboard
    }

    fn haptic(&self, _haptic_type: HapticType) {
        // Desktop: could potentially use system haptics on supported hardware
        // For now, no-op
    }

    fn safe_area_insets(&self) -> SafeAreaInsets {
        SafeAreaInsets::default() // No insets on desktop
    }

    fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    fn is_touch_device(&self) -> bool {
        false
    }
}

/// Mobile platform events
#[derive(Debug, Clone)]
pub enum MobileEvent {
    /// Keyboard visibility changed
    KeyboardVisibilityChanged { visible: bool, height: f32 },
    /// Safe area changed (orientation change, etc.)
    SafeAreaChanged(SafeAreaInsets),
    /// App entered background
    AppBackground,
    /// App entered foreground
    AppForeground,
    /// Low memory warning
    LowMemory,
    /// Screen orientation changed
    OrientationChanged { is_portrait: bool },
}

/// Screen orientation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Portrait,
    PortraitUpsideDown,
    LandscapeLeft,
    LandscapeRight,
}

impl Orientation {
    pub fn is_portrait(&self) -> bool {
        matches!(
            self,
            Orientation::Portrait | Orientation::PortraitUpsideDown
        )
    }

    pub fn is_landscape(&self) -> bool {
        !self.is_portrait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_area_insets() {
        let insets = SafeAreaInsets::new(44.0, 0.0, 34.0, 0.0);
        assert_eq!(insets.horizontal(), 0.0);
        assert_eq!(insets.vertical(), 78.0);
    }

    #[test]
    fn test_desktop_platform() {
        let platform = DesktopPlatform::new();
        assert!(!platform.is_touch_device());
        assert_eq!(platform.keyboard_height(), 0.0);
        assert!(platform.is_keyboard_visible());
    }
}
