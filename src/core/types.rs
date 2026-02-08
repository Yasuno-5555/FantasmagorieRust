//! Core types: ColorF, Vec2, Rectangle
//! Ported from types_core.hpp

use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Sub};

/// Unique identifier for an OS window
/// Used for multi-window context management
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WindowID(pub u64);

impl WindowID {
    /// The default window ID (0)
    pub const MAIN: Self = Self(0);
}

impl From<u64> for WindowID {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

/// RGBA color with floating-point components [0.0, 1.0]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ColorF {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl ColorF {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const fn rgba_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    pub fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 24) & 0xFF) as f32 / 255.0,
            g: ((hex >> 16) & 0xFF) as f32 / 255.0,
            b: ((hex >> 8) & 0xFF) as f32 / 255.0,
            a: (hex & 0xFF) as f32 / 255.0,
        }
    }

    /// Alias for from_hex
    pub fn hex(hex: u32) -> Self {
        Self::from_hex(hex)
    }

    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    // Predefined colors
    pub const WHITE: ColorF = ColorF::new(1.0, 1.0, 1.0, 1.0);
    pub const BLACK: ColorF = ColorF::new(0.0, 0.0, 0.0, 1.0);
    pub const TRANSPARENT: ColorF = ColorF::new(0.0, 0.0, 0.0, 0.0);
    pub const RED: ColorF = ColorF::new(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: ColorF = ColorF::new(0.0, 1.0, 0.0, 1.0);
    pub const BLUE: ColorF = ColorF::new(0.0, 0.0, 1.0, 1.0);

    // Keep function versions for backwards compatibility
    pub const fn white() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }
    pub const fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }
    pub const fn transparent() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
    pub const fn red() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }
    pub const fn green() -> Self {
        Self::new(0.0, 1.0, 0.0, 1.0)
    }
    pub const fn blue() -> Self {
        Self::new(0.0, 0.0, 1.0, 1.0)
    }

    pub fn lighten(self, amount: f32) -> Self {
        Self {
            r: self.r + (1.0 - self.r) * amount,
            g: self.g + (1.0 - self.g) * amount,
            b: self.b + (1.0 - self.b) * amount,
            a: self.a,
        }
    }

    pub fn darken(self, amount: f32) -> Self {
        Self {
            r: self.r * (1.0 - amount),
            g: self.g * (1.0 - amount),
            b: self.b * (1.0 - amount),
            a: self.a,
        }
    }

    pub fn mix(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

/// HSV color (for ColorPicker)
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub struct HSV {
    pub h: f32, // 0-360
    pub s: f32, // 0-1
    pub v: f32, // 0-1
    pub a: f32, // 0-1
}

impl HSV {
    pub fn from_rgb(c: ColorF) -> Self {
        let mx = c.r.max(c.g).max(c.b);
        let mn = c.r.min(c.g).min(c.b);
        let d = mx - mn;

        let v = mx;
        let s = if mx > 0.0 { d / mx } else { 0.0 };
        let h = if d < 0.00001 {
            0.0
        } else if mx == c.r {
            60.0 * ((c.g - c.b) / d).rem_euclid(6.0)
        } else if mx == c.g {
            60.0 * ((c.b - c.r) / d + 2.0)
        } else {
            60.0 * ((c.r - c.g) / d + 4.0)
        };

        Self { h, s, v, a: c.a }
    }

    pub fn to_rgb(self) -> ColorF {
        let c = self.v * self.s;
        let x = c * (1.0 - ((self.h / 60.0).rem_euclid(2.0) - 1.0).abs());
        let m = self.v - c;

        let (r, g, b) = if self.h < 60.0 {
            (c, x, 0.0)
        } else if self.h < 120.0 {
            (x, c, 0.0)
        } else if self.h < 180.0 {
            (0.0, c, x)
        } else if self.h < 240.0 {
            (0.0, x, c)
        } else if self.h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        ColorF::new(r + m, g + m, b + m, self.a)
    }
}

/// 2D vector
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
    
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        } else {
            Self::ZERO
        }
    }
}

/// 3D vector
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            self / len
        } else {
            Self::ZERO
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

/// Axis-aligned rectangle
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Rectangle {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rectangle {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        w: 0.0,
        h: 0.0,
    };

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }

    pub fn pos(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub fn size(&self) -> Vec2 {
        Vec2::new(self.w, self.h)
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsv_rgb_roundtrip() {
        let original = ColorF::new(0.8, 0.3, 0.5, 1.0);
        let hsv = HSV::from_rgb(original);
        let back = hsv.to_rgb();
        assert!((original.r - back.r).abs() < 0.001);
        assert!((original.g - back.g).abs() < 0.001);
        assert!((original.b - back.b).abs() < 0.001);
    }

    #[test]
    fn test_rectangle_contains() {
        let rect = Rectangle::new(10.0, 20.0, 100.0, 50.0);
        assert!(rect.contains(50.0, 40.0));
        assert!(!rect.contains(5.0, 40.0));
    }

    #[test]
    fn test_mat3_mul() {
        let m1 = Mat3::translation(10.0, 20.0);
        let m2 = Mat3::scale(2.0, 2.0);
        let m3 = m1 * m2;
        let v = m3 * Vec2::new(1.0, 1.0);
        // (1.0, 1.0) * Scale(2,2) = (2,2) -> Translate(10,20) = (12, 22)
        assert!((v.x - 12.0).abs() < 0.001);
        assert!((v.y - 22.0).abs() < 0.001);
    }
}

/// 3x3 Matrix for 2D affine transformations
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mat3(pub [f32; 9]);

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat3 {
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);

    pub fn translation(x: f32, y: f32) -> Self {
        Self([
            1.0, 0.0, x,
            0.0, 1.0, y,
            0.0, 0.0, 1.0,
        ])
    }

    pub fn rotation(angle_rad: f32) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self([
            c,   -s,  0.0,
            s,    c,  0.0,
            0.0, 0.0, 1.0,
        ])
    }

    pub fn scale(sx: f32, sy: f32) -> Self {
        Self([
            sx,  0.0, 0.0,
            0.0, sy,  0.0,
            0.0, 0.0, 1.0,
        ])
    }

    pub fn transform_point(&self, p: Vec2) -> Vec2 {
        Vec2 {
            x: self.0[0] * p.x + self.0[1] * p.y + self.0[2],
            y: self.0[3] * p.x + self.0[4] * p.y + self.0[5],
        }
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        Self([
            a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
            a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
            a[0] * b[2] + a[1] * b[5] + a[2] * b[8],

            a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
            a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
            a[3] * b[2] + a[4] * b[5] + a[5] * b[8],

            a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
            a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
            a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
        ])
    }
}

impl Mul<Vec2> for Mat3 {
    type Output = Vec2;
    fn mul(self, rhs: Vec2) -> Vec2 {
        self.transform_point(rhs)
    }
}
