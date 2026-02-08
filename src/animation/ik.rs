use crate::core::Vec2;

/// Inverse Kinematics Solver specialized for 2D.
pub struct IKSolver;

impl IKSolver {
    /// Analytical Two-Bone IK (Law of Cosines).
    /// Returns (angle_root, angle_joint) in radians.
    ///
    /// # Arguments
    /// * `root_pos` - Position of the root joint.
    /// * `len1` - Length of the first bone (Root -> Joint).
    /// * `len2` - Length of the second bone (Joint -> End).
    /// * `target` - Target position for the end effector.
    /// * `bend_right` - Preferred bending direction (true = clockwise/right, false = counter-clockwise/left).
    pub fn solve_two_bone(root_pos: Vec2, len1: f32, len2: f32, target: Vec2, bend_right: bool) -> Option<(f32, f32)> {
        let diff = target - root_pos;
        let dist = diff.length();
        let dist_sq = dist * dist;
        
        // Cannot reach?
        if dist >= len1 + len2 {
            // Fully extended towards target
            let angle = diff.y.atan2(diff.x);
            return Some((angle, 0.0));
        }
        
        // Too close? (Triangle inequality)
        if dist <= (len1 - len2).abs() {
            // Folded back?
             return None;
        }

        // Law of Cosines
        // c^2 = a^2 + b^2 - 2ab cos(C)
        // dist^2 = len1^2 + len2^2 - 2*len1*len2 * cos(JointAngle)
        // cos(JointAngle) = (len1^2 + len2^2 - dist^2) / (2 * len1 * len2)
        
        let cos_joint = (len1 * len1 + len2 * len2 - dist_sq) / (2.0 * len1 * len2);
        // Clamp for safety
        let cos_joint = cos_joint.clamp(-1.0, 1.0);
        
        // Joint angle is roughly PI - internal_angle.
        // But in 2D bone space, 0 is straight.
        // If cos_joint comes from internal angle of triangle (opposite to dist).
        // Wait, standard cos law finds angle OPPOSITE the side.
        // We want angle at Root (opposite len2) and angle at Joint (opposite dist... wait no).
        
        // Let's find Angle A (at Root, opposite len2) and Angle C (internal, opposite dist... no).
        // Triangle sides: a=len2, b=len1, c=dist.
        // Angle at Root (A) is opposite a=len2.
        // cos(A) = (b^2 + c^2 - a^2) / (2bc)
        let cos_root_internal = (len1 * len1 + dist_sq - len2 * len2) / (2.0 * len1 * dist);
        let angle_root_internal = cos_root_internal.clamp(-1.0, 1.0).acos();
        
        // Angle of the target vector
        let angle_to_target = diff.y.atan2(diff.x);
        
        // Root angle
        let angle1 = if bend_right {
            angle_to_target + angle_root_internal
        } else {
            angle_to_target - angle_root_internal
        };
        
        // Angle at Joint (B) internal?
        // cos(B) = (a^2 + c^2 - b^2) ... no, side c is dist.
        // Angle at Joint is opposite c=dist.
        // cos(JointInternal) = (a^2 + b^2 - c^2) / (2ab) = (len2^2 + len1^2 - dist^2) / (2*len1*len2)
        let cos_joint_internal = (len1 * len1 + len2 * len2 - dist_sq) / (2.0 * len1 * len2);
        let angle_joint_internal = cos_joint_internal.clamp(-1.0, 1.0).acos();
        
        // The joint rotation is the deviation from straight line.
        // Straight = PI internal angle. 
        // Bending right means negative relative rotation?
        // Let's return relative angle required for bone 2.
        // If straight, angle2 = 0.
        // If bent 90 degrees, angle2 depends on system.
        // Usually: PI - internal_angle.
        let angle2 = std::f32::consts::PI - angle_joint_internal;
        
        let angle2 = if bend_right { -angle2 } else { angle2 };
        
        Some((angle1, angle2))
    }

    /// Cyclic Coordinate Descent (CCD) for N-link chain.
    /// Modifies `joints` (global positions) and output `angles`.
    /// 
    /// Note: This is an iterative global solver.
    /// In a real ECS, you'd apply this to Transforms.
    /// 
    /// # Arguments
    /// * `joints` - Mutable slice of global positions of joints (Root -> End).
    /// * `target` - Target position.
    /// * `tolerance` - Distance threshold.
    /// * `max_iter` - Maximum iterations.
    pub fn solve_ccd(joints: &mut [Vec2], target: Vec2, tolerance: f32, max_iter: usize) {
        let count = joints.len();
        if count < 2 { return; }
        
        let dist_sq_threshold = tolerance * tolerance;
        
        for _ in 0..max_iter {
            let end_effector = *joints.last().unwrap();
            if end_effector.distance_squared(target) < dist_sq_threshold {
                return;
            }
            
            // Iterate from second-to-last joint back to root
            for i in (0..count - 1).rev() {
                let pivot = joints[i];
                let end = *joints.last().unwrap();
                
                let to_end = (end - pivot).normalized();
                let to_target = (target - pivot).normalized();
                
                // Angle to rotate = angle between vectors
                let dot = to_end.dot(to_target).clamp(-1.0, 1.0);
                let angle = dot.acos();
                
                // Cross product z-component to determine direction (2D)
                let cross = to_end.x * to_target.y - to_end.y * to_target.x;
                
                let rotation = if cross > 0.0 { angle } else { -angle };
                
                // Clamp rotation per step for stability? Optional.
                
                // Rotate all subsequent joints around this pivot
                let cos_r = rotation.cos();
                let sin_r = rotation.sin();
                
                for j in (i + 1)..count {
                    let mut rel = joints[j] - pivot;
                    let x_new = rel.x * cos_r - rel.y * sin_r;
                    let y_new = rel.x * sin_r + rel.y * cos_r;
                    joints[j] = pivot + Vec2::new(x_new, y_new);
                }
            }
        }
    }
}
