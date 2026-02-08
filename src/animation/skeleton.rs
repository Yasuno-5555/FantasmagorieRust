use crate::core::{Mat3, Vec2};
use crate::game::world::Transform;

#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    pub id: usize,
    pub parent_id: Option<usize>,
    pub local_transform: Transform,
    pub inverse_bind_pose: Mat3,
}

#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
    // Computed global matrices for the current frame
    pub global_poses: Vec<Mat3>, 
    pub palette: Vec<Mat3>, // Final matrices (Global * InverseBindPose) sent to GPU
}

impl Bone {
    pub fn new(name: &str, id: usize, parent_id: Option<usize>, local_transform: Transform) -> Self {
        Self {
            name: name.to_string(),
            id,
            parent_id,
            local_transform,
            inverse_bind_pose: Mat3::IDENTITY,
        }
    }
}

impl Skeleton {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_bones(bones: Vec<Bone>) -> Self {
        let count = bones.len();
        let mut s = Self {
            bones,
            global_poses: vec![Mat3::IDENTITY; count],
            palette: vec![Mat3::IDENTITY; count],
        };
        // Compute initial
        s.compute_global_pose();
        s
    }

    pub fn add_bone(&mut self, name: &str, parent_id: Option<usize>, transform: Transform) -> usize {
        let id = self.bones.len();
        let bone = Bone {
            name: name.to_string(),
            id,
            parent_id,
            local_transform: transform,
            inverse_bind_pose: Mat3::IDENTITY, 
        };
        self.bones.push(bone);
        self.global_poses.push(Mat3::IDENTITY);
        self.palette.push(Mat3::IDENTITY);
        id
    }

    /// Update global poses based on local transforms and hierarchy
    pub fn compute_global_pose(&mut self) {
        for i in 0..self.bones.len() {
            let local_mat = self.bones[i].local_transform.local_matrix();
            
            if let Some(parent_idx) = self.bones[i].parent_id {
                if parent_idx < self.global_poses.len() {
                    self.global_poses[i] = self.global_poses[parent_idx] * local_mat;
                } else {
                    self.global_poses[i] = local_mat;
                }
            } else {
                self.global_poses[i] = local_mat;
            }
            
            // Calculate final palette matrix
            self.palette[i] = self.global_poses[i] * self.bones[i].inverse_bind_pose;
        }
    }

    pub fn compute_skinning_matrices(&self) -> Vec<Mat3> {
        self.palette.clone()
    }
    
    pub fn calculate_inverse_bind_poses(&mut self) {
        self.compute_global_pose();
        for i in 0..self.bones.len() {
            self.bones[i].inverse_bind_pose = self.global_poses[i].inverse();
        }
        // Reset palette to identity
         for i in 0..self.bones.len() {
            self.palette[i] = Mat3::IDENTITY;
         }
    }

    /// Debug draw the skeleton hierarchy
    pub fn draw_debug(&self, dl: &mut crate::draw::DrawList, color: crate::core::ColorF) {
        for i in 0..self.bones.len() {
            let bone = &self.bones[i];
            if i >= self.global_poses.len() { continue; }
            
            let global = self.global_poses[i];
            let pos = global.transform_point(Vec2::ZERO);
            
            dl.add_circle(pos, 4.0, color, true);
            
            if let Some(parent_id) = bone.parent_id {
                if parent_id < self.global_poses.len() {
                    let parent_global = self.global_poses[parent_id];
                    let parent_pos = parent_global.transform_point(Vec2::ZERO);
                    dl.add_line(parent_pos, pos, 2.0, crate::core::ColorF::WHITE);
                }
            }
        }
    }
}
