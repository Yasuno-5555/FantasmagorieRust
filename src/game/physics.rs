use crate::core::Vec2;
use super::interaction::Collider;

pub struct CollisionManifold {
    pub normal: Vec2,
    pub penetration: f32,
}

pub fn check_collision(pos_a: Vec2, col_a: &Collider, pos_b: Vec2, col_b: &Collider) -> Option<CollisionManifold> {
    match (col_a, col_b) {
        (Collider::Circle { offset: oa, radius: ra }, Collider::Circle { offset: ob, radius: rb }) => {
            let p_a = pos_a + *oa;
            let p_b = pos_b + *ob;
            let dist_sq = p_a.distance_squared(p_b);
            let combined_radius = ra + rb;
            
            if dist_sq < combined_radius * combined_radius {
                let dist = dist_sq.sqrt();
                let normal = if dist > 0.0 { (p_b - p_a) / dist } else { Vec2::new(1.0, 0.0) };
                Some(CollisionManifold {
                    normal,
                    penetration: combined_radius - dist,
                })
            } else {
                None
            }
        }
        (Collider::AABB { offset: oa, size: sa }, Collider::AABB { offset: ob, size: sb }) => {
            let p_a = pos_a + *oa;
            let p_b = pos_b + *ob;
            
            let a_half = *sa * 0.5;
            let b_half = *sb * 0.5;
            let a_center = p_a + a_half;
            let b_center = p_b + b_half;
            
            let delta = b_center - a_center;
            let overlap_x = (a_half.x + b_half.x) - delta.x.abs();
            let overlap_y = (a_half.y + b_half.y) - delta.y.abs();
            
            if overlap_x > 0.0 && overlap_y > 0.0 {
                // Determine minimum penetration axis
                if overlap_x < overlap_y {
                    Some(CollisionManifold {
                        normal: if delta.x < 0.0 { Vec2::new(-1.0, 0.0) } else { Vec2::new(1.0, 0.0) },
                        penetration: overlap_x,
                    })
                } else {
                    Some(CollisionManifold {
                        normal: if delta.y < 0.0 { Vec2::new(0.0, -1.0) } else { Vec2::new(0.0, 1.0) },
                        penetration: overlap_y,
                    })
                }
            } else {
                None
            }
        }
        (Collider::Circle { offset: oa, radius: ra }, Collider::AABB { offset: ob, size: sb }) => {
            check_circle_aabb(pos_a + *oa, *ra, pos_b + *ob, *sb)
        }
        (Collider::AABB { offset: oa, size: sa }, Collider::Circle { offset: ob, radius: rb }) => {
            if let Some(mut m) = check_circle_aabb(pos_b + *ob, *rb, pos_a + *oa, *sa) {
                m.normal = m.normal * -1.0;
                Some(m)
            } else {
                None
            }
        }
        (Collider::Polygon { offset: oa, vertices: va }, Collider::Polygon { offset: ob, vertices: vb }) => {
            let abs_va: Vec<Vec2> = va.iter().map(|&v| v + pos_a + *oa).collect();
            let abs_vb: Vec<Vec2> = vb.iter().map(|&v| v + pos_b + *ob).collect();
            check_sat(&abs_va, &abs_vb)
        }
        (Collider::Circle { offset: oa, radius: ra }, Collider::Polygon { offset: ob, vertices: vb }) => {
            let p_a = pos_a + *oa;
            let abs_vb: Vec<Vec2> = vb.iter().map(|&v| v + pos_b + *ob).collect();
            check_circle_polygon(p_a, *ra, &abs_vb)
        }
        (Collider::Polygon { offset: oa, vertices: va }, Collider::Circle { offset: ob, radius: rb }) => {
            let p_b = pos_b + *ob;
            let abs_va: Vec<Vec2> = va.iter().map(|&v| v + pos_a + *oa).collect();
            if let Some(mut manifold) = check_circle_polygon(p_b, *rb, &abs_va) {
                manifold.normal = manifold.normal * -1.0;
                Some(manifold)
            } else {
                None
            }
        }
        // Handle AABB vs Polys by converting AABB to Poly on the fly or implementing specialized SAT
        (Collider::AABB { offset: oa, size: sa }, Collider::Polygon { offset: ob, vertices: vb }) => {
             let abs_va = vec![
                 pos_a + *oa,
                 pos_a + *oa + Vec2::new(sa.x, 0.0),
                 pos_a + *oa + Vec2::new(sa.x, sa.y),
                 pos_a + *oa + Vec2::new(0.0, sa.y)
             ];
             let abs_vb: Vec<Vec2> = vb.iter().map(|&v| v + pos_b + *ob).collect();
             check_sat(&abs_va, &abs_vb)
        }
        (Collider::Polygon { offset: oa, vertices: va }, Collider::AABB { offset: ob, size: sb }) => {
             let abs_va: Vec<Vec2> = va.iter().map(|&v| v + pos_a + *oa).collect();
             let abs_vb = vec![
                 pos_b + *ob,
                 pos_b + *ob + Vec2::new(sb.x, 0.0),
                 pos_b + *ob + Vec2::new(sb.x, sb.y),
                 pos_b + *ob + Vec2::new(0.0, sb.y)
             ];
             check_sat(&abs_va, &abs_vb)
        }
    }
}

fn check_circle_aabb(center: Vec2, radius: f32, aabb_pos: Vec2, aabb_size: Vec2) -> Option<CollisionManifold> {
    let aabb_center = aabb_pos + aabb_size * 0.5;
    let half_extents = aabb_size * 0.5;
    
    // Get difference vector between both centers
    let difference = center - aabb_center;
    
    // Clamp difference to AABB half extents
    let clamped = Vec2::new(
        difference.x.clamp(-half_extents.x, half_extents.x),
        difference.y.clamp(-half_extents.y, half_extents.y)
    );
    
    // Closest point on AABB boundary to circle center
    let closest = aabb_center + clamped;
    
    // Vector from closest point to circle center
    let diff = center - closest;
    let dist_sq = diff.x * diff.x + diff.y * diff.y;
    
    if dist_sq < radius * radius {
         let dist = dist_sq.sqrt();
         let normal = if dist > 0.0 { diff / dist } else { 
             // Center is inside AABB. Need to push out via shortest path.
             // Re-calculate based on clamped deviation
             if difference.x.abs() > difference.y.abs() {
                 if difference.x > 0.0 { Vec2::new(1.0, 0.0) } else { Vec2::new(-1.0, 0.0) }
             } else {
                 if difference.y > 0.0 { Vec2::new(0.0, 1.0) } else { Vec2::new(0.0, -1.0) }
             }
         };
         
         Some(CollisionManifold {
             normal,
             penetration: radius - dist,
         })
    } else {
        None
    }
}

fn check_circle_polygon(center: Vec2, radius: f32, vertices: &[Vec2]) -> Option<CollisionManifold> {
    let mut min_penetration = f32::MAX;
    let mut best_normal = Vec2::ZERO;

    // 1. Check side normals
    for i in 0..vertices.len() {
        let p1 = vertices[i];
        let p2 = vertices[(i + 1) % vertices.len()];
        let edge = p2 - p1;
        let axis = Vec2::new(-edge.y, edge.x).normalized();
        
        let (min, max) = project(vertices, axis);
        let p_center = axis.dot(center);
        
        if p_center + radius < min || p_center - radius > max {
            return None; // Gap
        }
        
        let pen = if p_center > (min + max) * 0.5 { max - (p_center - radius) } else { (p_center + radius) - min };
        if pen < min_penetration {
            min_penetration = pen;
            best_normal = axis;
        }
    }
    
    // 2. Check vertex closest to center axis
    let mut closest_dist = f32::MAX;
    let mut closest_point = Vec2::ZERO;
    for &v in vertices {
        let dist = center.distance(v);
        if dist < closest_dist {
            closest_dist = dist;
            closest_point = v;
        }
    }
    
    let axis = (closest_point - center).normalized();
    let (min, max) = project(vertices, axis);
    let p_center = axis.dot(center);
    
    if p_center + radius < min || p_center - radius > max {
        return None; 
    }
    
    let pen = if p_center > (min + max) * 0.5 { max - (p_center - radius) } else { (p_center + radius) - min };
    if pen < min_penetration {
        min_penetration = pen;
        best_normal = axis;
    }

    // Ensure normal points from Circle to Polygon
    let poly_center = get_center(vertices);
    if (poly_center - center).dot(best_normal) < 0.0 {
        best_normal = best_normal * -1.0;
    }

    Some(CollisionManifold {
        normal: best_normal,
        penetration: min_penetration,
    })
}

fn check_sat(va: &[Vec2], vb: &[Vec2]) -> Option<CollisionManifold> {
    let mut min_penetration = f32::MAX;
    let mut best_normal = Vec2::ZERO;

    let axes_a = get_axes(va);
    let axes_b = get_axes(vb);

    for axis in axes_a.into_iter().chain(axes_b.into_iter()) {
        let (min_a, max_a) = project(va, axis);
        let (min_b, max_b) = project(vb, axis);

        if max_a < min_b || max_b < min_a {
            return None; // Gap found
        }

        let penetration = (max_a - min_b).min(max_b - min_a);
        if penetration < min_penetration {
            min_penetration = penetration;
            best_normal = axis;
        }
    }

    // Ensure normal points from A to B
    let center_a = get_center(va);
    let center_b = get_center(vb);
    if (center_b - center_a).dot(best_normal) < 0.0 {
        best_normal = best_normal * -1.0;
    }

    Some(CollisionManifold {
        normal: best_normal,
        penetration: min_penetration,
    })
}

fn get_axes(vertices: &[Vec2]) -> Vec<Vec2> {
    let mut axes = Vec::new();
    for i in 0..vertices.len() {
        let p1 = vertices[i];
        let p2 = vertices[(i + 1) % vertices.len()];
        let edge = p2 - p1;
        axes.push(Vec2::new(-edge.y, edge.x).normalized());
    }
    axes
}

fn project(vertices: &[Vec2], axis: Vec2) -> (f32, f32) {
    let mut min = axis.dot(vertices[0]);
    let mut max = min;
    for &v in &vertices[1..] {
        let p = axis.dot(v);
        if p < min { min = p; }
        if p > max { max = p; }
    }
    (min, max)
}

fn get_center(vertices: &[Vec2]) -> Vec2 {
    let mut sum = Vec2::ZERO;
    for &v in vertices {
        sum = sum + v;
    }
    sum * (1.0 / vertices.len() as f32)
}

pub fn resolve_collision(
    pos_a: &mut Vec2, vel_a: &mut Vec2, mass_a: f32, res_a: f32,
    pos_b: &mut Vec2, vel_b: &mut Vec2, mass_b: f32, res_b: f32,
    manifold: &CollisionManifold
) {
    let inv_mass_a = if mass_a > 0.0 { 1.0 / mass_a } else { 0.0 };
    let inv_mass_b = if mass_b > 0.0 { 1.0 / mass_b } else { 0.0 };
    let total_inv_mass = inv_mass_a + inv_mass_b;
    
    if total_inv_mass == 0.0 { return; }
    
    // 1. Positional correction (resolve penetration)
    let percent = 0.2; // Slop
    let slop = 0.01;
    let correction = manifold.normal * (manifold.penetration - slop).max(0.0) / total_inv_mass * percent;
    *pos_a = *pos_a - correction * inv_mass_a;
    *pos_b = *pos_b + correction * inv_mass_b;
    
    // 2. Impulse resolution
    let relative_vel = *vel_b - *vel_a;
    let vel_along_normal = relative_vel.dot(manifold.normal);
    
    if vel_along_normal > 0.0 { return; } // Already separating
    
    let e = res_a.min(res_b);
    let mut j = -(1.0 + e) * vel_along_normal;
    j /= total_inv_mass;
    
    let impulse = manifold.normal * j;
    *vel_a = *vel_a - impulse * inv_mass_a;
    *vel_b = *vel_b + impulse * inv_mass_b;
}
