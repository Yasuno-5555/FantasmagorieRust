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
            let dist = p_a.distance(p_b);
            let combined_radius = ra + rb;
            
            if dist < combined_radius {
                let normal = (p_b - p_a).normalized();
                Some(CollisionManifold {
                    normal,
                    penetration: combined_radius - dist,
                })
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
        _ => None,
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
