use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct LutData {
    pub size: u32,
    pub data: Vec<f32>, // RGBA float data (flat)
}

impl LutData {
    pub fn load_from_cube(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open LUT: {}", e))?;
        let reader = BufReader::new(file);

        let mut size = 0;
        let mut data = Vec::new();
        let mut min_val = vec![f32::MAX; 3];
        let mut max_val = vec![f32::MIN; 3];

        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let line = line.trim();
            
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    size = parts[1].parse::<u32>().map_err(|_| "Invalid LUT size")?;
                }
                continue;
            }

            if line.starts_with("TITLE") || line.starts_with("DOMAIN_") {
                continue;
            }

            // Parse RGB values
            let coords: Vec<&str> = line.split_whitespace().collect();
            if coords.len() >= 3 {
                let r = coords[0].parse::<f32>().unwrap_or(0.0);
                let g = coords[1].parse::<f32>().unwrap_or(0.0);
                let b = coords[2].parse::<f32>().unwrap_or(0.0);
                
                data.push(r);
                data.push(g);
                data.push(b);
                data.push(1.0); // Alpha
            }
        }

        if size == 0 {
            // Auto-detect size if not specified (assuming perfect cube)
            // count = size^3
            let count = data.len() / 4;
            let s = (count as f32).powf(1.0/3.0).round() as u32;
            if s * s * s == count as u32 {
                size = s;
            } else {
                return Err("Failed to determine LUT size".to_string());
            }
        }

        Ok(LutData { size, data })
    }
    
    // Generate a default identity LUT for testing
    pub fn identity(size: u32) -> Self {
        let mut data = Vec::with_capacity((size * size * size * 4) as usize);
        let s = size as f32 - 1.0;
        
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    data.push(x as f32 / s);
                    data.push(y as f32 / s);
                    data.push(z as f32 / s);
                    data.push(1.0);
                }
            }
        }
        
        LutData { size, data }
    }
}
