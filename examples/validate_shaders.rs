use naga::front::wgsl;
use std::fs;
use std::path::Path;

fn main() {
    let source_dir = Path::new("src/backend/shaders");
    
    // List of files to check
    let files = [
        "k5_jfa.wgsl",
        "k5_resolve.wgsl",
        "seed.wgsl",
        "k4_resolver.wgsl", // Check others just in case
    ];

    println!("Starting WGSL validation...");

    for file_name in files {
        let path = source_dir.join(file_name);
        println!("Checking {:?}...", path);

        match fs::read_to_string(&path) {
            Ok(source) => {
                match wgsl::parse_str(&source) {
                    Ok(module) => {
                        println!("  [OK] Valid WGSL. Entry points: {:?}", 
                            module.entry_points.iter().map(|e| &e.name).collect::<Vec<_>>());
                        
                        // Also check validation (capabilities, types)
                        let mut validator = naga::valid::Validator::new(
                            naga::valid::ValidationFlags::all(),
                            naga::valid::Capabilities::all(),
                        );
                        match validator.validate(&module) {
                             Ok(_) => println!("  [OK] Validation passed."),
                             Err(e) => println!("  [FAIL] Validation Error: {:?}", e),
                        }
                    },
                    Err(e) => {
                        println!("  [FAIL] Parse Error in {:?}:", path);
                        // Print the error directly
                        println!("{}", e);
                        // Try to find location
                        if let Some(range) = e.location(&source) {
                             println!("  At character range: {:?}", range);
                             // Print context
                             let start = range.offset as usize;
                             let end = (range.offset + range.length) as usize;
                             // Show some context around it
                             let context_start = start.saturating_sub(50);
                             let context_end = (end + 50).min(source.len());
                             println!("  Context: \"...{}...\"", &source[context_start..context_end]);
                        }
                    }
                }
            },
            Err(e) => {
                 println!("  [Checking Skipped] File not found or unreadable: {}", e);
            }
        }
        println!("---------------------------------------------------");
    }
}
