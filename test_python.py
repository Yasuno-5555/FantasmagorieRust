# Quick test of fanta_rust Python module
import sys
sys.path.insert(0, r"e:\Projects\FantasmagorieRust\target\release")

try:
    import fanta_rust as fanta
    print("[OK] fanta_rust module imported successfully!")
    
    # Test Color
    c = fanta.Color(0.5, 0.3, 0.8)
    print(f"[OK] Color created: {c}")
    
    # Test Context
    ctx = fanta.Context(800, 600)
    print(f"[OK] Context created: {ctx.get_width()}x{ctx.get_height()}")
    
    # Test frame cycle
    ctx.begin_frame()
    
    # Build UI
    col = fanta.Column()
    fanta.Text("Hello from Rust!")
    fanta.Button("Click Me")
    fanta.End()
    
    draw_count = ctx.end_frame()
    print(f"[OK] Frame rendered with {draw_count} draw commands")
    
    print("\n=== All tests passed! Ouroboros lives! ===")
    
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
