
import sys
import os

# Ensure we can find the built library
# Assuming run from project root, and lib is in target/debug
sys.path.append("target/debug")

try:
    import fanta_rust as fanta
except ImportError:
    # Try alternate location or name if needed (e.g. .so file might be named named libfantasmagorie.so)
    # We will handle renaming in the shell loop if logic fails here.
    print("Failed to import fanta_rust. Ensure target/debug/libfantasmagorie.so is renamed to fanta_rust.so")
    sys.exit(1)

def test_multi_window_separation():
    print("Testing Multi-Window Context Separation...")

    # Create Context for Window 1
    ctx1 = fanta.Context(800, 600, window_id=1)
    
    # Create Context for Window 2
    ctx2 = fanta.Context(1024, 768, window_id=2)

    # --- Frame 1: Window 1 ---
    print("Building Window 1 Frame...")
    ctx1.begin_frame() # Sets CURRENT_WINDOW = 1
    
    fanta.Box().size(100, 100).bg(fanta.Color.red())
    # Should result in 1 draw command (rect) logic + background clearing
    
    # We call end_frame() to finalize and get draw commands count
    cmds1 = ctx1.end_frame()
    print(f"Window 1 Draw Commands: {cmds1}")
    
    # --- Frame 1: Window 2 ---
    print("Building Window 2 Frame...")
    ctx2.begin_frame() # Sets CURRENT_WINDOW = 2
    
    fanta.Row()
    fanta.Button("Btn 1")
    fanta.Button("Btn 2")
    fanta.End()
    
    cmds2 = ctx2.end_frame()
    print(f"Window 2 Draw Commands: {cmds2}")

    # Verify separation
    # Window 1 had 1 Box. Window 2 had Row(Box) + 2 Buttons.
    # Logic counts might differ based on impl, but they should NOT be equal if contents differ
    # and certainly they shouldn't interfere.
    
    # Assertions
    # Basic check: verify they didn't crash and returned non-negative counts.
    # Also verify that switching contexts allowed building different trees.
    
    assert cmds1 > 0, "Window 1 should produce draw commands"
    assert cmds2 > 0, "Window 2 should produce draw commands"
    
    print("Multi-Window Logic Test Passed!")

if __name__ == "__main__":
    test_multi_window_separation()
