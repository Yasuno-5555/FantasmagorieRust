# Fantasmagorie Rust - Windowed Hot Reload Runner
# ♾️ Phase 9: First Light
#
# This version uses the Inversion of Control pattern:
# Rust owns the window and event loop, Python provides the render callback.

import sys
import os

# Add the release directory to path for fanta_rust.pyd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "target", "release"))

import fanta_rust as fanta
import importlib
import traceback

# User's UI definition file
import test_ui

class HotReloader:
    """Manages hot-reloading of the UI module"""
    
    def __init__(self, module):
        self.module = module
        self.last_mtime = 0
        self.error_state = None
        
    def check_reload(self):
        """Check if source file changed and reload if needed"""
        try:
            current_mtime = os.path.getmtime(self.module.__file__)
            if current_mtime > self.last_mtime:
                if self.last_mtime != 0:
                    print("🔄 Reloading UI...")
                    try:
                        importlib.reload(self.module)
                        self.error_state = None
                        print("   ✅ Reload successful!")
                    except Exception as e:
                        self.error_state = traceback.format_exc()
                        print(f"   ❌ Reload failed: {e}")
                self.last_mtime = current_mtime
        except Exception as e:
            print(f"Watchdog error: {e}")

# Global reloader instance
reloader = HotReloader(test_ui)

def render_callback(width, height):
    """Called by Rust every frame"""
    
    # Check for hot reload
    reloader.check_reload()
    
    # Render
    if reloader.error_state:
        # Red Screen of Death
        fanta.Column().padding(20).bg(fanta.Color(0.3, 0.0, 0.0, 1.0))
        fanta.Text("🛑 Python Error").font_size(28)
        fanta.Text(reloader.error_state).font_size(12)
        fanta.End()
    else:
        try:
            reloader.module.render(fanta, width, height)
        except Exception as e:
            reloader.error_state = traceback.format_exc()
            print(f"Render error: {e}")

def main():
    print("=" * 60)
    print("🦀🐍 Fantasmagorie Rust - Windowed Mode")
    print("   Edit 'test_ui.py' to see changes!")
    print("   Close window to exit.")
    print("=" * 60)
    
    # Run the windowed application
    # This blocks until window is closed
    fanta.run_window(1280, 800, "Fantasmagorie Rust 🦀", render_callback)
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
