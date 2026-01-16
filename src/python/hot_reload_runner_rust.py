# Fantasmagorie Rust - Hot Reload Runner
# ♾️ The Eternal Loop (Ouroboros Reborn)
#
# Usage: python hot_reload_runner_rust.py
# Edit test_ui.py and save to see changes instantly!

import sys
import os

# Add the release directory to path for fanta_rust.pyd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "target", "release"))

import fanta_rust as fanta
import importlib
import time
import traceback

# User's UI definition file
import test_ui

def main():
    # Create context
    ctx = fanta.Context(1280, 800)
    
    last_mtime = 0
    ui_module = test_ui
    frame_count = 0

    print("=" * 60)
    print("🐍 Ouroboros Loop Started!")
    print("   Edit 'test_ui.py' to see changes.")
    print("   Press Ctrl+C to exit.")
    print("=" * 60)

    try:
        while True:
            ctx.begin_frame()
            frame_count += 1

            # 1. File Watchdog - Check for changes
            try:
                current_mtime = os.path.getmtime(ui_module.__file__)
                if current_mtime > last_mtime:
                    if last_mtime != 0:
                        print(f"🔄 Reloading UI... (frame {frame_count})")
                        try:
                            importlib.reload(ui_module)
                            print("   ✅ Reload successful!")
                        except Exception as e:
                            print(f"   ❌ Reload failed: {e}")
                    last_mtime = current_mtime
            except Exception as e:
                print(f"Watchdog Error: {e}")

            # 2. Render UI
            try:
                # Execute user code
                ui_module.render(fanta)
            except Exception as e:
                # --- Red Screen of Death (Rust Edition) ---
                # Don't crash on error - show traceback in UI
                tb_str = traceback.format_exc()
                
                fanta.Column().bg(fanta.Color(0.2, 0.0, 0.0, 1.0)).padding(20)
                fanta.Text("🛑 Python Runtime Error").font_size(32)
                fanta.Text(str(e)).font_size(18)
                fanta.Text(tb_str).font_size(12)
                fanta.End()

            draw_count = ctx.end_frame()
            
            # Print frame info periodically
            if frame_count % 60 == 0:
                print(f"   Frame {frame_count}: {draw_count} draw commands")
            
            # Sleep to reduce CPU usage (16ms ≈ 60 FPS)
            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n👋 Ouroboros Loop Ended. Goodbye!")

if __name__ == "__main__":
    main()
