import subprocess
import os

def run_check():
    cmd = ["cargo", "check", "--example", "theme_demo", "--features", "opengl"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="e:\\Projects\\FantasmagorieRust", encoding='utf-8')
        with open("compilation_errors.log", "w", encoding='utf-8') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        print("Done. Errors written to compilation_errors.log")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_check()
