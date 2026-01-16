import subprocess
import os

def run_check():
    cmd = ["cargo", "check", "--example", "theme_demo", "--features", "opengl"]
    try:
        # Popen to capture everything
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd="e:\\Projects\\FantasmagorieRust")
        stdout, stderr = process.communicate()
        
        with open("full_errors.txt", "w", encoding='utf-8') as f:
            f.write("--- STDOUT ---\n")
            f.write(stdout)
            f.write("\n--- STDERR ---\n")
            f.write(stderr)
        
        # Also print first 100 lines of stderr to the console for immediate viewing
        lines = stderr.splitlines()
        print("\n".join(lines[:100]))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_check()
