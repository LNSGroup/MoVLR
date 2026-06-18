import os
import time
import pyautogui
import subprocess

def build_cpp_project():
    build_command = [
        "/usr/bin/cmake",
        "--build", "/home/cxy/Desktop/mpc2/build",
        "--config", "Release",
        "--target", "mjpc", "--"
    ]

    # Add an environment variable to suppress unused variable warnings 
    env = os.environ.copy()
    env["CXXFLAGS"] = "-Wno-unused-variable"

    try:
        print("Building project...")
        subprocess.run(build_command, check=True, env=env)
        print("Build succeeded.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False
    return True

def run_binary(binary_path, timeout=2000):
    print(f"Running binary: {binary_path}")
    proc = subprocess.Popen([binary_path])
    try:
        proc.wait(timeout=timeout)  # Wait up to `timeout` seconds
    except subprocess.TimeoutExpired:
        print(f"Process still running after {timeout} seconds. Terminating...")
        proc.kill()
        proc.wait()
        print("Process terminated.")

def main():
    if not build_cpp_project():
        return

    gui_proc = run_binary("/home/cxy/Desktop/mpc2/build/bin/mjpc")
    print("Waiting for GUI window to appear...")
    time.sleep(2)

    try:
        print("GUI running for 120 seconds...")
        time.sleep(120)

        # Simulate Delete key press
        pyautogui.press('backspace')
        print("Backspace key pressed.")
        
        print("Terminating GUI...")
        gui_proc.terminate()
        gui_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("GUI did not exit in time. Killing it.")
        gui_proc.kill()

def build_project(args):
    """Builds the C++ project and runs the GUI application."""
    if not build_cpp_project():
        raise RuntimeError("Failed to build the C++ project.")

    run_binary(args.build_dir)
    print("GUI terminated")

if __name__ == "__main__":
    main()
