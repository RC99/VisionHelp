import subprocess
import threading
import time

def run_app_py():
    """Run app.py as a subprocess."""
    subprocess.run(["python", "app.py"])

def run_map_py():
    """Run map.py as a subprocess."""
    subprocess.run(["python", "map.py"])

def main():
    print("Press 1 to run app.py")
    print("Press 2 to run map.py")
    print("Press 3 to exit")

    while True:
        user_input = input("> ").strip()

        if user_input == "1":
            print("Starting app.py...")
            # Run app.py in a separate thread or process
            threading.Thread(target=run_app_py, daemon=True).start()

        elif user_input == "2":
            print("Starting map.py...")
            # Run map.py in a separate thread or process
            threading.Thread(target=run_map_py, daemon=True).start()

        elif user_input == "3":
            print("Exiting...")
            break

        else:
            print("Invalid input. Please press 1, 2, or 3.")

if __name__ == "__main__":
    main()
