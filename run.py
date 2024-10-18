import subprocess
if __name__ == "__main__":
    # Command to run `fastapi dev app/interfaces/api.py`
    command = ["fastapi", "dev", "app/interfaces/api.py"]

    # Call the command using subprocess
    subprocess.run(command)
