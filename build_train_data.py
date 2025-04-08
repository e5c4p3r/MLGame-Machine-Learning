import os
import subprocess
import time
import psutil
import shutil

# terminate the game process
def kill_process_tree(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.terminate()
    parent.terminate()

def main():
    play_cmd = "python -m mlgame -f 600 -i ./ml/ml_play_automatic.py . --difficulty NORMAL --level "

    # create folder
    train_data_folder = os.path.join("ml", "train_data")
    if not os.path.exists(train_data_folder):
        os.makedirs(train_data_folder)

    # delete old data
    if os.path.exists("train_data.pkl"):
        os.remove("train_data.pkl")

    repeat_times = 5
    for level in range(1, 24):
        print(f"start level {level}")
        count = 0
        while count < repeat_times:
            command = play_cmd + str(level)

            game = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            while not os.path.exists("train_data.pkl"):
                time.sleep(1)

            # Small delay to make sure file writing finishes
            time.sleep(1)

            count += 1
            filename = os.path.join(train_data_folder, f"{level}_{count}.pkl")

            shutil.move("train_data.pkl", filename)
            
            kill_process_tree(game.pid)
    print("Finished!")

if __name__ == "__main__":
    main()