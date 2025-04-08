import os
import pickle
import subprocess
import time
import psutil
from sklearn.metrics import accuracy_score

def kill_process_tree(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.terminate()
    parent.terminate()

def main():
    play_cmd = "python -m mlgame -f 600 -i ./ml/ml_play_model.py . --difficulty NORMAL --level "
    rule_actions_path = "rule_actions.pkl"
    model_actions_path = "model_actions.pkl"

    pass_count = 0
    fail_count = 0
    accuracy_list = []

    play_times = 3
    for level in range(1, 24):
        print(f"start level {level}")
        count = 0
        while count < play_times:
            command = play_cmd + str(level)

            game = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            line = None
            while not line in ["GAME_OVER\n", "GAME_PASS\n"]:
                time.sleep(1)
                line = game.stdout.readline()
            
            while not os.path.exists(rule_actions_path):
                time.sleep(1)

            while not os.path.exists(model_actions_path):
                time.sleep(1)

            time.sleep(2)

            count += 1

            print(f"{count} : {line}", end="")

            if line == "GAME_PASS\n":
                pass_count += 1
            elif line == "GAME_OVER\n":
                fail_count += 1

            with open(rule_actions_path, "rb") as f:
                rule_actions = pickle.load(f)
            with open(model_actions_path, "rb") as f:
                model_actions = pickle.load(f)

            accuracy = accuracy_score(rule_actions, model_actions)
            accuracy_list.append(accuracy)
            print(f"Accuracy: {accuracy:.3f}")

            kill_process_tree(game.pid)
    
    print("Finished!")

    print(f"Pass count: {pass_count}")
    print(f"Fail count: {fail_count}")
    print(f"Success rate: {pass_count / (pass_count + fail_count) * 100:.2f}%")

    # Calculate average accuracy
    if accuracy_list:
        average_accuracy = sum(accuracy_list) / len(accuracy_list)
        print(f"Average Accuracy: {average_accuracy:.3f}")
    else:
        print("No accuracy data available.")

if __name__ == "__main__":
    main()