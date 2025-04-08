import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

def main():
    train_data_folder = os.path.join("ml", "train_data")

    all_train_data = []
    for level in range(1, 24):
        for count in range(1, 6):
            filename = os.path.join(train_data_folder, f"{level}_{count}.pkl")
            with open(filename, "rb") as f:
                train_data = pickle.load(f)
                all_train_data.extend(train_data)
    print(f"Total training data count: {len(all_train_data)}")

    x = []  # features
    y = []  # labels

    # extract features and labels
    for state, action in all_train_data:
        x.append(state)
        y.append(action)

    x = np.array(x)
    y = np.array(y)

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x, y)
    
    with open("ml/models/knn.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Finished!")
    
if __name__ == "__main__":
    main()

