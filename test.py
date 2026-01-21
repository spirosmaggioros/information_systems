import argparse

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--filenames", nargs="+", type=str)

    args = parser.parse_args()
    gl_train_acc = []
    gl_train_auc = []
    gl_train_f1 = []

    gl_test_acc = []
    gl_test_auc = []
    gl_test_f1 = []

    for filename in args.filenames:
        stats = {
            "train_acc": 0.0,
            "train_auc": 0.0,
            "train_f1": 0.0,
            "test_acc": 0.0,
            "test_auc": 0.0,
            "test_f1": 0.0,
        }

        with open(filename, "r") as f:
            lines = f.readlines()

            for line in lines:
                comp = line.strip().split()

                if "train_acc:" in comp:
                    train_acc = float(comp[comp.index("train_acc:") + 1]) * 100
                    train_auc = float(comp[comp.index("train_auc:") + 1]) * 100
                    train_f1 = float(comp[comp.index("train_f1:") + 1]) * 100
                    test_acc = float(comp[comp.index("test_acc:") + 1]) * 100
                    test_auc = float(comp[comp.index("test_auc:") + 1]) * 100
                    test_f1 = float(comp[comp.index("test_f1:") + 1]) * 100

                    if test_f1 > stats["test_f1"]:
                        stats["train_acc"] = train_acc
                        stats["train_auc"] = train_auc
                        stats["train_f1"] = train_f1

                        stats["test_acc"] = test_acc
                        stats["test_auc"] = test_auc
                        stats["test_f1"] = test_f1

        gl_train_acc.append(stats["train_acc"])
        gl_train_auc.append(stats["train_auc"])
        gl_train_f1.append(stats["train_f1"])
        gl_test_acc.append(stats["test_acc"])
        gl_test_auc.append(stats["test_auc"])
        gl_test_f1.append(stats["test_f1"])

    gl_train_acc = np.array(gl_train_acc)
    gl_train_auc = np.array(gl_train_auc)
    gl_train_f1 = np.array(gl_train_f1)
    gl_test_acc = np.array(gl_test_acc)
    gl_test_auc = np.array(gl_test_auc)
    gl_test_f1 = np.array(gl_test_f1)

    print(gl_test_acc)
    print(f"Train acc mean: {np.mean(gl_train_acc)}, std: {np.std(gl_train_acc)} ")
    print(f"Train auc mean: {np.mean(gl_train_auc)}, std: {np.std(gl_train_auc)}")
    print(f"Train f1 mean: {np.mean(gl_train_f1)}, std: {np.std(gl_train_f1)}")

    print(f"Test acc mean: {np.mean(gl_test_acc)}, std: {np.std(gl_test_acc)} ")
    print(f"Test auc mean: {np.mean(gl_test_auc)}, std: {np.std(gl_test_auc)}")
    print(f"Test f1 mean: {np.mean(gl_test_f1)}, std: {np.std(gl_test_f1)}")
