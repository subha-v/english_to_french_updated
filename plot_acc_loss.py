import json
import matplotlib.pyplot as plt
import argparse

def plot_data(acc_data_fn, loss_data_fn):
    iter_2_val_acc = {}
    iter_2_loss = {}

    with open(acc_data_fn, "r") as f:
        iter_2_val_acc = json.load(f)


    with open(loss_data_fn, "r") as f:
        iter_2_loss = json.load(f)

    iters = list(iter_2_val_acc.keys())
    acc_values = list(iter_2_val_acc.values())
    acc_values = [element/100 for element in acc_values]

    loss_values = list(iter_2_loss.values())

    plt.figure()
    plt.plot(iters, acc_values, "g.-", label = "Validation Accuracy")
    plt.plot(iters, loss_values, "b.-", label = "Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc_fn", type=str, required=True)
    parser.add_argument("--loss_fn", type=str, required = True)
    args = parser.parse_args()
    plot_data(args.acc_fn, args.loss_fn)


