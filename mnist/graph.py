import matplotlib.pyplot as plt
import numpy as np

def plotn(N):
    A = np.loadtxt("exp-n%d-mnist.csv" % N, delimiter=",")
    print(A)

    l2s = A[:,0][np.argsort(-A[:, 0])]
    test_accs = A[:,1][np.argsort(-A[:, 0])]
    train_accs = A[:,2][np.argsort(-A[:, 0])]
    diff = train_accs - test_accs

    plt.rcParams["figure.figsize"] = (5, 3)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    color2 = 'tab:orange'
    ax1.set_xlabel(r"$\bf{NN\ L2\ Norm}$")
    ax1.set_ylabel(r"$\bf{Accuracy}$", color=color)
    ax1.plot(l2s, train_accs, marker="o", markersize=2, color=color, label="Train")
    ax1.plot(l2s, test_accs, marker="o", markersize=2, color=color2, label="Test", alpha=0.75)
    ax1.plot(l2s, diff, marker="o", markersize=2, color="purple", label="Gap", alpha=0.75)
    ax1.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.title("%d-Param Network" % N)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("exp-%d.pdf" % N)
    plt.clf()
    plt.close()


plotn(1078)
plotn(4282)
plotn(37354)
