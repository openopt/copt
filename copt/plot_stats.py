import matplotlib.pyplot as plt
import json
import pandas as pd

import os,sys

def main():
    filename = sys.argv[1]

    with open(filename, 'r') as f:
        stats = [json.loads(line) for line in f.readlines()]

    fig, axs = plt.subplots(nrows=1,ncols=3,sharex=True,figsize=(21,7))

    axs[0].loglog([s['objective'] for s in stats])
    axs[1].loglog([s['feasibility1'] for s in stats])
    axs[2].loglog([s['feasibility2'] for s in stats])

    for ax,name in zip(axs,["objective", "feasibility1", "feasibility2"]):
        ax.set_title(name)

    for ax in axs:
        ax.grid(True)

    plt.show()
    

if __name__ == "__main__":
    main()