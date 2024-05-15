from math import ceil, floor
import matplotlib.pyplot as plt

# setup plotting
from IPython import get_ipython
import psutil
inTerminal = not "IPKernelApp" in get_ipython().config
inJupyterNb = any(filter(lambda x: x.endswith("jupyter-notebook"), psutil.Process().parent().cmdline()))
get_ipython().run_line_magic("matplotlib", "" if inTerminal else "notebook" if inJupyterNb else "widget")

def nextplot():
    if inTerminal:
        plt.clf()     # this clears the current plot
    else:
        plt.figure()  # this creates a new plot

def plot_kernels(kernels):
    """
    Takes the weights of a single convolutional layer, generally in the
    shape [out_channels, in_channels, height, width].
    Produces a plot consisting of out_channels * in_channels images.
    """
    kernels = kernels.cpu().detach()  # get rid of gradients for numpy compatibility
    out_channels, in_channels = kernels.size(0), kernels.size(1)
    fig, axs = plt.subplots(nrows=out_channels, ncols=in_channels)
    for x in range(in_channels):
        for y in range(out_channels):
            axs[y, x].imshow(kernels[y, x])
            axs[y, x].axis("off")
    plt.show()


def plot_conv_module_output(output, ncols=4):
    """
    Takes the output of a convolutional layer or block, generally
    in the shape [out_channels, height, width].
    Plots out_channels images in ncols columns.
    """
    output = output.cpu().detach()
    out_channels = output.size(0)
    nrows = ceil(out_channels / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(out_channels):
        row = floor(i / ncols)
        col = i % ncols
        axs[row, col].imshow(output[i])
        axs[row, col].axis("off")
    plt.show()
