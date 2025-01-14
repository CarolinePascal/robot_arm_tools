import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

### PLOT PARAMETERS ###

cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("tab20c")
cmap_complement = colors.ListedColormap(plt.get_cmap("tab20c")(np.arange(2,40,4)))

markers = ["^","s","o","v","D","x","*","+"]
markers_complement = markers[::-1]
plotly_markers = ["triangle-up","square","circle","triangle-down","diamond","x","star","cross"]
plt_to_plotly_markers = dict(zip(markers,plotly_markers))

figsize = (10,9)

arrow = dict(arrowstyle='<-',lw=0.5,color="gray")

plt.rc('font', **{'size': 24, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

def log_formatter(x,pos):
    sci = "{:.0e}".format(x)
    sci = [int(item) for item in sci.split("e")]
    if(sci[0] == 5):
        return(r"$5\times10^{{{exponent}}}$".format(exponent=sci[1]))
    else:   
        return("")