#!/usr/bin/python3

#Utility packages
import numpy as np
from copy import deepcopy

#Acoustics packages
import measpy as ms
from measpy._tools import wrap, nth_octave_bands
from unyt import Unit
from csaps import csaps

#Mesh package
import open3d as o3d

#Plot tools
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
import plotly.graph_objects as go

cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("tab20c")
markers = ["^","s","o","v","D","x"]
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

def plot_weighting(weighting, frequencies, unit=Unit("1"), ax=None, logx=True, dby=True, plot_phase=True, unwrap_phase=True, validity_range = None, scalingFactor = 1.0, **kwargs):

    if dby and (unit != Unit("Pa")) and (unit != Unit("m/s")) and (unit != Unit("1")):
        dby = False
        print("Warning: dB cannot be plotted with unit " + str(unit.units) + ", plotting linear values instead")

    if type(ax) == type(None):
        if plot_phase:
            _, ax = plt.subplots(2)
            ax_0 = ax[0]
        else:
            _, ax = plt.subplots(1)
            ax_0 = ax
    else:
        if plot_phase:
            ax_0 = ax[0]
        else:
            ax_0 = ax

    #Interpolate weighting data
    spa = csaps(weighting.freqs, weighting.amp, smooth=0.9)
    spp = csaps(weighting.freqs, weighting.phase, smooth=0.9)
    interpolated_walues_amp = spa(frequencies)
    interpolated_walues_phase = spp(frequencies)

    if dby:
        if (unit == Unit("Pa")):
            modulus_to_plot = 20*np.log10(np.abs(interpolated_walues_amp)/ms.PREF)
            weighting_modulus_to_plot = 20*np.log10(np.abs(weighting.amp)/ms.PREF)
            modulus_label = r'20 log $|P|/P_0$ (dB SPL)'
        elif (unit == Unit("m/s")):
            modulus_to_plot = 20*np.log10(np.abs(interpolated_walues_amp)/ms.VREF)
            weighting_modulus_to_plot = 20*np.log10(np.abs(weighting.amp)/ms.VREF)
            modulus_label = r'20 log $|V|/V_0$ (dB SVL)'
        else:
            modulus_to_plot = 20*np.log10(np.abs(interpolated_walues_amp))
            weighting_modulus_to_plot = 20*np.log10(np.abs(weighting.amp))
            modulus_label = r'20 log $|$H$|$ (dB)'

        # Only keep finite values
        valid_indices = np.isfinite(modulus_to_plot)

    else:
        modulus_to_plot = np.abs(interpolated_walues_amp)
        weighting_modulus_to_plot = np.abs(weighting.amp)

        modulus_unit = "(" + str(unit.units) + ")" if unit != Unit("1") else "(-)"
        modulus_label = r'$|$H$|$' + " " + modulus_unit

        # Only keep positive values
        valid_indices = np.where(modulus_to_plot > 0)

    frequencies_to_plot = scalingFactor*frequencies[valid_indices]
    modulus_to_plot = modulus_to_plot[valid_indices]

    line_plot_kwargs = deepcopy(kwargs)
    line_plot_kwargs["label"] = None
    line_plot_kwargs["marker"] = "None"
    line_plot_kwargs["linewidth"] = 2

    ax_0.plot(frequencies_to_plot, modulus_to_plot, **line_plot_kwargs)
    if not plot_phase:
        if(scalingFactor != 1.0):
            ax_0.set_xlabel(r'$hk$')
        else:
            ax_0.set_xlabel('Frequency (Hz)')
        ax_0.yaxis.set_major_locator(MaxNLocator(10))
    else:
        ax_0.yaxis.set_major_locator(MaxNLocator(5))

    ax_0.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_0.set_ylabel(modulus_label,labelpad=15)

    if logx:
        ax_0.set_xscale('log')
        #ax_0.xaxis.set_minor_formatter(FuncFormatter(log_formatter))
    else:
        ax_0.xaxis.set_major_locator(MaxNLocator(10))
        ax_0.xaxis.set_minor_locator(AutoMinorLocator(2))

    ax_0.grid(linestyle= '-', which="major")
    ax_0.grid(linestyle = '--', which="minor")
    ax_0.tick_params(axis='both', which='both', pad=7)

    if(validity_range is not None and scalingFactor == 1.0):
        if(len(ax_0.patches) == 0):
            ax_0.axvspan(validity_range[0],validity_range[1],color="gray",alpha=0.175)
        
    if plot_phase:
        phase_to_plot = interpolated_walues_phase[valid_indices]
        weighting_phase_to_plot = weighting.phase
        if unwrap_phase:
            phase_to_plot = np.unwrap(phase_to_plot)

            #Fix eventual unwrap phase delta
            first_index = np.argmin(np.abs(frequencies - weighting.freqs[0]))
            phase_delta = 2*np.pi * np.round((np.unwrap(interpolated_walues_phase)[first_index] - weighting.phase[0]) / (2*np.pi))
            weighting_phase_to_plot = weighting.phase + phase_delta
            
        ax[1].plot(frequencies_to_plot, phase_to_plot, **line_plot_kwargs)

        if(scalingFactor != 1.0):
            ax[1].set_xlabel(r'$hk$')
        else:
            ax[1].set_xlabel('Frequency (Hz)')
        if logx:
            ax[1].set_xscale('log')
            #ax[1].xaxis.set_minor_formatter(FuncFormatter(log_formatter))
        else:
            ax[1].xaxis.set_major_locator(MaxNLocator(10))
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(2))

        ax[1].set_ylabel('Phase (rad)')
        ax[1].yaxis.set_major_locator(MaxNLocator(5))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))

        ax[1].grid(linestyle = '-', which="major")
        ax[1].grid(linestyle = '--', which="minor")

        if(validity_range is not None):
            if(len(ax[1].patches) == 0):
                ax[1].axvspan(validity_range[0],validity_range[1],color="gray",alpha=0.175)

        ax[1].tick_params(axis='both', which='both', pad=7)
    
    marker_plot_kwargs = deepcopy(kwargs)
    marker_plot_kwargs["linestyle"] = "None"
    marker_plot_kwargs["markerfacecolor"] = "None"
    marker_plot_kwargs["markersize"] = 7
    marker_plot_kwargs["markeredgewidth"] = 2

    ax_0.plot(scalingFactor*weighting.freqs, weighting_modulus_to_plot, **marker_plot_kwargs)
    if plot_phase:
        ax[1].plot(scalingFactor*weighting.freqs, weighting_phase_to_plot, **marker_plot_kwargs)

    h, l = ax_0.get_legend_handles_labels()
    total_label_length = np.sum([len(label) for label in l])
    if(total_label_length > 20):
        ncol = 2
        spacing = 1.0
    else:
        ncol = 6
        spacing = 0.5
    ax_0.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=ncol, borderaxespad=0.25, reverse=False,columnspacing=spacing,fontsize = 20)

    return ax

def plot_spatial_data(data, points, unit=Unit("1"), ax=None, dby=True, plot_phase=True, unwrap_phase=True, **kwargs):

    if dby and (unit != Unit("Pa")) and (unit != Unit("m/s")) and (unit != Unit("1")):
        dby = False
        print("Warning: dB cannot be plotted with unit " + str(unit.units) + ", plotting linear values instead")

    if type(ax) == type(None):
        if plot_phase:
            _, ax = plt.subplots(2)
            ax_0 = ax[0]
        else:
            _, ax = plt.subplots(1)
            ax_0 = ax
    else:
        if plot_phase:
            ax_0 = ax[0]
        else:
            ax_0 = ax

    if dby:
        if (unit == Unit("Pa")):
            modulus_to_plot = 20*np.log10(np.abs(data)/ms.PREF)
            modulus_label = r'20 log $|P|/P_0$ (dB SPL)'
        elif (unit == Unit("m/s")):
            modulus_to_plot = 20*np.log10(np.abs(data)/ms.VREF)
            modulus_label = r'20 log $|V|/V_0$ (dB SVL)'
        else:
            modulus_to_plot = 20*np.log10(np.abs(data))
            modulus_label = r'20 log $|$H$|$ (dB)'

        # Only keep finite values
        valid_indices = np.isfinite(modulus_to_plot)

    else:
        modulus_to_plot = np.abs(data)

        modulus_unit = "(" + str(unit.units) + ")" if unit != Unit("1") else "(-)"
        modulus_label = r'$|$H$|$' + " " + modulus_unit

        # Only keep positive values
        valid_indices = np.where(modulus_to_plot > 0)

    points_to_plot = points[valid_indices] + 1
    modulus_to_plot = modulus_to_plot[valid_indices]

    kwargs["markerfacecolor"] = "None"
    kwargs["linewidth"] = 2
    kwargs["markersize"] = 7
    kwargs["markeredgewidth"] = 2

    ax_0.plot(points_to_plot, modulus_to_plot, **kwargs)

    if not plot_phase:
        ax_0.set_xlabel("Points")
        ax_0.yaxis.set_major_locator(MaxNLocator(10))
    else:
        ax_0.yaxis.set_major_locator(MaxNLocator(5))

    ax_0.set_ylabel(modulus_label)
    ax_0.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax_0.set_xticks(points_to_plot)
    
    ax_0.grid(linestyle= '-', which="major")
    ax_0.grid(linestyle = '--', which="minor")
    ax_0.tick_params(axis='both', which='both', pad=7)
        
    if plot_phase:
        phase_to_plot = np.angle(data)[valid_indices]
        if unwrap_phase:
            phase_to_plot = np.unwrap(phase_to_plot)

        ax[1].plot(points_to_plot, phase_to_plot, **kwargs)
        
        ax[1].set_xlabel('Points')
        ax[1].set_xticks(points_to_plot)

        ax[1].set_ylabel('Phase (rad)')
        ax[1].yaxis.set_major_locator(MaxNLocator(5))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax[1].grid(linestyle = '-', which="major")
        ax[1].grid(linestyle = '--', which="minor")
        ax[1].tick_params(axis='both', which='both', pad=7)

    h, l = ax_0.get_legend_handles_labels()
    total_label_length = np.sum([len(label) for label in l])
    if(total_label_length > 20):
        ncol = 2
        spacing = 1.0
    else:
        ncol = 6
        spacing = 0.5
    ax_0.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=ncol, borderaxespad=0.25, reverse=False,columnspacing=spacing,fontsize = 20)

    return ax

def plot_polar_data(data, points, unit=Unit("1"), ax=None, dby=True, plot_phase=True, unwrap_phase=True, **kwargs):

    if dby and (unit != Unit("Pa")) and (unit != Unit("m/s")) and (unit != Unit("1")):
        dby = False
        print("Warning: dB cannot be plotted with unit " + str(unit.units) + ", plotting linear values instead")

    if type(ax) == type(None):
        if plot_phase:
            _, ax = plt.subplots(1,2,subplot_kw={'projection': 'polar'})
            ax_0 = ax[0]
        else:
            _, ax = plt.subplots(1,subplot_kw={'projection': 'polar'})
            ax_0 = ax
    else:
        if plot_phase:
            ax_0 = ax[0]
        else:
            ax_0 = ax

    if dby:
        if (unit == Unit("Pa")):
            modulus_to_plot = 20*np.log10(np.abs(data)/ms.PREF)
            modulus_label = r'20 log $|P|/P_0$ (dB SPL)'
        elif (unit == Unit("m/s")):
            modulus_to_plot = 20*np.log10(np.abs(data)/ms.VREF)
            modulus_label = r'20 log $|V|/V_0$ (dB SVL)'
        else:
            modulus_to_plot = 20*np.log10(np.abs(data))
            modulus_label = r'20 log $|$H$|$ (dB)'

        # Only keep finite values
        valid_indices = np.where(np.isfinite(modulus_to_plot))[0]

    else:
        modulus_to_plot = np.abs(data)

        modulus_unit = "(" + str(unit.units) + ")" if unit != Unit("1") else "(-)"
        modulus_label = r'$|$H$|$' + " " + modulus_unit

        # Only keep positive values
        valid_indices = np.where(modulus_to_plot > 0)[0]

    points_to_plot = np.arange(0,2*np.pi,2*np.pi/len(valid_indices))
    points_to_plot = np.append(points_to_plot,points_to_plot[0])

    points_labels = np.append(points[valid_indices] + 1, points[valid_indices][0] + 1)
    modulus_to_plot = modulus_to_plot[valid_indices]
    modulus_to_plot = np.append(modulus_to_plot,modulus_to_plot[0])

    kwargs["markerfacecolor"] = "None"
    kwargs["linewidth"] = 2
    kwargs["markersize"] = 7
    kwargs["markeredgewidth"] = 2

    ax_0.plot(points_to_plot, modulus_to_plot, **kwargs)

    ax_0.xaxis.set_ticks(points_to_plot)
    ax_0.xaxis.set_ticklabels(points_labels)
    
    if(len(ax_0.texts) == 0):
        ax_0.annotate("",xy=(0.5,0.5),xytext=(1.0,0.5),xycoords="axes fraction",arrowprops=arrow)
        ax_0.annotate('x',xy=(0.5,0.5),xytext=(0.95,0.44),xycoords="axes fraction",color="gray",fontsize=20)
        ax_0.annotate("",xy=(0.5,0.5),xytext=(0.5,1.0),xycoords="axes fraction",arrowprops=arrow)
        ax_0.annotate('y',xy=(0.5,0.5),xytext=(0.44,0.95),xycoords="axes fraction",color="gray",fontsize=20)

    ax_0.yaxis.set_major_formatter(ScalarFormatter())
    #ax_0.yaxis.get_major_formatter().set_useOffset(False)
    ax_0.set_rmin(0)
    ax_0.yaxis.set_major_locator(MaxNLocator(4))
    #ax_0.yaxis.set_minor_locator(MaxNLocator(2))
    
    ax_0.grid(linestyle= '-', which="major")
    #ax_0.grid(linestyle = '--', which="minor")
    ax_0.tick_params(axis='both', which='major', pad=10, labelsize=20)

    ax_0.set_title(modulus_label, y=-0.325)

    if plot_phase:
        phase_to_plot = np.angle(data)[valid_indices]
        if unwrap_phase:
            phase_to_plot = np.unwrap(phase_to_plot)
        else:
            phase_to_plot = wrap(phase_to_plot)
            if(np.abs(np.min(phase_to_plot) + np.pi) < np.abs(np.min(phase_to_plot)) and np.max(phase_to_plot) <= 0):
                phase_to_plot += 2*np.pi
        phase_to_plot = np.append(phase_to_plot,phase_to_plot[0])

        ax[1].plot(points_to_plot, phase_to_plot, **kwargs)

        ax[1].xaxis.set_ticks(points_to_plot)
        ax[1].xaxis.set_ticklabels(points_labels)

        if(len(ax[1].texts) == 0):
            ax[1].annotate("",xy=(0.5,0.5),xytext=(1.0,0.5),xycoords="axes fraction",arrowprops=arrow)
            ax[1].annotate('x',xy=(0.5,0.5),xytext=(0.95,0.44),xycoords="axes fraction",color="gray",fontsize=20)
            ax[1].annotate("",xy=(0.5,0.5),xytext=(0.5,1.0),xycoords="axes fraction",arrowprops=arrow)
            ax[1].annotate('y',xy=(0.5,0.5),xytext=(0.44,0.95),xycoords="axes fraction",color="gray",fontsize=20)

        ax[1].yaxis.set_major_formatter(ScalarFormatter())
        #ax[1].yaxis.get_major_formatter().set_useOffset(False)
        if not unwrap_phase:
            if(np.max(phase_to_plot) > np.pi):
                ax[1].set_rmin(0)
                ax[1].set_rmax(2*np.pi)
            else:
                ax[1].set_rmin(-np.pi)
                ax[1].set_rmax(np.pi)

        ax[1].yaxis.set_major_locator(MaxNLocator(4))
        #ax[1].yaxis.set_minor_locator(MaxNLocator(2))
        
        ax[1].grid(linestyle= '-', which="major")
        #ax[1].grid(linestyle = '--', which="minor")
        ax[1].set_title("Phase (rad)", y=-0.325)
        ax[1].tick_params(axis='both', which='major', pad=10, labelsize=20)

    h, l = ax_0.get_legend_handles_labels()
    total_label_length = np.sum([len(label) for label in l])
    if(total_label_length > 20):
        ncol = 2
    else:
        ncol = 6

    spacing = 0.5
    if(plot_phase):
        ax_0.legend(bbox_to_anchor=(-0.2, 1.0, 2.7, .1), loc='lower left', ncol=ncol, borderaxespad=2, reverse=False, mode="expand", columnspacing=spacing)
    else:
        ax_0.legend(bbox_to_anchor=(0.5,1.0), loc='lower center', ncol=ncol, borderaxespad=2, reverse=False, columnspacing=spacing)

    return ax

def plot_3d_data(data, points, ax=None, interactive = False, **kwargs):

    if type(ax) == type(None):
        if(interactive):
            ax = go.Figure()
        else:
            _, ax = plt.subplots(1,subplot_kw=dict(projection='3d'))

    if(interactive):
        ax.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode="markers",marker_symbol="circle",marker={"size": kwargs.pop("s",4),"color": data,"colorscale": "Jet"},hovertext=data, customdata=data, hovertemplate="(%{x:.3f},%{y:.3f},%{z:.3f})<br>value: %{customdata}<extra></extra>"))

        ax.update_layout(scene = {"xaxis_title":"x (m)","yaxis_title":"y (m)","zaxis_title":"z (m)"})
        if("label" in kwargs):
            ax.update_layout(coloraxis_colorbar={"title":kwargs["label"]})

    else:
        sc = ax.scatter(*points.T, c = data, s=40, cmap = "jet")
        ax.set_xlabel("x (m)",labelpad=15)
        ax.set_ylabel("y (m)",labelpad=15)
        ax.set_zlabel("z (m)",labelpad=15)
        cbar = plt.colorbar(sc, fraction=0.04, pad = 0.075) 
        
        if("label" in kwargs):
            cbar.set_label(kwargs["label"],labelpad=10)
        
        xlim = ax.get_xlim()
        deltaX = xlim[1] - xlim[0]
        meanX = np.mean(xlim)
        ylim = ax.get_ylim()
        deltaY = ylim[1] - ylim[0]
        meanY = np.mean(ylim)
        zlim = ax.get_zlim()
        deltaZ = zlim[1] - zlim[0]
        meanZ = np.mean(zlim)

        delta = np.max([deltaX,deltaY,deltaZ])

        ax.set_xlim(meanX - 0.5*delta, meanX + 0.5*delta)
        ax.set_ylim(meanY - 0.5*delta, meanY + 0.5*delta)
        ax.set_zlim(meanZ - 0.5*delta, meanZ + 0.5*delta)

        ax.grid(linestyle= '-', which="major")
        ax.grid(linestyle = '--', which="minor")
        ax.tick_params(axis='z', which='both', pad=10)

        ax.set_box_aspect((1,1,1))

    return(ax)

#Data processing tools and parameters

#TFE computation frequencies
fmin = 60
fmax = 5000

#TFE smoothing 
fminOct = 50
fmaxOct = 5000
octBand = 12
octBandFrequencies = np.round(nth_octave_bands(octBand,fminOct,fmaxOct)[0])

#Validity range frequencies
fminValidity = 50
fmaxValidity = 1000

def save_fig(fig, name, interactive = False):
    if(interactive):
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.write_html(name)
    else:
        fig.tight_layout()
        fig.savefig(name, dpi = 300, bbox_inches = 'tight')

def set_title(ax, title):
    try:
        ax[0].set_title(title, pad=60)
    except TypeError:
        ax.set_title(title, pad=60)

def plot_absolute_error(wExp, wTh, frequencies, ax, validity_range = None, scalingFactor = 1.0, **kwargs):
    wAbs = ms.Weighting(freqs = wTh.freqs, amp = np.abs(wExp.amp/wTh.amp), phase = np.unwrap(wrap(wExp.phase - wTh.phase)))

    plot_weighting(wAbs, frequencies, unit=Unit("1"), ax=ax, validity_range=validity_range, scalingFactor=scalingFactor, **kwargs)

    return(wAbs)          

def plot_relative_error(wExp, wTh, frequencies, ax, validity_range = None, scalingFactor = 1.0, **kwargs):
    valuesRel = (wExp.amp*np.exp(1j*wExp.phase) - wTh.amp*np.exp(1j*wTh.phase)) / (wTh.amp*np.exp(1j*wTh.phase))
    wRel = ms.Weighting(freqs = wTh.freqs, amp = np.abs(valuesRel))

    plot_weighting(wRel, frequencies, unit=Unit("1"), ax=ax, plot_phase=False, dby=False, validity_range=validity_range, scalingFactor=scalingFactor, **kwargs)

    return(wRel)

def plot_relative_separated_error(wExp, wTh, frequencies, ax, validity_range = None, scalingFactor = 1.0, **kwargs):
    ampValuesRelSep = np.abs(wExp.amp - wTh.amp)/np.abs(wTh.amp)
    phaseValuesRelSep = np.unwrap(wrap(wExp.phase - wTh.phase))/(2*np.pi)
    wRelSep = ms.Weighting(freqs = wTh.freqs, amp = ampValuesRelSep, phase = phaseValuesRelSep)

    plot_weighting(wRelSep, frequencies, unit=Unit("1"), ax=ax, dby=False, validity_range=validity_range, scalingFactor=scalingFactor, **kwargs)

    ax[1].set_ylabel(r'Phase (rad/2$\pi$)')

    return(wRelSep)

def compute_l2_errors(wExp, wTh, frequencyRange = None):

    if(frequencyRange is None):
        frequencyRange = [np.min(wTh.freqs),np.max(wTh.freqs)]

    indexValidity = np.where((wTh.freqs > frequencyRange[0]) & (wTh.freqs < frequencyRange[1]))
    errorAbs = np.sqrt(np.sum(np.abs(wExp.amp*np.exp(1j*wExp.phase) - wTh.amp*np.exp(1j*wTh.phase))[indexValidity]**2))
    errorRel = errorAbs/np.sqrt(np.sum(np.abs(wTh.amp*np.exp(1j*wTh.phase))[indexValidity]**2))

    return(errorAbs,errorRel)

def plot_absolute_error_spatial(dataExp, dataTh, points, ax, **kwargs):
    dataAbs = dataExp/dataTh
    plot_spatial_data(dataAbs, points, unit=Unit("1"), ax=ax, **kwargs)
    return(dataAbs)

def plot_relative_error_spatial(dataExp, dataTh, points, ax, **kwargs):
    dataRel = (dataExp - dataTh)/dataTh
    plot_spatial_data(dataRel, points, unit=Unit("1"), ax=ax, plot_phase=False, dby=False, **kwargs)
    return(dataRel)

def plot_relative_separated_error_spatial(dataExp, dataTh, points, ax, **kwargs):
    ampValuesRelSep = np.abs(np.abs(dataExp) - np.abs(dataTh))/np.abs(dataTh)
    phaseValuesRelSep = np.unwrap(wrap(np.angle(dataExp) - np.angle(dataTh)))/(2*np.pi)
    dataRelSep = ampValuesRelSep*np.exp(1j*phaseValuesRelSep)
    plot_spatial_data(dataRelSep, points, unit=Unit("1"), ax=ax, dby=False, **kwargs)
    return(dataRelSep)

def compute_l2_errors_spatial(dataExp, dataTh):
    errorAbs = np.sqrt(np.sum(np.abs(dataExp - dataTh)**2))
    errorRel = errorAbs/np.sqrt(np.sum(np.abs(dataTh)**2))

    return(errorAbs,errorRel)