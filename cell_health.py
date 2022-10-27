__author__           = "Anzal KS"
__copyright__        = "Copyright 2022-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"
from pathlib import Path
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from PIL import Image
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontProperties
from matplotlib import cm
import seaborn as sns
import trace_pattern_plot as tpp
import matplotlib.image as mpimg
from PIL import Image
import multiprocessing
import time
import argparse


class Args: pass 
args_ = Args()

def cell_trace(file_name):
    f = str(file_name)
    reader = nio.AxonIO(f)
    channels =reader.header['signal_channels']
    chan_count = len(channels)
    file_id = file_name.stem
    block = reader.read_block(signal_group_mode='split-all')
    segments = block.segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate.magnitude
    ti = sample_trace.t_start
    tf = sample_trace.t_stop
    cell_trace_all = []
    for s, segment in enumerate(segments):
        cell = channel_name_to_index(reader,'IN0')
        analogsignals = segment.analogsignals[cell]
        unit = str(analogsignals.units).split()[1]
        trace = np.array(analogsignals)
        cell_trace_all.append(trace) 
        t = np.linspace(0,float(tf-ti),len(trace))
    cell_  = (t,cell_trace_all)
#    print(cell_)
    return [cell_, sampling_rate]

def baseline_measure(file_name):
    #1st 300 m, last 300ms
    cell_trace_all = cell_trace(file_name)
    sampling_rate = cell_trace_all[1]
    trace =cell_trace_all[0]
    baseline_shift = []
    for i,t in enumerate(trace):
        bl_i = np.mean(trace[0:int(0.3*sampling_rate)])
        bl_f = np.mean(trace[-int(0.3*sampling_rate):-1])
        bl_s = bl_f-bl_i
        baseline_shift.append(bl_s)
    baseline_shift_av = np.mean(baseline_shifti,axis=0)
    print(f=' baseline shift = {baseline_shift},'
          f'average Bl_shift = {baseline_shift_av}' )
    return [baseline_shift,baseline_shift_av]

def series_res_measure_optical(file_name, title):
    cell_trace_all = cell_trace(file_name)
    sampling_rate = cell_trace_all[1]
    trace = cell_trace_all[0]
    injected_current = -20 #pA
    series_r = []
    for i,t in enumerate(trace):
        if title=='points':
            bl = np.mean(trace[int(11.7*sampling_rate):int(11.9*sampling_rate)])
            dip = np.mean(trace[int(12.3*sampling_rate):int(12.425*sampling_rate)])
            del_v = dip-bl
            r = (del_v/injected_current)*1000
            series_r.append(r)
        elif title=='pattern':
            bl = np.mean(trace[int(11.4*sampling_rate):int(11.6*sampling_rate)])
            dip = np.mean(trace[int(11.8*sampling_rate):int(12*sampling_rate)])
            del_v = dip-bl
            r = (del_v/injected_current)*1000
            series_r.append(r)
        else:
            print(f'for file {file_name} something if off with timing of'
                  f'-20pulse')
    series_r_av = np.mean(series_r,axis=0)
    return [series_r,series_r_av]

def baseline_comapre_optical(file_name):
    #1st 300 m, last 300ms
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = baseline_measure(pre_f)[1]
    post = baseline_measure(post_f)[1]
    baseline_shift_pre_post = post -pre
    print (f' shift in recording after and before = {baseline_shift_pre_post}')
    return baseline_shift_pre_post 

def series_r_compare_optical(file_name,title):
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = series_res_measure_optical(pre_f)[1]
    post = series_res_measure_optical(post_f)[1]
    series_r_shift_pre_post = post -pre
    print (f' shift in recording after and before = {series_r_shift_pre_post}')
    return series_r_shift_pre_post

def peak_dist_plot(points_or_pattern_file_set,title, fig, axs, plt_no):
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = peak_event(pre_f)[1]
    post= peak_event(post_f)[1]
    b = "#377eb8"
    o = "#ff7f00"
    pk ="#f781bf"
    max_peak = np.max(post)
    min_peak = np.min(post)
    b1=axs[plt_no].boxplot(pre, patch_artist=True,
                           boxprops=dict(facecolor=b, color=b))
    b2=axs[plt_no].boxplot(post, patch_artist=True,
                           boxprops=dict(facecolor=o, color=o))
    axs[plt_no].set_title(f'Distribution of response over trials for {title}',
                          fontproperties=sub_titles)
    axs[plt_no].set_ylabel('Cell response to patterns (mV)',
                           fontproperties=sub_titles)
    axs[plt_no].legend([b1["boxes"][0], b2["boxes"][0]], 
                       ['pre', 'post'],
                       ncol =2, loc='upper center',
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       title='Frame presentation')
    if 'pattern'in title:
        vl =axs[plt_no].vlines([4.5, 5.5], min_peak, max_peak,
                               linestyles='dashed', colors='red')
        axs[plt_no].legend([b1["boxes"][0], b2["boxes"][0], vl], 
                           ['pre', 'post','trained pattern'],
                           ncol =2, loc='upper center',
                           bbox_to_anchor=(0.5, -0.2),
                           fancybox=True,
                           title='Frame presentation')
