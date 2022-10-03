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

channel_names = ['IN0', 'FrameTTL', 'Photodiode', 'Field']

#sns.set_style("white")

y_labels = FontProperties()
y_labels.set_family('sans-serif')
y_labels.set_size('large')
#font.set_style('bold')

sub_titles = FontProperties()
sub_titles.set_family('sans-serif')
sub_titles.set_size('x-large')

main_title = FontProperties()
main_title.set_family('sans-serif')
main_title.set_weight('bold')
main_title.set_size('xx-large')

def list_folder(p):
    f_list = []
    f_list = list(p.glob('*_cell_*'))
    f_list.sort()
    return f_list

def list_files(p):
    f_list = []
    f_list=list(p.glob('**/*abf'))
    f_list.sort()
    return f_list

def image_files(i):
    f_list = []
    f_list = list(i.glob('**/*bmp'))
    f_list.sort()
    return f_list

def find_ttl_start(trace, N):
    data = np.array(trace)
    data -= data.min()
    data /= data.max()
    pulses = []
    for i, x in enumerate(data[::N]):
        if (i + 1) * N >= len(data):
            break
        y = data[(i+1)*N]
        if x < 0.2 and y > 0.75:
            pulses.append(i*N)
    return pulses

def channel_name_to_index(reader, channel_name):
    for signal_channel in reader.header['signal_channels']:
        if channel_name == signal_channel[0]:
            return int(signal_channel[1])

def training_finder(f_name):
    f = str(f_name)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split('\\')[-1]
    protocol_name = protocol_name.split('.')[-2]
#    print(f'protocol name = {protocol_name}')
    if 'training' in protocol_name:
        f_name= f_name
    elif 'Training' in protocol_name:
        f_name = f_name
#        print(f'training {f_name}')
    else:
#        print('not training')
        f_name = None
#    print(f'out_ training prot = {f_name}')
    return f_name 

def pre_post_sorted(f_list):
    found_train=False
    for f_name in f_list:
        training_f = training_finder(f_name)
#        print(f'parsed prot train = {training_f}')
        if ((training_f != None) and (found_train==False)):
            training_indx = f_list.index(training_f)
            # training indx for post will have first element as the training protocol trace
            pre = f_list[:training_indx]
            post = f_list[training_indx:]
#            pprint(f'training file - {training_f} , indx = {training_indx} '
#                f'pre file ={pre} '
#                f'post file = {post} '
#                )
            found_train = True
        elif ((training_f != None) and (found_train==True)):
            no_c_train = f_name
        else:
            pre_f_none, post_f_none, no_c_train = None, None, None
    return [pre, post, no_c_train, pre_f_none, post_f_none ]

def protocol_tag(file_name):
    f = str(file_name)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split('\\')[-1]
    protocol_name = protocol_name.split('.')[-2]
    if '42_points' in protocol_name:
        print('point_protocol')
        title = 'Points'
    elif 'Baseline_5_T_1_1_3_3' in protocol_name:
        print('pattern protocol')
        title = 'Patterns'
    elif 'Training' in protocol_name:
        print('training')
        title = 'Training pattern'
    elif 'training' in protocol_name:
        print('training')
        title = 'Training pattern'
    elif 'RMP' in protocol_name:
        print('rmp')
        title='rmp'
    elif 'Input_res' in protocol_name:
        print ('InputR')
        title ='InputR'
    elif 'threshold' in protocol_name:
        print('step_current')
        title = 'step_current'
    else:
        print('non optical protocol')
        title = None
    return title

def file_pair_pre_pos(pre_list,post_list):
    point = []
    pattern = [] 
    rmp = []
    InputR = []
    step_current = []
    for pre in pre_list:
        tag = protocol_tag(pre)
#        print(f' tag on the file ={tag}')
        if tag=='Points':
            point.append(pre)
        elif tag=='Patterns':
            pattern.append(pre)
        elif tag =='rmp':
            rmp.append(pre)
        elif tag=='InputR':
            InputR.append(pre)
        elif tag =='step_current':
            step_current.append(pre)
        else:
            tag = None
            continue
    for post in post_list:
        tag = protocol_tag(post)
        if tag=='Points':
            point.append(post)
        elif tag=='Patterns':
            pattern.append(post)
        elif tag=='rmp':
            rmp.append(post)
        elif tag=='InputR':
            InputR.append(post)
        elif tag=='step_current':
            step_current.append(post)
        else:
            tag = None
            continue
    #pprint(f'point files = {point} '
    #       f'pattern files = {pattern}'
    #      )
    return [point, pattern,rmp, InputR, step_current]

def channel_name_to_index(reader, channel_name):
    for signal_channel in reader.header['signal_channels']:
        if channel_name == signal_channel[0]:
            return int(signal_channel[1])

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

def peak_event(file_name):
    f = str(file_name)
    reader = nio.AxonIO(f)
    channels = reader.header['signal_channels']
    chan_count = len(channels)
    file_id = file_name.stem
    block  = reader.read_block(signal_group_mode='split-all')
    segments = block.segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate.magnitude
    ti = sample_trace.t_start
    tf = sample_trace.t_stop
    cell_trace_all = []
    TTL_sig_all = []
    for s, segment in enumerate(segments):
        cell = channel_name_to_index(reader,'IN0')
        analogsignals = segment.analogsignals[cell]
        unit = str(analogsignals.units).split()[1]
        trace = np.array(analogsignals)
        cell_trace_all.append(trace) 
        t = np.linspace(0,float(tf-ti),len(trace))
#    print (f'IN0 = {cell_trace_all}')
    for s, segment in enumerate(segments):
        cell = channel_name_to_index(reader,'FrameTTL')
        analogsignals = segment.analogsignals[cell]
        unit = str(analogsignals.units).split()[1]
        trace = np.array(analogsignals)
        TTL_sig_all.append(trace) 
        t = np.linspace(0,float(tf-ti),len(trace))
#    print (f' TTL = {TTL_sig_all}')
    ttl_av = np.average(TTL_sig_all,axis=0 )
    ttl_xi= find_ttl_start(trace, 3)
    ttl_xf = (ttl_xi+0.2*sampling_rate).astype(int)
    #print(len(ttl_xf- ttl_xi))
    cell_trace  = np.average(cell_trace_all, axis =0)
    cell_trace_base_line = np.mean(cell_trace[0:2000] )
    cell_trace_av = cell_trace - cell_trace_base_line
    cell_trace_b_sub = cell_trace_all-cell_trace_base_line
#    print(f' baseline = {cell_trace_av}')
    #print(ttl_xi[0])
    #print(ttl_xf[0])
    event_av = []
    events = []
    for i,ti in enumerate(ttl_xi): 
        event_av.append(np.max(cell_trace_av[ttl_xi[i]:ttl_xf[i]]))
        pattern = []
        for n, ni in enumerate(cell_trace_b_sub):
            pattern.append(np.max(ni[ttl_xi[i]:ttl_xf[i]]))
        events.append(pattern)
    return [event_av, events]

def summate_points(points_file):
    point_events = peak_event(points_file)[1]
    point_events = np.mean(point_events,axis=1)
    p1= np.sum(point_events[0:5])
    p2= np.sum(point_events[5:10])
    p3= np.sum(point_events[15:20])
    p4= np.sum(point_events[20:25])
    p5= np.sum(point_events[25:30])
    p6= np.sum(np.hstack((point_events[30:34],point_events[25])))
    p7= np.sum(np.hstack((point_events[34:38],point_events[27])))
    p8= np.sum(point_events[10:15])
    p9= np.sum(np.hstack((point_events[38:40],point_events[28:31])))
    p10= np.sum( np.hstack((point_events[40:],point_events[26:29])))
    sum_of_points = (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)
    #np.array(summate_points)
    print(f'length of summated points = {len(sum_of_points)}')
    return sum_of_points

def series_res_measure(cell_trace,injected_current,sampling_rate):
    trace = np.transpose(np.mean(cell_trace,axis=0))[0]
    vl= np.mean(trace[int(0.45*sampling_rate):int(0.55*sampling_rate)])
    vb= np.mean(trace[int(0*sampling_rate):int(0.15*sampling_rate)])
    v = vb-vl
    series_r = (v/injected_current)*1000
    print(f'vl = {vl}, vb = {vb}, series resistance indvidual value = {series_r}')
    return series_r

def image_plot(img_path, title, fig, axs, plt_no):
    fps = str(img_path).split('_')
    favr = np.array(Image.open(str(img_path)).convert('L'))
    favr = np.round(np.mean(favr),1) 
    fps = [s for s in fps if "us" in s]
    fps = fps[0].split('.')[0]
    print(f' frame rate of camera = {fps}, mean px val = {favr}')
    img = mpimg.imread(str(img_path))
    axs[plt_no].imshow(img,cmap='gray', vmin = 0, 
                       vmax =255,interpolation='none')
    axs[plt_no].set_xticks([])
    axs[plt_no].set_yticks([])
    axs[plt_no].set_xlabel(f'frame rate = {fps}, avrg px val ={favr}',
                           fontproperties=sub_titles)
    axs[plt_no].set_title(title, fontproperties=sub_titles)


def count_action_potentials(step_c_file, title,fig,axs, plt_no):
    pre_f =step_c_file[0]
#    post_f = step_c_file[1]
    cell_ = cell_trace(pre_f)[0]
    t = cell_[0]
    vms = cell_[1]
    for i,vm in enumerate(vms):
        axs[plt_no].plot(t,vm, label=f'Trial no. {i}')
    axs[plt_no].set_ylim(-80,60)
    axs[plt_no].set_ylabel('cell response (mV)', fontproperties=sub_titles)
    axs[plt_no].set_xlabel('time (s)', fontproperties=sub_titles)
    axs[plt_no].set_title(title, fontproperties=sub_titles)
 #   axs[plt_no].legend(loc='upper center', 
 #                      bbox_to_anchor=(0.5, -0.15),
 #                      fancybox=True,
 #                      title="Trial number")
def input_res(input_r_file, title,fig,axs, plt_no):
    pre_f = input_r_file[0]
    post_f = input_r_file[1]
    trace_data_pre = cell_trace(pre_f)
    trace_data_post = cell_trace(post_f)
    cell_pre=trace_data_pre[0]
    cell_post=trace_data_post[0]
    sampling_rate = trace_data_pre[1]
    injected_current= -20 #pA
    t= cell_pre[0]
    vms_pre = cell_pre[1]
    series_r_pre = series_res_measure(vms_pre,injected_current,sampling_rate)
    series_r_post = series_r_pre
    vms_post = cell_post[1]
    series_r_post = series_res_measure(vms_post,injected_current,sampling_rate)
    series_r_post= series_r_post
    series_r_f = np.round((series_r_post - series_r_pre),1)
    print(f'series_r del = {series_r_f}')
    for i, vm in enumerate(vms_pre):
        axs[plt_no].plot(t, vm, color="#377eb8", label=f'trial no. {i}')
    for i, vm in enumerate(vms_post):
        axs[plt_no].plot(t, vm, color= "#ff7f00", label=f'trial no. {i}')
    axs[plt_no].set_ylim(-80, -55)
    axs[plt_no].set_ylabel('cell response (mV)', fontproperties=sub_titles)
    axs[plt_no].set_xlabel('time (s)', fontproperties=sub_titles)
    axs[plt_no].set_title(f'{title}: {series_r_f} MOhm', fontproperties=sub_titles)

def training_trace_plot(training_f, title,fig,axs, plt_no):
    pre_f = training_f
    cell_ = cell_trace(pre_f)[0]
    t= cell_[0]
    vms = cell_[1]
    for i, vm in enumerate(vms):
        axs[plt_no].plot(t,vm, label=f'trial no. {i}')
    axs[plt_no].set_ylim(-80, 50)
    axs[plt_no].set_ylabel('cell response (mV)', fontproperties=sub_titles)
    axs[plt_no].set_xlabel('time (s)', fontproperties=sub_titles)
    axs[plt_no].set_title(title, fontproperties=sub_titles)

def raw_peak_dist(points_or_pattern_file_set,title,fig,axs,plt_no):
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = np.transpose(peak_event(pre_f)[1])
    post = np.transpose(peak_event(post_f)[1])
    x = np.arange(1,(len(pre[0])+1),1)
    print(f'____length of x = {len(x)} length of pre = {len(pre[0])}______')
    #axs[plt_no].set_xlim(-1,3)
    #axs[plt_no].set_ylim(-1,20)
    axs[plt_no].set_ylabel('Cell response to patterns (mV)',
                           fontproperties=sub_titles)
    for i in range(len(pre)):
        axs[plt_no].scatter(x,pre[i], color='#377eb8', label='pre')
        if title=='pattern':
            axs[plt_no].scatter(x[4],pre[i][4],color='k')
    for i in range(len(post)):
        axs[plt_no].scatter(x,post[i],color='#ff7f00', label='post')
        if title=='pattern':
            axs[plt_no].scatter(x[4],post[i][4],color='r', label='trained_pat')
    axs[plt_no].set_xlabel('Frame number',
                          fontproperties=sub_titles)
   # axs[plt_no].set_xticks([])
    axs[plt_no].set_title(f'Raw response for {title}', 
                          fontproperties=sub_titles)
    axs[plt_no].legend(ncol =3, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       title="Frame presentation")


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
#    axs[plt_no].set_title(title, fontproperties=sub_titles)

def pre_vs_post_avresp(points_or_pattern_file_set,title, fig, axs, plt_no):
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = peak_event(pre_f)[0]
    post = peak_event(post_f)[0]
    pre_x = np.ones(len(pre))
    post_x = np.ones(len(post))*2
    axs[plt_no].set_xlim(-1,3)
    #axs[plt_no].set_ylim(-1,3)
    axs[plt_no].set_ylabel('Cell response to patterns (mV)',
                           fontproperties=sub_titles)
    axs[plt_no].scatter(pre_x,pre, color='#377eb8', label='pre')
    axs[plt_no].scatter(post_x,post,color='#ff7f00', label='post')
    if title=='pattern':
        axs[plt_no].scatter(pre_x[4],pre[4], color='k')
        axs[plt_no].scatter(post_x[4],post[4], color='k', label='trained_pat')
    axs[plt_no].set(xlabel=None)
    axs[plt_no].set_xticks([])
    axs[plt_no].set_title(f'Average response for {title}', fontproperties=sub_titles)
    axs[plt_no].legend(ncol =3, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       title="Frame presentation")

def plasticity_test(points_or_pattern_file_set,title, fig, axs, plt_no):
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = peak_event(pre_f)[0]
    post = peak_event(post_f)[0]
    x_y_lim =np.max(np.maximum(pre,post))+2 
    indices = np.arange(0,len(pre), 1)
    if len(indices)>11:
        n_col = 8
    else:
        n_col=5
    #print(f'indices = {indices}')
#    cmap = cm.get_cmap('jet', len(indices))
#    cmap.set_under('gray')
    for i in  indices :
        axs[plt_no].scatter(pre[i],post[i], label =i+1)
#    axs[plt_no].scatter(pre,post, c=indices, cmap=cmap, vmin=0, 
#                        vmax=indices.max())
    axs[plt_no].set_xlim(-1,x_y_lim)
    axs[plt_no].set_ylim(-1,x_y_lim)
    axs[plt_no].plot([-1,x_y_lim], [-1,x_y_lim], linestyle='--', color='k')
    axs[plt_no].set_xlabel('Pre response (mV)', fontproperties=sub_titles)
    axs[plt_no].set_ylabel('Post reponse in (mV)', fontproperties=sub_titles)
    axs[plt_no].set_title(title, fontproperties=sub_titles)
    axs[plt_no].legend(ncol=n_col,loc='upper center', 
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       title="frame numbers")

def summation_plot(points_file,pattern_file,title,fig,axs,plt_no):
    sum_of_points = summate_points(points_file)
    #average response for patterns
    pattern_response = peak_event(pattern_file)[0]
    print(f'length of patterns = {len(pattern_response)}')
    x_y_lim =np.max(np.maximum(sum_of_points,pattern_response))+2 
    if 'post' in title:
        axs[plt_no].scatter(sum_of_points,pattern_response, 
                            color='#ff7f00', label ='Cell responses')
        axs[plt_no].scatter(sum_of_points[4],pattern_response[4], 
                            color='black', label= 'Response for trained pattern')
    else:
        axs[plt_no].scatter(sum_of_points,pattern_response, 
                            label='cell Responses')
        axs[plt_no].scatter(sum_of_points[4],pattern_response[4], 
                            color = 'black', label ='Response for trained pattern')
    axs[plt_no].set_xlim(-1,x_y_lim)
    axs[plt_no].set_ylim(-1,x_y_lim)
    axs[plt_no].plot([-1,x_y_lim], [-1,x_y_lim], linestyle='--', color='k')
    axs[plt_no].set_xlabel('Summated response for points(mV)', fontproperties=sub_titles)
    axs[plt_no].set_ylabel('Response for patterns (mV)', fontproperties=sub_titles)
    axs[plt_no].set_title(title, fontproperties=sub_titles)
    axs[plt_no].legend(ncol =3, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       title="Patterns and summated points")

def plot_summary(cell, images, outdir):
    cell_id = str(cell.stem)
    abf_list = list_files(cell)
    outdir = outdir/'summary_plots'
    outdir.mkdir(exist_ok=True, parents=True)
    plot_name = f'{str(outdir)}/{cell_id}.png'
    sorted_f_list = pre_post_sorted(abf_list)
#    print(f'pre post sorted list function result ={sorted_f_list}')
    pre_f_list = sorted_f_list[0]
    post_f_list = sorted_f_list[1][1:] # post sorted list has the training protocol as first element so skipping it
    training_f = sorted_f_list[1][0]
    no_c_train = sorted_f_list[2]
#    pprint(f'pre = {pre_f_list} , post = {post_f_list}')
    paired_list = file_pair_pre_pos(pre_f_list, post_f_list)
    #pprint(f'points = {paired_list[0]} , patterns = {paired_list[1]}')
    #fig, axs = plt.subplots(2,3, figsize = (20,10))
    #print(f' training protocol = {training_f}')
    fig, axs = plt.subplots(4,4, figsize = (40,35))
    #fig, axs = plt.subplots(7,2, figsize = (20,70))
    axs=axs.flatten()
    plasticity_test(paired_list[0], 'points', fig, axs, 0)
    plasticity_test(paired_list[1],'pattern', fig, axs, 1)
    pre_vs_post_avresp(paired_list[0],'points', fig, axs, 2)
    pre_vs_post_avresp(paired_list[1],'pattern', fig, axs, 3)
    count_action_potentials(paired_list[4], 'Cell firing',fig,axs, 4)  
    input_res(paired_list[3], 'series resistance change',fig,axs, 5)
    try:
        image_plot(images[0], 'slice with fluorescence & IR', fig, axs, 6)
        image_plot(images[1], 'slice with only IR', fig, axs, 7)
    except:
        print('no image data found')
    peak_dist_plot(paired_list[0], 'points', fig, axs, 10)
    peak_dist_plot(paired_list[1],'pattern', fig, axs, 11)
    summation_plot(paired_list[0][0],paired_list[1][0],'summation pre training',
                   fig,axs,12)
    summation_plot(paired_list[0][1],paired_list[1][1],'summation post training',
                   fig,axs,13)
    raw_peak_dist(paired_list[0],'points',fig,axs,14)
    raw_peak_dist(paired_list[1],'pattern',fig,axs,15)
    if training_f !=None:
        training_trace_plot(training_f, 'Response to training protocol',
                            fig, axs, 8)
        try:
            training_trace_plot(no_c_train, 'Training response without current inj',
                                fig, axs, 9)
        except:
            print('no non training trace')
    plt.suptitle(f'cell ID = {cell_id}', 
                 fontproperties=main_title)
    plt.subplots_adjust(hspace=.8, top=0.95)
#    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')

def main(**kwargs):
    #To run the individual trace plots in one go activate below line
    tpp.main(**kwargs)
    p = Path(kwargs['abf_path'])
    c = Path(kwargs['pattern_path'])
    i = Path(kwargs['image_path'])
    outdir = p/'result_plots_multi'
    outdir.mkdir(exist_ok=True, parents=True)
    cells = list_folder(p)
    #pprint(cells)
    images = image_files(i)
#    print(images)
#    cell_id = str(p/../).split('/')[-1]
#    print(f'plot saving folder = {outdir}')

    processes = []
    for cell in cells:
        p_ = multiprocessing.Process(target=plot_summary,args=[cell,images,outdir])
        p_.start()
        processes.append(p_)
        #To print the processing ID and process
        print('current process:', p_.name, p_.pid)
    for p_ in processes:
        p_.join()


if __name__  == '__main__':
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--abf-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path of folder with folders as cell data '
                       )
    parser.add_argument('--pattern-path', '-p'
                        , required = False,default ='./', type=str
                        , help = 'path of folder with polygon'
                        'pattern files '
                       )
    parser.add_argument('--image-path', '-i'
                        , required = False, default ='./', type=str
                        , help = 'path to image data with slice image'
                       )

    parser.parse_args(namespace=args_)
#timing the run with time.time
#    ts =time.time()
    main(**vars(args_)) 
#    tf =time.time()
#    print(f'total time = {tf-ts} (s)')
