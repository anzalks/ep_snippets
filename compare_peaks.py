__author__           = "Anzal KS"
__copyright__        = "Copyright 2019-, Anzal KS"
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

def pattern_files(c):
    f_list = []
    f_list = list(c.glob('**/*txt'))
    f_list.sort()
    return f_list

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
    for f_name in f_list:
        training_f = training_finder(f_name)
        print(f'parsed prot train = {training_f}')
        if training_f != None:
            training_indx = f_list.index(training_f)
            pre = f_list[:training_indx]
            post = f_list[training_indx+1:]
#            pprint(f'training file - {training_f} , indx = {training_indx} '
#                   f'pre file ={pre} '
#                   f'post file = {post} '
#                  )
        else:
            pre_f_none, post_f_none = None, None
    return [pre, post, pre_f_none, post_f_none ]

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
        print(f' tag on the file ={tag}')
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
    print(cell_)
    return cell_




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
#    print(f' baseline = {cell_trace_av}')
    #print(ttl_xi[0])
    #print(ttl_xf[0])
    events = []
    for i,ti in enumerate(ttl_xi): 
        events.append(np.max(cell_trace_av[ttl_xi[i]:ttl_xf[i]]))
    print(file_id)
#    print(events)
    return events
#    print(cell_trace_av[ttl_xi[0]],cell_trace_av[ttl_xf[0]])
#    events = []
#    for i in ttl_xi:
#        for j in ttl_xf:
#            events.append(cell_trace_av[i,j])
#    print(events)

def count_action_potentials(step_c_file, title,fig,axs, plt_no):
    pre_f =step_c_file[0]
#    post_f = step_c_file[1]
    cell_ = cell_trace(pre_f)
    t = cell_[0]
    vms = cell_[1]
    for i,vm in enumerate(vms):
        axs[plt_no].plot(t,vm, label=f'Trial no. {i}')
    axs[plt_no].set_ylim(-80,60)
    axs[plt_no].set_xlabel('time (s)')
    axs[plt_no].set_title(title, fontproperties=sub_titles)
 #   axs[plt_no].legend(loc='upper center', 
 #                      bbox_to_anchor=(0.5, -0.15),
 #                      fancybox=True,
 #                      title="Trial number")
def input_res(input_r, title,fig,axs, plt_no):
    pre_f = input_r[0]
    cell_ = cell_trace(pre_f)
    t= cell_[0]
    vms = cell_[1]
    for i, vm in enumerate(vms):
        axs[plt_no].plot(t,vm, label=f'trial no. {i}')
    axs[plt_no].set_ylim(-75, -55)
    axs[plt_no].set_xlabel('time (s)')
    axs[plt_no].set_title(title, fontproperties=sub_titles)

def peak_comapre(points_or_pattern_file_set,title, fig, axs, plt_no):
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = peak_event(pre_f)
    post = peak_event(post_f)
    pre_x = np.ones(len(pre))
    post_x = np.ones(len(post))*2
    axs[plt_no].set_xlim(-1,3)
    #axs[plt_no].set_ylim(-1,3)
    axs[plt_no].set_ylabel('Cell response to patterns (mV)',
                           fontproperties=sub_titles)
    axs[plt_no].scatter(pre_x,pre, color='g', label='pre')
    axs[plt_no].scatter(post_x,post,color='r', label='post')
    if title=='pattern':
        axs[plt_no].scatter(pre_x[4],pre[4], color='k')
        axs[plt_no].scatter(post_x[4],post[4], color='k', label='trained_pat')
    axs[plt_no].set(xlabel=None)
    axs[plt_no].set_xticks([])
    axs[plt_no].set_title(title, fontproperties=sub_titles)
    axs[plt_no].legend(ncol =3, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       title="frame presentation")



def pre_post_plot(points_or_pattern_file_set,title, fig, axs, plt_no):
    pre_f = points_or_pattern_file_set[0]
    post_f = points_or_pattern_file_set[1]
    pre = peak_event(pre_f)
    post = peak_event(post_f)
    indices = np.arange(0,len(pre), 1)
    if len(indices)>11:
        n_col = 8
    else:
        n_col=5
    #print(f'indices = {indices}')
    cmap = cm.get_cmap('jet', len(indices))
    cmap.set_under('gray')
    for i in  indices :
        axs[plt_no].scatter(pre[i],post[i], label =i+1)
#    axs[plt_no].scatter(pre,post, c=indices, cmap=cmap, vmin=0, 
#                        vmax=indices.max())
    axs[plt_no].set_xlim(-1,30)
    axs[plt_no].set_ylim(-1,30)
    axs[plt_no].plot([-1,30], [-1,30], linestyle='--', color='k')
    axs[plt_no].set_xlabel('Pre response (mV)', fontproperties=sub_titles)
    axs[plt_no].set_ylabel('Post reponse in (mV)', fontproperties=sub_titles)
    axs[plt_no].set_title(title, fontproperties=sub_titles)
    axs[plt_no].legend(ncol=n_col,loc='upper center', 
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       title="frame numbers")

def plot_summary(cell, outdir):
    cell_id = str(cell.stem)
    abf_list = list_files(cell)
    outdir = outdir/'scatter_po_vs_pat'
    outdir.mkdir(exist_ok=True, parents=True)
    plot_name = f'{str(outdir)}/{cell_id}.png'
    sorted_f_list = pre_post_sorted(abf_list)
    pre_f_list = sorted_f_list[0]
    post_f_list = sorted_f_list[1]
    pprint(f'pre = {pre_f_list} , post = {post_f_list}')
    paired_list = file_pair_pre_pos(pre_f_list, post_f_list)
    #pprint(f'points = {paired_list[0]} , patterns = {paired_list[1]}')
    fig, axs = plt.subplots(2,3, figsize = (20,10))
    axs=axs.flatten()
    pre_post_plot(paired_list[0], 'points', fig, axs, 0)
    pre_post_plot(paired_list[1],'pattern', fig, axs, 1)
    peak_comapre(paired_list[0],'points', fig, axs, 2)
    peak_comapre(paired_list[1],'pattern', fig, axs, 3)
    count_action_potentials(paired_list[4], 'Cell firing',fig,axs, 4)  
    input_res(paired_list[3], 'series resistance',fig,axs, 5)
    plt.suptitle(f'cell ID = {cell_id}', 
                 fontproperties=main_title)
    plt.subplots_adjust(hspace=1)
#    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')

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
def main(**kwargs):
    p = Path(kwargs['abf_path'])
    c = Path(kwargs['pattern_path'])
    outdir = p/'result_plots'
    outdir.mkdir(exist_ok=True, parents=True)
    cells = list_folder(p)
    pprint(cells)
#    cell_id = str(p/../).split('/')[-1]
    print(f'plot saving folder = {outdir}')
    for cell in cells:
        print(f'cell  = {cell.stem}')
        plot_summary(cell, outdir)
#    abf_list = list_files(cell)
#    p_files = pattern_files(c)
#    plot_name = f'{str(outdir)}/{cell_id}.png'
#    sorted_f_list = pre_post_sorted(abf_list)
#    pre_f_list = sorted_f_list[0]
#    post_f_list = sorted_f_list[1]
##    pprint(f'pre = {pre_f_list} , post = {post_f_list}')
#    paired_list = file_pair_pre_pos(pre_f_list, post_f_list)
#    pprint(f'points = {paired_list[0]} , patterns = {paired_list[1]}')
#    fig, axs = plt.subplots(1,2, figsize = (15,5))
#    pre_post_plot(paired_list[0], 'points', fig, axs, 0)
#    pre_post_plot(paired_list[1],'pattern', fig, axs, 1)
#    plt.suptitle(f'cell ID = {cell_id}', 
#                 fontproperties=main_title)
##    plt.show()
#    fig.savefig(plot_name, bbox_inches='tight')
#

if __name__  == '__main__':
    import argparse
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

    parser.parse_args(namespace=args_)
    main(**vars(args_)) 
