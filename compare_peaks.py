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
    if 'Training' in protocol_name:
        f_name= f_name
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
    else:
        print('non optical protocol')
        title = None
    return title

def file_pair_pre_pos(pre_list,post_list):
    point = []
    pattern = [] 
    for pre in pre_list:
        tag = protocol_tag(pre)
        print(f' tag on the file ={tag}')
        if tag=='Points':
            point.append(pre)
        elif tag=='Patterns':
            pattern.append(pre)
        else:
            tag = None
            continue
    for post in post_list:
        tag = protocol_tag(post)
        if tag=='Points':
            point.append(post)
        elif tag=='Patterns':
            pattern.append(post)
        else:
            tag = None
            continue
    pprint(f'point files = {point} '
           f'pattern files = {pattern}'
          )
    return [point, pattern]

def channel_name_to_index(reader, channel_name):
    for signal_channel in reader.header['signal_channels']:
        if channel_name == signal_channel[0]:
            return int(signal_channel[1])

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
    print (f'IN0 = {cell_trace_all}')

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
    print(len(ttl_xf- ttl_xi))
    cell_trace  = np.average(cell_trace_all, axis =0)
    cell_trace_base_line = np.mean(cell_trace[0:2000] )
    cell_trace_av = cell_trace - cell_trace_base_line
    print(f' baseline = {cell_trace_av}')
    print(ttl_xi[0])
    print(ttl_xf[0])
    events = []
    for i,ti in enumerate(ttl_xi): 
        events.append(np.max(cell_trace_av[ttl_xi[i]:ttl_xf[i]]))
    print(file_id)
    print(events)
    return events
#    print(cell_trace_av[ttl_xi[0]],cell_trace_av[ttl_xf[0]])
#    events = []
#    for i in ttl_xi:
#        for j in ttl_xf:
#            events.append(cell_trace_av[i,j])
#    print(events)





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
    print(f'indices = {indices}')
    cmap = cm.get_cmap('jet', len(indices))
    cmap.set_under('gray')
    for i in  indices :
        axs[plt_no].scatter(pre[i],post[i], label =i+1)
#    axs[plt_no].scatter(pre,post, c=indices, cmap=cmap, vmin=0, 
#                        vmax=indices.max())
    axs[plt_no].set_xlim(-1,10)
    axs[plt_no].set_ylim(-1,10)
    axs[plt_no].plot([-1,10], [-1,10], linestyle='--', color='k')
    axs[plt_no].set_xlabel('Pre response (mV)', fontproperties=sub_titles)
    axs[plt_no].set_ylabel('Post reponse in (mV)', fontproperties=sub_titles)
    axs[plt_no].set_title(title, fontproperties=sub_titles)
    axs[plt_no].legend(ncol=n_col,loc='upper center', 
                       bbox_to_anchor=(0.5, -0.15),
                       fancybox=True,
                       title="frame numbers")

def proj_file_selector(file_name,p_files):
    f = str(file_name)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split('\\')[-1]
    protocol_name = protocol_name.split('.')[-2]
#    print(f'polygon protocol name = {protocol_name}')
    if '42_points' in protocol_name:
        print('point_protocol')
        for p in p_files:
            if '42_points' in str(p):
                pat_file = p
                title = 'Points'
                zoom = 0.6
                continue
    elif 'Baseline_5_T_1_1_3_3' in protocol_name:
        print('pattern protocol')
        for p in p_files:
            if '10_patterns' in str(p):
                pat_file = p
                title = 'Patterns'
                zoom = 1.5
                continue
    elif 'Training' in protocol_name:
        print('training')
        for p in p_files:
            if 'training_pattern' in str(p):
                pat_file = p
                print(f'Training pattern file={p}')
                title = 'Training pattern'
                zoom = 1.5
                continue
    else:
        print('non optical protocol')
        pat_file = None
    return [pat_file, title, zoom]


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

def plot_pat(fig, axs, frame, x_pos,y_pos, zoom):
    frame = Image.fromarray(frame).convert('RGB')
    axs[0].set_title('projected frames', fontproperties=sub_titles)
    axs[0].set_yticks([])
    axs[0].scatter(x_pos,y_pos)
    axs[0].set_ylabel('grid points',fontproperties=y_labels)
    axs[0].set_ylim(-2,2)
#    img = OffsetImage(frame, zoom=3)
    img = OffsetImage(frame, zoom=zoom)
    ab = AnnotationBbox(img,(x_pos,y_pos),pad=0)
    axs[0].add_artist(ab)
#    x,y = axs[0].transData.transform((x_pos,y_pos))
#    axs[0].figure.figimage(frame, x,y)

def channel_name_to_index(reader, channel_name):
    for signal_channel in reader.header['signal_channels']:
        if channel_name == signal_channel[0]:
            return int(signal_channel[1])

#def channel_to_trace(file_name,channel_id):
#    f = str(file_name)
#    reader = nio.AxonIO(f)
#    channels = reader.header['signal_channels']
#    chan_count = len(channels)
#    file_id = file_name.stem
#    print(f'channel count = {chan_count}')
#    if chan_count > 1:
#        block  = reader.read_block(signal_group_mode='split-all')
#        segments = block.segments
#        sample_trace = segments[0].analogsignals[0]
#        sampling_rate = sample_trace.sampling_rate
#        ti = sample_trace.t_start
#        tf = sample_trace.t_stop
#        unit = str(sample_trace.units).split()[1]
#        print(ti, tf)
#        trace_average = []
#        for s, segment in enumerate(segments):
#            feild_id = channel_name_to_index(reader,channel_id)
#            analogsignals = segment.analogsignals[feild_id]
#            trace = np.array(analogsignals)
#            trace_average.append(trace)
#            print(f'length of trace = {len(trace)}')
#            t = np.linspace(0,float(tf-ti),len(trace))
#            lplt.plot(t,trace,alpha=0.7, label = f'trace-{s}')
#        trace_average = np.mean(trace_average, axis=0)
#        plt.plot(t, trace_average, color='r', label = 'average trace')
#        plt.title(f'{file_id} chan count ={chan_count}')
#        plt.legend(loc ="upper right")
#        plt.ylabel(unit)
#        plt.xlabel('t(s)')
#        plt.show(block=False)
#        plt.pause(0.5)
#        plt.close()
#        print(f' length of trace av = {len(trace_average)}')
#        print(trace_average)

def subplot_channels(file_name,plt_no,channel_name,fig, axs,frames, zoom):
    f = str(file_name)
    reader = nio.AxonIO(f)
    channels = reader.header['signal_channels']
    chan_count = len(channels)
    file_id = file_name.stem
#    print(f'channel count = {chan_count}')
    if chan_count > 1:
        block  = reader.read_block(signal_group_mode='split-all')
        segments = block.segments
        sample_trace = segments[0].analogsignals[0]
        sampling_rate = sample_trace.sampling_rate
        ti = sample_trace.t_start
        tf = sample_trace.t_stop
#        unit = str(sample_trace.units).split()[1]
#        print(unit)
        trace_average = []
        for s, segment in enumerate(segments):
            feild_id = channel_name_to_index(reader,channel_name)
            analogsignals = segment.analogsignals[feild_id]
            unit = str(analogsignals.units).split()[1]
            if unit =='pA':
                unit = 'V'
            else:
                pass 
            trace = np.array(analogsignals)
            trace_average.append(trace)
#            print(f'length of trace = {len(trace)}')
            t = np.linspace(0,float(tf-ti),len(trace))
            axs[plt_no].plot(t,trace,alpha=0.7,linewidth=2, 
                             label = f'trial - {s+1}')
        axs[plt_no].set_ylabel(unit, fontproperties=y_labels)
        trace_average = np.mean(trace_average, axis=0)
        if channel_name =='FrameTTL':
            x_pat = find_ttl_start(trace_average, 3)
            for pat_num,x in enumerate(x_pat):
                y_pos = 0
                fr= np.array(frames[pat_num+1]*255, dtype = np.uint8)
                frame = np.invert(fr)
#                frame = fr 
                x_pos = x/sampling_rate
#                print(f'x and y = {x_pat[1], x_pos, y_pos}')
                try:
                    plot_pat(fig, axs, frame, x_pos,y_pos, zoom)
                except:
                    continue
        if channel_name =='IN0':
            axs[plt_no].set_ylim(-75,-50)
            axs[plt_no].set_title('Cell recording',
                                 fontproperties=sub_titles)
            axs[plt_no].plot(t, trace_average, 
                             color='k',linewidth=0.25,
                             label = 'trial average')
        elif channel_name =='Field':
            axs[plt_no].set_ylim(-0.5,0.2)
        else:
            y_u = np.max(trace_average)
            y_l = np.min(trace_average)
            if y_u<=0:
                y_u = y_u-(y_u*.5)
                y_l = y_l+(y_l*.5)
            else:
                y_u = y_u+(y_u*.5)
                y_l = y_l-(y_l*.5)
            axs[plt_no].plot(t, trace_average, 
                             color='k',linewidth=0.25,
                             label = 'trial average')
            axs[plt_no].set_ylim(y_l,y_u)
            axs[plt_no].set_title(channel_name,
                                  fontproperties=sub_titles)
        axs[plt_no].legend(bbox_to_anchor=(1,0.5),loc='center left')

def plot_all_cannels(file_name,chanel_name,plt_no, fig, axs, frames, zoom):
    f = str(file_name)
    reader = nio.AxonIO(f)
    channels = reader.header['signal_channels']
    chan_count = len(channels)
    chan_count = chan_count+1
#    print(f' channel name inside the fucntion ={chanel_name}')
    subplot_channels(file_name,plt_no,chanel_name,fig,axs, frames, zoom)





def main(**kwargs):
    p = Path(kwargs['abf_path'])
    c = Path(kwargs['pattern_path'])
    outdir = p/'results_scatter_plot'
    cell_id = str(p.parent).split('/')[-1]
    print(f' folder parent = {cell_id}')
    outdir.mkdir(exist_ok=True,parents=True)
    abf_list = list_files(p)
    p_files = pattern_files(c)
    plot_name = f'{str(outdir)}/{cell_id}.png'
    sorted_f_list = pre_post_sorted(abf_list)
    pre_f_list = sorted_f_list[0]
    post_f_list = sorted_f_list[1]
#    pprint(f'pre = {pre_f_list} , post = {post_f_list}')
    paired_list = file_pair_pre_pos(pre_f_list, post_f_list)
    pprint(f'points = {paired_list[0]} , patterns = {paired_list[1]}')
    fig, axs = plt.subplots(1,2, figsize = (15,5))
    pre_post_plot(paired_list[0], 'points', fig, axs, 0)
    pre_post_plot(paired_list[1],'pattern', fig, axs, 1)
    plt.suptitle(f'cell ID = {cell_id}', 
                 fontproperties=main_title)
#    plt.show()
    fig.savefig(plot_name, bbox_inches='tight')
#    for f_name in abf_list:
#        print(f'{f_name}')
#        plot_name = str(outdir)+'/'+str(f_name.stem)+'.png'
#            p_file, title, zoom = proj_file_selector(f_name,p_files)
#            frames = pattern_dict(p_file)   
##            fig, axs = plt.subplots(5,1, figsize = (9,10),sharex=True)
#            fig, axs = plt.subplots(5,1, figsize = (15,10),sharex=True)
#            for plt_no,chanel_name in enumerate(channel_names):
#                plt_no = plt_no+1
##                print (f_name, chanel_name,plt_no)
#                plot_all_cannels(f_name,chanel_name,plt_no, 
#                                 fig, axs,frames, zoom)
#            plt.xlabel('time(s)',fontproperties=y_labels)
##            if title=='Points':
##                xlim = [6.3,7]
##            elif title=='Patterns':
##                xlim = [4,5.4]
##            elif title=='Training pattern':
##                axs[1].set_ylim(-70,60)
##                xlim = left=0
##                for i in range(5):
##                    print(f'range = ({i})' )
##                    if i>0:
##                        axs[i].get_legend().remove()
##                    else:
##                        continue
##            else:
##                continue
#
#            if title=='Points':
#                xlim = [1.25,11.25]
#            elif title=='Patterns':
#                xlim = [0.8,11]
#            elif title=='Training pattern':
#                axs[1].set_ylim(-70,60)
#                xlim = left=0
#                for i in range(5):
#                    print(f'range = ({i})' )
#                    if i>0:
#                        axs[i].get_legend().remove()
#                    else:
#                        continue
#            else:
#                continue
#
#            plt.xlim(xlim)
#            plt.suptitle(f'Response to {title}',
#                         fontproperties=main_title)
##            fig.tight_layout()
#            fig.align_ylabels()
#            plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None,
#                                wspace=None, hspace=0.5)
##            plt.show(block=False)
#            fig.savefig(plot_name, bbox_inches='tight')
##            plt.pause(2)
#            plt.close()
#        except:
#            continue







if __name__  == '__main__':
    import argparse
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--abf-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path of folder with  abf files '
                       )
    parser.add_argument('--pattern-path', '-p'
                        , required = False,default ='./', type=str
                        , help = 'path of folder with polygon'
                        'pattern files '
                       )

    parser.parse_args(namespace=args_)
    main(**vars(args_)) 