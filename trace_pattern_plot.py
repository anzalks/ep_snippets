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
    return f_list

def pattern_files(c):
    f_list = []
    f_list = list(c.glob('**/*txt'))
    return f_list

def pattern_dict(p_file):
    txt = str(p_file)
    txt_read = np.loadtxt(txt, comments="#", delimiter=" ", dtype=int,
                          unpack=True)
    txt_read = np.transpose(txt_read) 
    #transposing so that the first index is the first frame
    Framedict = {}
#    print(f'file name = {str(p_file.stem)}')
#    print(f'range of len  = {len(txt_read)}')
    if len(txt_read)==8:
        print('single pattern')
        fr_no = 0 
        num_cols = txt_read[1]
        num_rows = txt_read[2]
        bright_idx = (txt_read[3:])-1
        arr_size = (num_rows,num_cols)
        zero_array = np.zeros([num_rows,num_cols])
        bright_idx_2d = np.transpose(np.unravel_index(bright_idx,arr_size))
        for j in range(len(bright_idx_2d)):
            zero_array[bright_idx_2d[j][0]][bright_idx_2d[j][1]] = 1
        Framedict[fr_no]=zero_array
    else:
        for i in range(len(txt_read)):
            fr_no = txt_read[i][0]
            num_cols = txt_read[i][1]
            num_rows = txt_read[i][2]
            bright_idx = (txt_read[i][3:])-1
            arr_size = (num_rows,num_cols)
            zero_array = np.zeros([num_rows,num_cols])
            bright_idx_2d = np.transpose(np.unravel_index(bright_idx,arr_size))
            for j in range(len(bright_idx_2d)):
                zero_array[bright_idx_2d[j][0]][bright_idx_2d[j][1]] = 1
            Framedict[fr_no]=zero_array
#    print(f'framedict keys = {Framedict.keys()}')
    return Framedict

def proj_file_selector(file_name,p_files):
    f = str(file_name)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split('\\')[-1]
    protocol_name = protocol_name.split('.')[-2]
    print(f'polygon protocol name = {protocol_name}')
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
#            plt.plot(t,trace,alpha=0.7, label = f'trace-{s}')
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
    outdir = p/'results'
    outdir.mkdir(exist_ok=True,parents=True)
    abf_list = list_files(p)
    p_files = pattern_files(c)
#    print(f'total no. of projection protocols ={len(p_files)}')
    for f_name in abf_list:
        try:
            print(f'{f_name}')
            plot_name = str(outdir)+'/'+str(f_name.stem)+'.png'
            p_file, title, zoom = proj_file_selector(f_name,p_files)
            frames = pattern_dict(p_file)   
#            fig, axs = plt.subplots(5,1, figsize = (9,10),sharex=True)
            fig, axs = plt.subplots(5,1, figsize = (15,10),sharex=True)
            for plt_no,chanel_name in enumerate(channel_names):
                plt_no = plt_no+1
#                print (f_name, chanel_name,plt_no)
                plot_all_cannels(f_name,chanel_name,plt_no, 
                                 fig, axs,frames, zoom)
            plt.xlabel('time(s)',fontproperties=y_labels)
#            if title=='Points':
#                xlim = [6.3,7]
#            elif title=='Patterns':
#                xlim = [4,5.4]
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

            if title=='Points':
                xlim = [1.25,11.25]
            elif title=='Patterns':
                xlim = [0.8,11]
            elif title=='Training pattern':
                axs[1].set_ylim(-70,60)
                xlim = left=0
                for i in range(5):
                    print(f'range = ({i})' )
                    if i>0:
                        axs[i].get_legend().remove()
                    else:
                        continue
            else:
                continue

            plt.xlim(xlim)
            plt.suptitle(f'Response to {title}',
                         fontproperties=main_title)
#            fig.tight_layout()
            fig.align_ylabels()
            plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None,
                                wspace=None, hspace=0.5)
#            plt.show(block=False)
            fig.savefig(plot_name, bbox_inches='tight')
#            plt.pause(2)
            plt.close()
        except:
            continue







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
