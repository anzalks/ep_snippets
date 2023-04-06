__author__           = "Anzal KS"
__copyright__        = "Copyright 2022-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import neo.io as nio
import numpy as np
import pandas as pd
from scipy import signal as spy
from scipy import stats as stats
from pprint import pprint
import multiprocessing
import time
import argparse
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing

class Args: pass 
args_ = Args()

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

"""
1D array and get locations with a rapid rise, N defines the rolling window
"""
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


"""
data filter function
"""
def filter_data(data, cutoff, filt_type, fs, order=3):
    b, a = spy.butter(order, cutoff, btype = filt_type, analog=False, output='ba', fs=fs)                                                                                     
    return spy.filtfilt(b, a, data) 

"""
Convert channel names to index as an intiger
"""
def channel_name_to_index(reader, channel_name):
    for signal_channel in reader.header['signal_channels']:
        if channel_name == signal_channel[0]:
            return int(signal_channel[1])

"""
function to find the protocol name for any abf file
"""
def protocol_file_name(file_name):
    f = str(file_name)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split('\\')[-1]
    protocol_name = protocol_name.split('.')[-2]
    return protocol_name
        
"""
Detects the file name with training data (LTP protocol) in it 
"""
def training_finder(f_name):
    protocol_name = protocol_file_name(f_name)
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

"""
Sort the list of suplied files into pre and post trainign files and return the list 
"""
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

"""
Tag protocols with training, patterns, rmp measure etc.. assign a title to the file
"""
def protocol_tag(file_name):
    protocol_name = protocol_file_name(file_name)
    if '12_points' in protocol_name:
        #print('point_protocol')
        title = 'Points'
    elif '42_points' in protocol_name:
        #print('point_protocol')
        title = 'Points'
    elif 'Baseline_5_T_1_1_3_3' in protocol_name:
        #print('pattern protocol')
        title = 'Patterns'
    elif 'patternsx' in protocol_name:
        #print('pattern protocol')
        title = 'Patterns'
    elif 'patterns_x' in protocol_name:
        #print('pattern protocol')
        title = 'Patterns'
    elif 'Training' in protocol_name:
        #print('training')
        title = 'Training pattern'
    elif 'training' in protocol_name:
        #print('training')
        title = 'Training pattern'
    elif 'RMP' in protocol_name:
        #print('rmp')
        title='rmp'
    elif 'Input_res' in protocol_name:
        #print ('InputR')
        title ='InputR'
    elif 'threshold' in protocol_name:
        #print('step_current')
        title = 'step_current'
    else:
        #print('non optical protocol')
        title = None
    return title

"""
Pair files pre and post with point, patterns, rmp etc..
"""
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
#    print(f'point files = {point} '
#           f'pattern files = {pattern}'
#          )
    return [point, pattern,rmp, InputR, step_current]

"""
pair pre and post, points and patterns for each cell.
"""
def file_pair(cell_path): 
    cell_id = str(cell_path.stem)
    abf_list = list_files(cell_path)
    sorted_f_list = pre_post_sorted(abf_list)
    pre_f_list = sorted_f_list[0]
    post_f_list = sorted_f_list[1][1:]
    training_f = sorted_f_list[1][0]
    no_c_train = sorted_f_list[2]
    paired_list = file_pair_pre_pos(pre_f_list, post_f_list)
    paired_points = paired_list[0]
    paired_patterns = paired_list[1]
    return [paired_points,paired_patterns]

"""
pattern label functions
"""
# plug in iteration umber and it returns a pattern type
def pat_selector(i):
    if i==0:
        pattern='Trained pattern'
    elif i==1:
        pattern='Overlapping pattern'
    elif i==2:
        pattern='Non overlapping pattern'
    else:
        pattern ='_NA'
    return pattern
def point_selector(i):
    if i<=4:
        point='Trained point'
    elif i>4:
        point='Untrained point'
    return point

"""
Injected_currentfinder
"""
def current_injected(reader):
    protocol_raw = reader.read_raw_protocol()
    protocol_raw = protocol_raw[0]
    protocol_trace = []
    for n in protocol_raw:
        protocol_trace.append(n[0])
    i_min = np.abs(np.min(protocol_trace))
    i_max = np.abs(np.max(protocol_trace))
    i_av = np.around((i_max-i_min),2)
    return i_av
"""
convert abf to a nested dictionary
"""
def abf_to_df(file_name,channel_name,pre_post_status):
    global df_from_abf
    df_from_abf = pd.DataFrame()
    f = str(file_name)
    reader = nio.AxonIO(f)
    channels = reader.header['signal_channels']
    chan_count = len(channels)
    file_id = file_name.stem
    block  = reader.read_block(signal_group_mode='split-all')
    segments = block.segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate.magnitude
    sampling_rate_unit = str(sample_trace.sampling_rate.units).split()[-1]
    ti = sample_trace.t_start
    tf = sample_trace.t_stop
    injected_current = current_injected(reader)
    
    TTL_sig_all = []
    for s, segment in enumerate(segments):
        ttl_ch_no = channel_name_to_index(reader,'FrameTTL')
        ttl_signal = segment.analogsignals[ttl_ch_no]
        ttl_unit = str(ttl_signal.units).split()[1]
        ttl_trace = np.array(ttl_signal)
        TTL_sig_all.append(ttl_trace) 
        t = np.linspace(0,float(tf-ti),len(ttl_trace))
    ttl_av = np.average(TTL_sig_all,axis=0 )
    
    ttl_xi= find_ttl_start(ttl_av, 3)
    
    #print (f' TTL len = {len(ttl_xi)}')
    if len(ttl_xi)<5:
        frame_type='pattern'
    else:
        frame_type='point'
    print()
    t=[]
    for s, segment in enumerate(segments):
        df_segment = pd.DataFrame()
        cell_ch_no = channel_name_to_index(reader,channel_name)
        field_ch_no = channel_name_to_index(reader,'Field')
        pd_ch_no = channel_name_to_index(reader,'Photodiode')
        ttl_ch_no = channel_name_to_index(reader,'FrameTTL')

        cell_signal = segment.analogsignals[cell_ch_no]
        cell_Signal_unit = str(cell_signal.units).split()[-1]
        cell_trace = np.hstack(np.ravel(np.array(cell_signal)))
        
        field_signal = segment.analogsignals[field_ch_no]
        field_signal_unit=str(field_signal.units).split()[-1]
        field_trace = np.hstack(np.ravel(np.array(field_signal)))
        
        pd_signal = segment.analogsignals[pd_ch_no]
        pd_signal_unit=str(pd_signal.units).split()[-1]
        pd_trace = np.hstack(np.ravel(np.array(pd_signal)))
        
        ttl_signal = segment.analogsignals[ttl_ch_no]
        ttl_signal_unit=str(ttl_signal.units).split()[-1]
        ttl_trace = np.hstack(np.ravel(np.array(ttl_signal)))

        t = np.linspace(0,float(tf-ti),len(cell_trace))
        """
        rmp=stats.mode(cell_trace)[0]
        
        IR_baseline =stats.mode(cell_trace[int(sampling_rate*(tf.magnitude-0.7)):-1])[0]
        if frame_type=='pattern':
            IR_dip =stats.mode(cell_trace[int(sampling_rate*(tf.magnitude-1)):int(sampling_rate*(tf.magnitude-0.8))])[0]
        else:
            IR_dip =stats.mode(cell_trace[int(sampling_rate*(tf.magnitude-2.15)):int(sampling_rate*(tf.magnitude-1.95))])[0]
        InputR = np.absolute(((IR_baseline-IR_dip)/injected_current)*1000) # multiplied by 1000 to convert to MOhms
        """
        df_segment['trial_no']=[s]
        #df_segment[f'rmp({cell_Signal_unit})']=rmp
        #df_segment[f'InputR(MOhm)']=InputR
        df_segment[f'cell_trace({cell_Signal_unit})']=[cell_trace]
        df_segment[f'field_trace({field_signal_unit})']=[field_trace]
        df_segment[f'pd_trace({pd_signal_unit})']=[pd_trace]
        df_segment[f'ttl_trace({ttl_signal_unit})']=[ttl_trace]
        df_segment['time(s)']=[t]
        df_from_abf = df_from_abf.append(df_segment,ignore_index=True)
    df_from_abf.insert(loc=0, column=f'sampling_rate({sampling_rate_unit})', value=sampling_rate)
    df_from_abf.insert(loc=0, column='pre_post_status', value=pre_post_status)    
    df_from_abf.insert(loc=0, column='frame_type', value=frame_type)
    return df_from_abf

"""
raw data from multiple abfs to dict, combine all the sorted abfs for a point or pattern to single nested dictionary
"""
def combine_abfs_for_one_frame_type(points_or_pattern_file_set_abf,cell_ID,ch_id='cell'):
    global all_frames_df
    if ch_id=='cell':
        ch_name='IN0'
    elif ch_id=='field':
        ch_name='Field'
    #print(f'ch Id = {ch_name}')
    pre_f = points_or_pattern_file_set_abf[0]
    post_f = points_or_pattern_file_set_abf[1:]
    #all_frames_df =pd.DataFrame()
    all_frames_df = abf_to_df(pre_f,ch_name,'pre')
    for ix,i in enumerate(post_f):
        m= abf_to_df(i,ch_name,f'post_{ix}')
        all_frames_df = all_frames_df.append(m,ignore_index=True) # cell_all_frames_data.append(n)
    all_frames_df.insert(loc=0, column='cell_ID', value=cell_ID)
    return all_frames_df
"""
combine all abfs related to one cell and give out a single pandas df from that for all frames (both patterns and points)

"""
def combine_frames_for_as_cell(cell_path):
    cell_ID = str(cell_path).split('/')[-1]
    files_paired = file_pair(cell_path)
    points_file_list = files_paired[0]
    patterns_file_list = files_paired[1]
    points_raw_df = combine_abfs_for_one_frame_type(points_file_list,cell_ID,ch_id='cell')
    patterns_raw_df = combine_abfs_for_one_frame_type(patterns_file_list,cell_ID,ch_id='cell')
    all_frames_raw_df = pd.concat([points_raw_df,patterns_raw_df],ignore_index=True)
    return all_frames_raw_df





def main():
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cells-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path of folder with folders as cell data '
                       )


#    parser.parse_args(namespace=args_)
    args = parser.parse_args()
    print(args.cells_path)
    p = Path(args.cells_path)
    print(p)
    outdir = p/'cell_health_dist_all_trial_all_frames'
    outdir.mkdir(exist_ok=True, parents=True)
    cells = list_folder(p)
    cells_dict = pd.DataFrame()
    total_cells = len(cells) 
    pool = multiprocessing.Pool(processes=5)
    processes = []
    results = []
    for cell_no, cell in enumerate(cells):
        progress_bar = tqdm(total=total_cells, desc="converting", unit="cell")
        cell_ID = str(cell).split('/')[-1]
        result =pool.apply_async(combine_frames_for_as_cell, args=(cell,))
        results.append(result)
        #cells_dict= pd.concat([cells_dict,result],ignore_index=True)
        progress_bar.update(1)
    pool.close()
    pool.join
    output = [result.get() for result in results]
    for i in output:
        cells_dict=cells_dict.append(i,ignore_index=True)
    progress_bar.close()
    cells_dict.to_pickle(f'{outdir}/cells_dict_from_multiproces.pkl')

if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {tf-ts} (s)')
