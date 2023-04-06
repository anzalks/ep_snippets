__author__           = "Anzal KS"
__copyright__        = "Copyright 2022-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import time

class Args: pass 
args_ = Args()

def list_files(p):
    f_list = []
    f_list=list(p.glob('**/*pkl'))
    f_list.sort()
    return f_list


def plot_rmp_raw_plot(all_cell_df,outdir):
    injected_current = -20
    rmp_mean=pd.DataFrame()
    out_folder = outdir/'rmp_dist_all_trial_all_frames'
    out_folder.mkdir(exist_ok=True, parents=True)
    all_cell_df_cp = all_cell_df.copy()
    rmp_var_all_cells=[]
    trials = all_cell_df['trial_no'].unique()
    v_cell =0
    for c, cell in enumerate(all_cell_df['cell_ID'].unique()):
        fig, axs = plt.subplots(3,1, figsize = (12,10)) 
        axs=axs.flatten()
        outfile = f'{out_folder/cell}'
        print(outfile)
        rmp_cell_protocols =[]
        inputR_cell_protocols =[]
        inputR_cell_protocols_trial_av = []
        for i,rows in all_cell_df[all_cell_df['cell_ID']==cell].iterrows():
            sampling_rate = rows['sampling_rate(Hz)']
            rmp_all_trials=[]
            inputR_all_trials = []
            for trial in trials:
                if rows['trial_no']==trial:
                    sampling_rate = rows['sampling_rate(Hz)']
                    ts,tf =rows['time(s)'][0],rows['time(s)'][-1]
                    frame_type = rows['frame_type']
                    IR_baseline =np.median(rows['cell_trace(mV)'][int(sampling_rate*(tf-0.7)):-1])
                    if frame_type=='pattern':
                        IR_dip =np.median(rows['cell_trace(mV)'][int(sampling_rate*(tf-1)):int(sampling_rate*(tf-0.8))])
                    else:
                        IR_dip =np.median(rows['cell_trace(mV)'][int(sampling_rate*(tf-2.15)):int(sampling_rate*(tf-1.95))])
                    inputR = np.absolute(((IR_baseline-IR_dip)/injected_current)*1000) # multiplied by 1000 to convert to MOhms
                    rmp = np.median(rows['cell_trace(mV)'])
                    rmp_all_trials.append(rmp)
                    inputR_all_trials.append(inputR)
                    #print(f'rmp = {rmp}, inputR = {inputR}')
                    axs[0].plot(rows['time(s)'],rows['cell_trace(mV)'],alpha=0.3)
                    #print(f'rows inside trial loop {rows[0:4]}')
                    #print(f'trial numer....... = {trial}')
                else:
                    continue
            rmp_all_trials = np.array(rmp_all_trials)
            inputR_all_trials = np.array(inputR_all_trials)
            rmp_trial_av = np.mean(rmp_all_trials)
            inputR_trial_av = np.mean(inputR_all_trials)
            
            rmp_cell_protocols.append(rmp_all_trials)
            inputR_cell_protocols.append(inputR_all_trials)
            inputR_cell_protocols_trial_av.append(inputR_trial_av)
        
        rmp_cell_protocols = np.array(rmp_cell_protocols)
        inputR_cell_protocols = np.array(inputR_cell_protocols)
        inputR_cell_protocols_trial_av = np.array(inputR_cell_protocols_trial_av)
        inputR_cell_protocols_trial_av = np.reshape(inputR_cell_protocols_trial_av,(int(len(inputR_cell_protocols_trial_av)/3),3))
        inputR_cell_protocols_trial_av = np.mean(inputR_cell_protocols_trial_av,axis=1)
        num_protocols = np.arange(0,np.shape(inputR_cell_protocols_trial_av)[0],1)
        
        inputR_cell_protocols_std = np.std(inputR_cell_protocols_trial_av)
        inputR_percentage_cut = np.around(inputR_cell_protocols_trial_av[0]*0.15)
        inputR_cut_up =inputR_cell_protocols_trial_av[0]+inputR_percentage_cut
        inputR_cut_lo =inputR_cell_protocols_trial_av[0]-inputR_percentage_cut
        
        
        rmp_cell_av = np.mean(rmp_cell_protocols)
        rmp_cell_var = np.var(rmp_cell_protocols)
        
        if (rmp_cell_var<0.3)and(inputR_cut_lo <= inputR_cell_protocols_trial_av[1] <= inputR_cut_up):
            print(f'valid  cell {cell}, {c}')
            valid_status = 'valid'
            v_cell +=1
        else:
            print(f'......')
            valid_status = 'not valid'
        
        
        inputR_cell_av = np.mean(inputR_cell_protocols_trial_av)
        rmp_var = str(np.around(rmp_cell_var,2))
        
        axs[0].axhline(rmp_cell_av,color='r',label='mean')
        axs[0].axhline(rmp_cell_av+rmp_cell_var,color='k',alpha=0.7,label='varience')
        axs[0].axhline(rmp_cell_av-rmp_cell_var,color='k',alpha=0.7)
        
        axs[0].axhline(rmp_cell_av+1.5,color='g',alpha=0.7,label ='1.5 mV cutoff')
        axs[0].axhline(rmp_cell_av-1.5,color='g',alpha=0.7)
        
        axs[0].set_ylim(-75,-55)
        axs[0].set_title(f'raw traces, all trial all frames')
        axs[0].set_xlabel('time(s)')
        axs[0].set_ylabel('cell mem pot (mV)')
        
        
        axs[1].scatter(num_protocols,inputR_cell_protocols_trial_av)
        axs[1].axhline(inputR_cell_av,color='r')
        axs[1].axhline(inputR_cell_protocols_trial_av[0]+inputR_percentage_cut,color='b',alpha=0.7,label='15% of initial InputR')
        axs[1].axhline(inputR_cell_protocols_trial_av[0]-inputR_percentage_cut,color='b',alpha=0.7)
        axs[1].axvline(1.5,color='k',label ='ltp induction')
        
        axs[1].set_title(f'InputR averaged over 3 trials for each protocol')
        axs[1].set_ylim(40,250)
        axs[1].set_xlabel('protocol number')
        axs[1].set_ylabel('InputR (MOhms)')
        
        
        axs[2].scatter(np.arange(0,len(inputR_cell_protocols),1),inputR_cell_protocols)
        axs[2].axhline(inputR_cell_av,color='r')
        axs[2].axhline(inputR_cell_protocols_trial_av[0]+inputR_percentage_cut,color='b',alpha=0.7)
        axs[2].axhline(inputR_cell_protocols_trial_av[0]-inputR_percentage_cut,color='b',alpha=0.7)
        axs[2].axvline(6.5,color='k')
        
        axs[2].set_title(f'InputR all trials all protocols')
        axs[2].set_ylim(40,250)
        axs[2].set_xlabel('trial number')
        axs[2].set_ylabel('InputR (MOhms)')        
            
        plt.suptitle(f'cell_ID: {cell} | validity: {valid_status}')
        fig.legend(loc='upper center',bbox_to_anchor =(0.5, -0.02), ncol = 4)
        plt.tight_layout()
        plt.savefig(f'{outfile}.png',bbox_inches='tight')
    print(f'total v_cell number = {v_cell}')
    



def main():
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cells-path', '-f'
                        , required = False,default ='./', type=str
                        , help = 'path of folder with folders as cell data '
                       )
    
    args = parser.parse_args()
    print(args.cells_path)
    pickle_file_path = Path(args.cells_path)
    file_path = pickle_file_path.parent
    print(file_path)
    outdir = file_path/'cell_health_dist_all_trial_all_frames'
    outdir.mkdir(exist_ok=True, parents=True)
    
    all_cell_df = pd.read_pickle(pickle_file_path)
    plot_rmp_raw_plot(all_cell_df,outdir)


if __name__  == '__main__':
    #timing the run with time.time
    ts =time.time()
    main(**vars(args_)) 
    tf =time.time()
    print(f'total time = {tf-ts} (s)')