#!/usr/bin/env python3
__author__           = "Anzal KS"
__copyright__        = "Copyright 2019-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from pathlib import Path
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neo.io as nio
import scipy.signal as signal
from pprint import pprint

class Args: pass 
args_ = Args()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def input_R_protocol(f, fi, Vm_trail, sampling_rate,trace_unit, protocol_unit):
    fig = plt.figure(figsize=(16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    iter_num = 0
    mean_R=[]
    for v in  enumerate(Vm_trail):
        iter_num +=1
        trace_number = v[0]
        trace = v[1][0]
        time = v[1][1]
        Vb= np.mean(trace[int(sampling_rate*0.35):int(sampling_rate*0.38)])
        Vl= np.mean(trace[int(sampling_rate*0.15):int(sampling_rate*0.20)])
        input_R = (np.around((Vb-Vl),decimals=2)*1000)/(50)
        mean_R.append(input_R)
        if iter_num ==2:
            ax1.plot(time, trace, label = f'trace no. {iter_num}', alpha = 0.7)
            ax1.scatter(time[int(sampling_rate*0.35)],Vb, color = 'r', 
                        label ='baseline')
            ax1.scatter(time[int(sampling_rate*0.20)],Vl, color = 'k', 
                             label ='input_V')
        P_traces= protocols[0]
        for p in P_traces:
            for i in p:
                t_ = len(i)/sampling_rate
                t = np.linspace(0,float(t_), len(i))
                ax2.plot(t,i)
    mean_R = np.mean(mean_R)
    ax1.set_title('Recording')
    ax1.set_ylabel(trace_unit)
    ax1.set_xlabel('time(s)')
    ax1.legend()
    ax2.set_title('Protocol trace')
    ax2.set_ylabel(protocol_unit)
    ax2.set_xlabel('time(s)')
    #        ax2.legend()
    plt.figtext(0.10,-0.05, f"Input resistance (averaged from {iter_num} traces) = "+
                str(np.around(mean_R,decimals =2))+" MOhm ", fontsize=12, va="top", ha="left")
    plt.suptitle(f'Protocol type: {protocol_type} - {protocol_used}',fontsize=15)
    plt.figtext(0.10, -0.10, f"sampling rate = {sampling_rate}" ,
                fontsize=12, va="top", ha="left" )
    plt.figtext(0.10, -0.15, f"total recording time ="
                f" {np.around(total_time,decimals = 2)} s" ,
                fontsize=12, va="top", ha="left")
    outfile = str(outdir)+"/"+str(f.stem)+ f" {protocol_used}_{fi}.png"
    plt.savefig(outfile,bbox_inches = 'tight')
    print("-----> Saved to %s" % outfile)
    fig = plt.close()


def Base_line_protocol(f, fi, Vm_trail, sampling_rate,trace_unit, protocol_unit):
    fig = plt.figure(figsize=(16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    RMP = []
    iter_num = 0
    for v in  enumerate(Vm_trail):
        trace_number = v[0]
        trace = v[1][0]
        time = v[1][1]
        mean_RMP = np.mean(trace)
        if mean_RMP <-50:
            iter_num +=1
            ax1.plot(time, trace, label=f'trace no. {iter_num}', alpha = 0.5)
            P_traces= protocols[0]
            RMP = mean_RMP
            for p in P_traces:
                for i in p:
                    t_ = len(i)/sampling_rate
                    t = np.linspace(0,float(t_), len(i))
                    ax2.plot(t,i)
#        First_inj = mpatches.Patch(color='green', label='First injection')
#        Thres_inj = mpatches.Patch(color='black', label='Threshold injection')
#        Last_inj = mpatches.Patch(color='red', label='Final injection')
#        ax2.legend(handles=[First_inj,Thres_inj,Last_inj])
    ax1.set_title('Recording')
    ax1.set_ylabel(trace_unit)
    ax1.set_xlabel('time(s)')
    ax1.legend()
    ax1.set_ylim(-90,-20)
    ax2.set_title('Protocol trace')
    ax2.set_ylabel(protocol_unit)
    ax2.set_xlabel('time(s)')
    #        ax2.legend()
    plt.figtext(0.10,-0.05, "Resting membrane potential average from"
                f" {iter_num} traces= "+
                str(np.around(RMP,decimals = 2))+" mV", fontsize=12, va="top", ha="left")
    plt.suptitle(f'Protocol type: {protocol_type} - {protocol_used}',fontsize=15)
    plt.figtext(0.10, -0.10, f"sampling rate = {sampling_rate}" ,
                fontsize=12, va="top", ha="left" )
    plt.figtext(0.10, -0.15, f"total recording time = {total_time} s" ,
                fontsize=12, va="top", ha="left")
    outfile = str(outdir)+"/"+str(f.stem)+ f" {protocol_used}_{fi}.png"
    plt.savefig(outfile,bbox_inches = 'tight')
    print("-----> Saved to %s" % outfile)
    fig = plt.close()

def threshold_protocol(f, fi, Vm_trail, sampling_rate,trace_unit,
                       protocol_unit):
    global Threshold_voltage
    fig = plt.figure(figsize=(16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    thresh_state = 0
    iter_num = 0
    trace_num = 0
    for v in  enumerate(Vm_trail):
        iter_num +=1
        trace_number = v[0]
        trace = v[1][0]
        time = v[1][1]
        v = np.copy(trace)
        t = np.copy(time)
        if thresh_state == 0:
            Vm = str(np.around(np.mean(v[0:299]),decimals=2))
            del_Vm = str(np.around((np.mean(v[len(v)-300:len(v)])-
                                np.mean(v[0:299])), decimals= 2))
            v_lowercut = v
            t = time
            v_lowercut[v_lowercut<-50] = -50
            v_smooth = butter_bandpass_filter(v_lowercut,1, 500, sampling_rate, order=1)
            peaks = signal.find_peaks(x=v_smooth, height=None,  threshold=None, distance=None, 
                                      prominence=5, width=None, wlen=None, rel_height=0.5, 
                                      plateau_size=None)
            v_cut = butter_bandpass_filter(v_smooth,50, 500, sampling_rate, order=1)
            v_peaks = v_smooth
            t_peaks = t
            thr_peak = 3
            if trace[peaks[0]]>0:
                thresh_state = 1
                dv = np.diff(v_smooth)
                dt = np.diff(t)
                dv_dt = dv/dt
                dv_dt_max = np.max(dv/dt)
                v_dt_max = np.where(dv_dt == dv_dt_max)[0]-20
                t_dt_max = np.where(dv_dt == dv_dt_max)[0]-20
                ax1.scatter(time[peaks[0]-10], trace[peaks[0]-10], color='r', label = 'spike')
                ax1.plot(t, v,alpha = 0.5, label = 'smoothened')
                ax1.plot(time, trace, alpha = 0.5, label=f'raw trace no. {iter_num}')
                ax1.scatter(time[t_dt_max],trace[v_dt_max], label = "threshold",
                            color = 'k')
                Threshold_voltage = "firing threshold = "+str(np.around(trace[v_dt_max][0],
                                                                        decimals=2))
                trace_num = iter_num
                plt.figtext(0.10,0.0, Threshold_voltage+"mV", fontsize=12,
                            va="top", ha="left")
                plt.figtext(0.10,-0.05, "membrane voltage = "+
                            Vm +"mV", fontsize=12, va="top", ha="left")
                plt.figtext(0.10,-0.10, "membrane voltage difference = "+
                            del_Vm +"mV", fontsize=12, va="top", ha="left")
        P_traces= protocols[0]
        iter_num_p = 0
        Threshold_injection = "NA"
        for p in P_traces:
            iter_num_p +=1
            if iter_num_p == 1:
                for i in p:
                    t_ = len(i)/sampling_rate
                    t = np.linspace(0,float(t_), len(i))
                    ax2.plot(i, color = 'g')

            elif iter_num_p == trace_num:
                c_inj = []
                for i in p:
                    t_ = len(i)/sampling_rate
                    t = np.linspace(0,float(t_), len(i))
                    ax2.plot(i, color = 'k')
                    c_inj.append(i)
                Threshold_injection = "Injected current at threshold =  "+ str(np.max(c_inj))

            elif iter_num_p == len(P_traces):
                for i in p:
                    t_ = len(i)/sampling_rate
                    t = np.linspace(0,float(t_), len(i))
                    ax2.plot(i, color = 'r')
            First_inj = mpatches.Patch(color='green', label='First injection')
            Thres_inj = mpatches.Patch(color='black', label='Threshold injection')
            Last_inj = mpatches.Patch(color='red', label='Final injection')
            ax2.legend(handles=[First_inj,Thres_inj,Last_inj])


    plt.figtext(0.55,0.0, Threshold_injection+"pA", fontsize=12,
                va="top", ha="left")
    ax1.set_title('Recording')
    ax1.set_ylabel(trace_unit)
    ax1.set_xlabel('time(s)')
    ax1.legend()
    ax2.set_title('Protocol trace')
    ax2.set_ylabel(protocol_unit)
    ax2.set_xlabel('time(s)')
    #        ax2.legend()
    plt.suptitle(f'Protocol type: {protocol_type} - {protocol_used}',fontsize=15)
    plt.figtext(0.10, -0.15, f"sampling rate = {sampling_rate}" ,
                fontsize=12, va="top", ha="left" )
    plt.figtext(0.10, -0.20, f"total recording time = {total_time}" ,
                fontsize=12, va="top", ha="left")
    outfile = str(outdir)+"/"+str(f.stem)+ f" {protocol_used}_{fi}.png"
    plt.savefig(outfile,bbox_inches = 'tight')
    print("-----> Saved to %s" % outfile)
    fig = plt.close()

def List_files(p):
    global f_list
    global outdir
    f_list = []
    f_list=list(p.glob('**/*abf'))
    outdir = p/'results'
    outdir.mkdir(exist_ok=True, parents=True)
    return f_list

def total_trace(f):
    global Vm_trail
    global sampling_rate
    global trace_unit
    global total_time
    f= str(f)
    Vm_trail = []
    reader = nio.AxonIO(filename=f)
    segments = reader.read_block().segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate
    trace_unit = str(sample_trace.units).split()[1]
#    print(sampling_rate)
#    print(trace_unit)
    for si, segment in enumerate(segments):
        analog_signals = segment.analogsignals
        for trace in analog_signals:
            v = trace
            v = np.ravel(v)
            v = v.magnitude
            tf = trace.t_stop
            ti = trace.t_start
            t = np.linspace(0,float(tf - ti), len(v))
            m = [v,t]
            Vm_trail.append(m)
            total_time = float(tf-ti)
    return Vm_trail

def protocol_class(f):
    global protocol_type
    global protocols
    global protocol_unit
    protocols = []
    reader = nio.AxonIO(f)
    protocols = reader.read_raw_protocol()
    clamp_stat = protocols[2][0]
    if clamp_stat == 'pA':
        protocol_type = 'Current_clamp'
    elif clamp_stat == 'mV':
        protocol_type = 'Voltage_clamp'
    else:
        protocol_type = 'Unkown'
    protocol_unit = clamp_stat
#    print(f'protocol type is {protocol_type}')
    return protocol_type

def protocol_ckecker(protocol_type, protocols):
    global protocol_used
    protocol_used = 'unidentified protocol'
    print("*********")
    if protocol_type == 'Current_clamp':
        len_proto = len(protocols[0])
        try:
            trace_min = np.min(protocols[0][0][0])
        except:
            print(f" this file has CC recording but the protocol trace look"
                  f"like this {protocols}")
            trace_min = np.nan
        if len_proto == 20:
            protocol_used = 'Threshold_check'
            print(f"the current clamp protcol '{protocol_used}' has"
                  f" {len_proto} traces in the protocol")
        elif trace_min == -50:
            protocol_used = 'input_res_check'
            print(f"the current clamp protcol '{protocol_used}' has"
                  f" {len_proto} traces in the protocol")
        elif trace_min == 0.0:
            if len_proto ==3:
                protocol_used = 'Base_line_V'
                print(f"the current clamp protcol '{protocol_used}' has"
                      f" {len_proto} traces in the protocol")
            else:
                protocol_used = "something_unknown"
        elif np.isnan(trace_min):
            protocol_used = "Unknown CC protocol"
            print(f"the current clamp protcol '{protocol_used}' has"
                  f" {len_proto} traces in the protocol")
    elif protocol_type == 'Voltage_clamp':
        print("VC")
        len_proto = len(protocols)
        print(len_proto)
        protocol_used = "VC_trials"

    else:
        print('unidenitfied protocol')
    print("*****")
    return protocol_used

def main(**kwargs):
    p = Path(kwargs['folder_path'])
    outdir = p/'results'
    outdir.mkdir(exist_ok=True,parents=True)
    List_files(p)
    for fi, f in enumerate(f_list):
        print(f"analysing file: {str(f)} ")
        total_trace(f)
        protocol_class(f)
        protocol_ckecker(protocol_type, protocols)
        if protocol_used == 'Threshold_check':
            continue
            threshold_protocol(f, fi, Vm_trail, sampling_rate,trace_unit,
                               protocol_unit)
        elif protocol_used == 'Base_line_V':
            Base_line_protocol(f, fi, Vm_trail, sampling_rate,trace_unit, protocol_unit)
        elif protocol_used == 'input_res_check':
            input_R_protocol(f, fi, Vm_trail, sampling_rate,trace_unit, protocol_unit)
        else:
            print("not threshold")

if __name__  == '__main__':
    import argparse
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--folder-path', '-f'
            , required = False, default ='./', type=str
            , help = 'path of folder with  abf files '
            )
    parser.parse_args(namespace=args_)
    main(**vars(args_)) 

