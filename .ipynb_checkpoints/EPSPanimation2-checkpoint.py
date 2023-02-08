def plot_optical(f, outdir):
    global t
    global trace
    f_n = f.stem
    outdir = f"{outdir}/{f_n}"
    f = str(f)
    file_name = f.split("/")[-1]
    reader = nio.AxonIO(f)
    channels = reader.header["signal_channels"]
    chan_count = len(channels)
    block = reader.read_block(signal_group_mode="split-all")
    segments = block.segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate
    ti = sample_trace.t_start
    tf = sample_trace.t_stop
    total_time = int(tf - ti)
    protocol_raw = reader.read_raw_protocol()
    protocol_raw = protocol_raw[0]
    print(total_time)
    unit = str(sample_trace.units).split()[1]
    all_chan_trace = []
    t_all = []
    print("total channel loop begin")
    for i in range(chan_count):
        trace_all = list()
        tr = list()
        for s, segment in enumerate(segments):
            analogsignals = segment.analogsignals[i]
            unit = str(analogsignals.units).split()[1]
            trace = np.array(analogsignals)
            trace = np.mean(trace, axis=1)
            trace_all.append(trace)
            t = np.linspace(0, float(tf - ti), len(trace))
            tr.append(t)
        trace_all = np.mean(trace_all, axis=0)
        tr = np.mean(tr, axis=0)
        all_chan_trace.append(trace_all)
        t_all.append(tr)
    t_all = np.mean(t_all, axis=0)
    data = [
        t_all,
        all_chan_trace[0],
        all_chan_trace[1],
        all_chan_trace[2],
        all_chan_trace[3],
    ]
    print(f" shape of data = {np.shape(data)}")
    data = [t_all, all_chan_trace[0]]
    print(f"shape of t, data ={np.shape(data[0])}, {np.shape(data[1])}")

    fig = plt.figure()
    axis = plt.axes(xlim=(0, 17), ylim=(-100, -30))
    (line,) = axis.plot([], [], lw=3)
    num_of_frames = 100
    num_of_sec_per_frame = 1
    frames = np.linspace(0, t_all[-1]*int(sampling_rate), num_of_frames)

    def init():
        line.set_data([],[])
        return (line,)

    def animate(i):
        # print("animating")
        x = data[0][: int(i + num_of_sec_per_frame*int(sampling_rate))]
        y = data[1][: int(i + num_of_sec_per_frame*int(sampling_rate))]

        line.set_data(x, y)
        # axis.set_xlim(min(x), max(x))
        # axis.set_xlim(data[0][int(i)], data[0][int(i + num_of_sec_per_frame*int(sampling_rate))])

        return (line,)

    # anim = FuncAnimation(
    #     fig, animate, init_func=init, frames=len(data[0])-25000, blit=True
    # )
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, blit=True
    )
    anim.save(f"plot_anim.gif", fps=num_of_frames/t_all[-1])
