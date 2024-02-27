import matplotlib.pyplot as plt
import numpy as np
import typing

class dataPlot:
    def __init__(self, keys:typing.Tuple[str], min_window_size=11) -> None:
        self.keys = tuple(keys)
        self.fig_num = len(keys)
        self.fig, self.axs = plt.subplots(self.fig_num, 1, sharex=True, figsize=plt.figaspect(self.fig_num/2))
        self.axs = [self.axs] if self.fig_num == 1 else self.axs
        self.objs = [ [] for _ in range(self.fig_num) ]

        for i in range(self.fig_num):
            key = self.keys[i]
            if key=="critic_loss":
                self.axs[i].set_yscale("log")
            self.objs[i] += self.axs[i].plot([0,1], color="lightsteelblue") # Line2D, data
            self.objs[i] += self.axs[i].plot([0,1], color="midnightblue") # Line2D, moving average
            self.axs[i].set_title(key)

        self.min_window_size = min_window_size
        self.auto_window_size = True

        # plt.show()

    @classmethod
    def from_log(cls, file, min_window_size=11):
        log = np.load(file)
        this = cls(log.keys(), min_window_size=min_window_size)
        datas = []
        for key in log.keys():
            datas.append(log[key])
        datas = tuple(datas)
        this.set_data(datas)
        return this

    def set_data(self, data:typing.Tuple[typing.Iterable]):
        for i in range(self.fig_num):
            l = len(data[i])
            if self.auto_window_size and (aws:=int(l/100))>self.min_window_size:
                window_size = aws if aws%2==1 else aws+1
            else:
                window_size = self.min_window_size
            v_ma = moving_average(data[i], window_size=window_size)
            self.objs[i][0].set_data(np.arange(len(data[i])), data[i])
            self.objs[i][1].set_data(np.arange(len(v_ma)), v_ma)
            self.axs[i].set_xlim(0, len(data[i]))
            self.axs[i].set_ylim(min(data[i]), max(data[i]))
        plt.draw()
        
    def save_fig(self, file_name:str):
        plt.show()
        self.fig.savefig(file_name)

def logPlot(file:str,
            window_size=21,
            start=0,
            end=-1,
            show=True,
            save_file:str=None):
    '''
        draw plot of values, actor loss and critic loss from `.npz` file.
    '''
    log = np.load(file)
    fig_num = len(log.keys())
    fig, axs = plt.subplots(fig_num, 1, sharex=True, figsize=plt.figaspect(fig_num/2))
    if fig_num == 1:
        axs = [axs]
    for i in range(fig_num):
        key = list(log.keys())[i]
        data = log[key][start:end]
        axs[i].plot(data, color="lightsteelblue")
        v_ma = moving_average(data, window_size=window_size)
        axs[i].plot(v_ma, color="midnightblue")
        axs[i].set_title(key)
    log.close()
    if show:
        plt.show()
    if save_file:
        fig.savefig(save_file)
    return fig

def moving_average(a, window_size:int=11):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def concat_log(files:list,
              file_name:str):
    is_init = True
    d = {}
    for file in files:
        log = np.load(file)
        if is_init:
            for key in log.keys():
                d[key] = log[key]
            is_init = False
        else:
            for key in log.keys():
                d[key] = np.hstack((d[key], log[key]))
        log.close()
    np.savez(file_name, **d)
    return