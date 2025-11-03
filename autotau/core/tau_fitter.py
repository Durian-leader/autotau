import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TauFitter:
    """
    用于拟合指数上升/下降过程的tau值的类
    
    可以用于拟合开启过程（上升指数）和关闭过程（下降指数）的时间常数tau
    """
    
    def __init__(self, time, signal, t_on_idx=None, t_off_idx=None, normalize=False, language='en'):
        """
        初始化TauFitter类
        
        参数:
        -----
        time : array-like
            时间数据
        signal : array-like
            信号数据
        t_on_idx : list [start, end], optional
            开启过程的时间范围 [开始时间, 结束时间]
        t_off_idx : list [start, end], optional
            关闭过程的时间范围 [开始时间, 结束时间]
        normalize : bool, optional
            是否将信号归一化到0-1范围
        language : str, optional
            语言选择 ('cn'为中文, 'en'为英文)
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.t_on_idx = t_on_idx
        self.t_off_idx = t_off_idx
        self.tau_on_popt = None
        self.tau_off_popt = None
        self.tau_on_pcov = None
        self.tau_off_pcov = None
        self.tau_on_r_squared = None
        self.tau_off_r_squared = None
        self.tau_on_r_squared_adj = None
        self.tau_off_r_squared_adj = None
        self.normalize = normalize
        self.language = language  # 'cn' for Chinese, 'en' for English
        
        # 语言字典
        self.text = {
            'cn': {
                'fit_fail_on': '拟合开启过程失败: ',
                'fit_fail_off': '拟合关闭过程失败: ',
                'fit_first_on': '请先进行开启过程拟合',
                'fit_first_off': '请先进行关闭过程拟合',
                'orig_data': '原始数据',
                'fit_curve': '拟合曲线',
                'on_process': '开启过程拟合',
                'off_process': '关闭过程拟合',
                'time': '时间 (s)',
                'signal': '信号',
                'fit_equation': '拟合方程',
                'adj_r_squared': '调整R平方',
                'r_squared': 'R平方'
            },
            'en': {
                'fit_fail_on': 'Failed to fit turn-on process: ',
                'fit_fail_off': 'Failed to fit turn-off process: ',
                'fit_first_on': 'Please fit turn-on process first',
                'fit_first_off': 'Please fit turn-off process first',
                'orig_data': 'Original Data',
                'fit_curve': 'Fitted Curve',
                'on_process': 'Turn-on Process Fitting',
                'off_process': 'Turn-off Process Fitting',
                'time': 'Time (s)',
                'signal': 'Signal',
                'fit_equation': 'Fitting Equation',
                'adj_r_squared': 'Adjusted R Square',
                'r_squared': 'R Square'
            }
        }

    def set_language(self, language):
        """设置语言 / Set language"""
        if language in ['cn', 'en']:
            self.language = language
        else:
            raise ValueError("Language must be 'cn' or 'en'")

    def fit_tau_on(self, time=None, signal=None, interp=True, points_after_interp=100):
        """拟合开启过程的tau值 / Fit tau value for turn-on process"""
        if time is None and signal is None:
            mask = (self.time >= self.t_on_idx[0]) & (self.time <= self.t_on_idx[1])
            t_on = self.time[mask]
            signal_on = self.signal[mask]
        elif (time is None and signal is not None) or (time is not None and signal is None):
            raise ValueError("time和signal必须同时为None或同时不为None")    
        else:
            t_on = np.array(time)
            signal_on = np.array(signal)
            
        t_on = t_on - t_on[0]
        
        if self.normalize:
            signal_on = self.normalize_signal(signal_on)

        if interp:
            t_dense = np.linspace(t_on[0], t_on[-1], points_after_interp)
            current_dense = np.interp(t_dense, t_on, signal_on)
        else:
            t_dense = t_on
            current_dense = signal_on
            
        try:
            popt, pcov = curve_fit(self.exp_rise, t_dense, current_dense, 
                                 maxfev=100_000, 
                                 bounds=((0, 0, -np.inf), (np.inf, np.inf, np.inf)))
            tau_on_r_squared, tau_on_r_squared_adj = self.compute_r_squared(t_on, signal_on, popt, self.exp_rise)
            if time is None and signal is None:
                self.tau_on_popt = popt
                self.tau_on_pcov = pcov
                self.tau_on_r_squared = tau_on_r_squared
                self.tau_on_r_squared_adj = tau_on_r_squared_adj
            return popt, pcov, tau_on_r_squared, tau_on_r_squared_adj
        except RuntimeError as e:
            print(f"{self.text[self.language]['fit_fail_on']}{str(e)}")
            return None, None, None, None

    def fit_tau_off(self, time=None, signal=None, interp=True, points_after_interp=100):
        """拟合关闭过程的tau值 / Fit tau value for turn-off process"""
        if time is None and signal is None:
            mask = (self.time >= self.t_off_idx[0]) & (self.time <= self.t_off_idx[1])
            t_off = self.time[mask]
            signal_off = self.signal[mask]
        elif (time is None and signal is not None) or (time is not None and signal is None):
            raise ValueError("time和signal必须同时为None或同时不为None")
        else:
            t_off = np.array(time)
            signal_off = np.array(signal)
            
        t_off = t_off - t_off[0]
            
        if self.normalize:
            signal_off = self.normalize_signal(signal_off)  

        if interp:
            t_dense = np.linspace(t_off[0], t_off[-1], points_after_interp)
            current_dense = np.interp(t_dense, t_off, signal_off)
        else:
            t_dense = t_off
            current_dense = signal_off
            
        try:
            popt, pcov = curve_fit(self.exp_decay, t_dense, current_dense,
                                 maxfev=100_000,
                                 bounds=((0, 0, -np.inf), (np.inf, np.inf, np.inf)))
            tau_off_r_squared, tau_off_r_squared_adj = self.compute_r_squared(t_off, signal_off, popt, self.exp_decay)
            if time is None and signal is None: 
                self.tau_off_popt = popt
                self.tau_off_pcov = pcov
                self.tau_off_r_squared = tau_off_r_squared
                self.tau_off_r_squared_adj = tau_off_r_squared_adj
            return popt, pcov, tau_off_r_squared, tau_off_r_squared_adj
        except RuntimeError as e:
            print(f"{self.text[self.language]['fit_fail_off']}{str(e)}")
            return None, None, None, None

    def get_tau_on(self):
        """获取开启过程的tau值 / Get tau value for turn-on process"""
        if self.tau_on_popt is None:
            return None
        return self.tau_on_popt[1]
        
    def get_tau_off(self):
        """获取关闭过程的tau值 / Get tau value for turn-off process"""
        if self.tau_off_popt is None:
            return None
        return self.tau_off_popt[1]

    def plot_tau_on(self, figsize=(6, 4)):
        """绘制开启过程的tau值 / Plot tau value for turn-on process"""
        if self.tau_on_popt is None:
            print(self.text[self.language]['fit_first_on'])
            return
            
        mask = (self.time >= self.t_on_idx[0]) & (self.time <= self.t_on_idx[1])
        t_plot = self.time[mask]
        s_plot = self.signal[mask]
        
        plt.figure(figsize=figsize)
        plt.plot(t_plot, s_plot, 'o', label=self.text[self.language]['orig_data'])
        t_fit = np.linspace(t_plot[0], t_plot[-1], 100)
        plt.plot(t_fit, self.exp_rise(t_fit - t_fit[0], *self.tau_on_popt), '-', 
                label=self.text[self.language]['fit_curve'])
        plt.title(self.text[self.language]['on_process'])
        plt.xlabel(self.text[self.language]['time'])
        plt.ylabel(self.text[self.language]['signal'])
        
        text_cn = (f'{self.text["cn"]["fit_equation"]}: y = A(1-e^(-t/τ)) + C\n'
                  f'τ_on = {self.get_tau_on():.5f} s\n'
                  f'A = {self.tau_on_popt[0]:.5f}\n'
                  f'C = {self.tau_on_popt[2]:.5f}\n'
                  f'{self.text["cn"]["r_squared"]} = {self.tau_on_r_squared:.5f}\n'
                  f'{self.text["cn"]["adj_r_squared"]} = {self.tau_on_r_squared_adj:.5f}')
                  
        text_en = (f'{self.text["en"]["fit_equation"]}: y = A(1-e^(-t/τ)) + C\n'
                  f'τ_on = {self.get_tau_on():.5f} s\n'
                  f'A = {self.tau_on_popt[0]:.5f}\n'
                  f'C = {self.tau_on_popt[2]:.5f}\n'
                  f'{self.text["en"]["r_squared"]} = {self.tau_on_r_squared:.5f}\n'
                  f'{self.text["en"]["adj_r_squared"]} = {self.tau_on_r_squared_adj:.5f}')
                  
        plt.text(0.05, 0.95, text_cn if self.language == 'cn' else text_en,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_tau_off(self, figsize=(6, 4)):
        """绘制关闭过程的tau值 / Plot tau value for turn-off process"""
        if self.tau_off_popt is None:
            print(self.text[self.language]['fit_first_off'])
            return
            
        mask = (self.time >= self.t_off_idx[0]) & (self.time <= self.t_off_idx[1])
        t_plot = self.time[mask]
        s_plot = self.signal[mask]
        
        plt.figure(figsize=figsize)
        plt.plot(t_plot, s_plot, 'o', label=self.text[self.language]['orig_data'])
        t_fit = np.linspace(t_plot[0], t_plot[-1], 100)
        plt.plot(t_fit, self.exp_decay(t_fit - t_fit[0], *self.tau_off_popt), '-', 
                label=self.text[self.language]['fit_curve'])
        plt.title(self.text[self.language]['off_process'])
        plt.xlabel(self.text[self.language]['time'])
        plt.ylabel(self.text[self.language]['signal'])
        
        text_cn = (f'{self.text["cn"]["fit_equation"]}: y = Ae^(-t/τ) + C\n'
                  f'τ_off = {self.get_tau_off():.5f} s\n'
                  f'A = {self.tau_off_popt[0]:.5f}\n'
                  f'C = {self.tau_off_popt[2]:.5f}\n'
                  f'{self.text["cn"]["r_squared"]} = {self.tau_off_r_squared:.5f}\n'
                  f'{self.text["cn"]["adj_r_squared"]} = {self.tau_off_r_squared_adj:.5f}')
                  
        text_en = (f'{self.text["en"]["fit_equation"]}: y = Ae^(-t/τ) + C\n'
                  f'τ_off = {self.get_tau_off():.5f} s\n'
                  f'A = {self.tau_off_popt[0]:.5f}\n'
                  f'C = {self.tau_off_popt[2]:.5f}\n'
                  f'{self.text["en"]["r_squared"]} = {self.tau_off_r_squared:.5f}\n'
                  f'{self.text["en"]["adj_r_squared"]} = {self.tau_off_r_squared_adj:.5f}')
                  
        plt.text(0.05, 0.95, text_cn if self.language == 'cn' else text_en,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def compute_r_squared(x_data, y_data, popt, func):
        """计算 R² 和 调整后的 R² / Compute R² and adjusted R²"""
        y_fit = func(x_data, *popt)  # 计算预测值 
        y_mean = np.mean(y_data)  # 计算 y 的均值

        # 计算 RSS（残差平方和）
        rss = np.sum((y_data - y_fit) ** 2)

        # 计算 TSS（总平方和）
        tss = np.sum((y_data - y_mean) ** 2)

        # 计算 R²
        r_squared = 1 - (rss / tss)

        # 计算调整后的 R²
        n = len(y_data)  # 样本数
        p = len(popt)    # 拟合参数数
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        return r_squared, r_squared_adj

    @staticmethod
    def exp_rise(t, A, tau, C):
        """开启过程的指数函数模型：y = A * (1 - exp(-t/tau)) + C"""
        return A * (1 - np.exp(-t / tau)) + C

    @staticmethod
    def exp_decay(t, A, tau, C):
        """关闭过程的指数函数模型：y = A * exp(-t/tau) + C"""
        return A * np.exp(-t / tau) + C
    
    @staticmethod
    def normalize_signal(signal):
        """将信号归一化到0-1范围 / Normalize signal to 0-1 range"""
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        if signal_max - signal_min > 1e-10:  # 避免数值问题
            return (signal - signal_min) / (signal_max - signal_min)
        else:
            return np.zeros_like(signal)
