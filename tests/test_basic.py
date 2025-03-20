import unittest
import numpy as np
from autotau import TauFitter, AutoTauFitter

class TestBasic(unittest.TestCase):
    def test_import(self):
        """测试基本的包导入"""
        from autotau import (
            TauFitter, 
            AutoTauFitter, 
            CyclesAutoTauFitter,
            ParallelAutoTauFitter,
            ParallelCyclesAutoTauFitter
        )
        self.assertTrue(True)
    
    def test_tau_fitter_basic(self):
        """测试TauFitter基本功能"""
        # 创建一些测试数据
        time_data = np.linspace(0, 1, 100)
        # 指数上升数据 y = 2*(1-exp(-t/0.2))+0.1
        signal_data = 2 * (1 - np.exp(-time_data / 0.2)) + 0.1
        
        # 添加一些噪声
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, size=len(time_data))
        signal_data_noisy = signal_data + noise
        
        # 创建TauFitter对象，直接在初始化时设置时间窗口
        fitter = TauFitter(time_data, signal_data_noisy, t_on_idx=[0, 0.5])
        
        # 拟合上升过程
        fitter.fit_tau_on()
        
        # 检查拟合结果是否接近真实值
        tau_on = fitter.get_tau_on()
        self.assertAlmostEqual(tau_on, 0.2, delta=0.05)

if __name__ == '__main__':
    unittest.main() 