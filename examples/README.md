# AutoTau 示例

此目录包含AutoTau库的使用示例。

## 基本示例

[basic_examples.py](basic_examples.py) 包含以下示例：

- TauFitter 基本使用示例
- AutoTauFitter 自动寻找最佳拟合窗口示例
- CyclesAutoTauFitter 多周期数据处理示例
- 并行处理性能比较

## 运行示例

```bash
# 确保当前目录是包的根目录
cd /path/to/autotau

# 运行基本示例
python examples/basic_examples.py
```

## 示例数据

示例程序使用包中的示例数据文件`autotau/transient.csv`，该文件包含一个模拟的电流上升和下降信号。

## 注意事项

- 运行示例前，请确保已按照主README文件中的说明安装了AutoTau包。
- 示例中的可视化结果会通过matplotlib显示，请确保您的环境支持图形显示。 