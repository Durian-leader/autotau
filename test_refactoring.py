"""
测试 autotau v0.3.0 架构重构

验证项：
1. AutoTauFitter 新增的 executor 参数工作正常
2. CyclesAutoTauFitter 新增的 fitter_factory 参数工作正常
3. 废弃警告正确显示
4. 向后兼容性
"""

import numpy as np
import warnings

# 测试数据生成
def generate_test_signal():
    """生成简单的测试信号（模拟 transient 响应）"""
    t = np.linspace(0, 0.2, 200)  # 200ms, 200 points

    # 模拟指数上升和下降
    period = 0.1  # 100ms period
    tau = 0.02    # 20ms time constant

    signal = np.zeros_like(t)
    for i, time in enumerate(t):
        phase = (time % period) / period
        if phase < 0.5:
            # On phase: exponential rise
            signal[i] = 1 - np.exp(-phase * period / tau)
        else:
            # Off phase: exponential decay
            signal[i] = np.exp(-(phase - 0.5) * period / tau)

    # Add some noise
    signal += np.random.normal(0, 0.01, len(t))

    return t, signal


def test_1_autotau_fitter_serial():
    """测试 1: AutoTauFitter 串行模式（默认）"""
    print("\n" + "="*70)
    print("测试 1: AutoTauFitter 串行模式（executor=None，默认）")
    print("="*70)

    from autotau import AutoTauFitter

    t, signal = generate_test_signal()
    period = 0.1
    sample_step = t[1] - t[0]

    # 测试新 API：executor=None（串行，默认）
    fitter = AutoTauFitter(
        t, signal,
        sample_step=sample_step,
        period=period,
        window_scalar_min=0.1,
        window_scalar_max=0.4,
        window_points_step=10,
        executor=None  # ✨ 显式指定串行
    )

    tau_on_popt, tau_on_r2, tau_off_popt, tau_off_r2 = fitter.fit_tau_on_and_off()

    print(f"✓ 串行拟合成功")
    print(f"  Tau On:  τ={tau_on_popt[1]:.4f}s, R²={tau_on_r2:.4f}")
    print(f"  Tau Off: τ={tau_off_popt[1]:.4f}s, R²={tau_off_r2:.4f}")
    print("✓ 测试通过")

    return True


def test_2_cycles_auto_tau_fitter_serial():
    """测试 2: CyclesAutoTauFitter 串行模式（默认）"""
    print("\n" + "="*70)
    print("测试 2: CyclesAutoTauFitter 串行模式（fitter_factory=None，默认）")
    print("="*70)

    from autotau import CyclesAutoTauFitter

    # 生成多周期信号
    t = np.linspace(0, 0.5, 500)  # 500ms, 5 cycles
    period = 0.1
    tau = 0.02

    signal = np.zeros_like(t)
    for i, time in enumerate(t):
        phase = (time % period) / period
        if phase < 0.5:
            signal[i] = 1 - np.exp(-phase * period / tau)
        else:
            signal[i] = np.exp(-(phase - 0.5) * period / tau)
    signal += np.random.normal(0, 0.01, len(t))

    # 测试新 API：fitter_factory=None（串行，默认）
    fitter = CyclesAutoTauFitter(
        t, signal,
        period=period,
        sample_rate=1000,
        fitter_factory=None,  # ✨ 显式指定使用默认工厂（串行）
        window_scalar_min=0.1,
        window_scalar_max=0.4,
        window_points_step=10
    )

    results = fitter.fit_all_cycles(r_squared_threshold=0.95)

    print(f"✓ 串行拟合成功")
    print(f"  检测到 {len(results)} 个周期")

    # 显示前3个周期的结果
    for i, result in enumerate(results[:3]):
        print(f"  Cycle {i}: τ_on={result.get('tau_on', 'N/A'):.4f}s, "
              f"τ_off={result.get('tau_off', 'N/A'):.4f}s")

    print("✓ 测试通过")

    return True


def test_3_deprecation_warnings():
    """测试 3: 废弃警告"""
    print("\n" + "="*70)
    print("测试 3: 废弃警告显示")
    print("="*70)

    # 捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # 确保所有警告都被捕获

        # 测试 ParallelAutoTauFitter 废弃警告
        from autotau import ParallelAutoTauFitter
        t, signal = generate_test_signal()
        period = 0.1
        sample_step = t[1] - t[0]

        fitter = ParallelAutoTauFitter(
            t, signal,
            sample_step=sample_step,
            period=period,
            max_workers=2
        )

        # 检查警告
        assert len(w) >= 1, "未捕获到废弃警告"
        assert issubclass(w[0].category, DeprecationWarning), "警告类型不正确"
        assert "ParallelAutoTauFitter 已被废弃" in str(w[0].message), "警告消息不正确"

        print("✓ ParallelAutoTauFitter 废弃警告正确显示")
        print(f"  警告类型: {w[0].category.__name__}")
        print(f"  警告消息: {str(w[0].message)[:80]}...")

    # 测试 ParallelCyclesAutoTauFitter 废弃警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from autotau import ParallelCyclesAutoTauFitter

        fitter = ParallelCyclesAutoTauFitter(
            t, signal,
            period=period,
            sample_rate=1000,
            max_workers=2
        )

        assert len(w) >= 1, "未捕获到废弃警告"
        assert issubclass(w[0].category, DeprecationWarning), "警告类型不正确"
        assert "ParallelCyclesAutoTauFitter 已被废弃" in str(w[0].message), "警告消息不正确"

        print("✓ ParallelCyclesAutoTauFitter 废弃警告正确显示")
        print(f"  警告类型: {w[0].category.__name__}")
        print(f"  警告消息: {str(w[0].message)[:80]}...")

    print("✓ 测试通过")

    return True


def test_4_parallel_executor():
    """测试 4: 使用 executor 的并行模式（可选）"""
    print("\n" + "="*70)
    print("测试 4: AutoTauFitter 并行模式（使用 ProcessPoolExecutor）")
    print("="*70)

    from autotau import AutoTauFitter
    from concurrent.futures import ProcessPoolExecutor

    t, signal = generate_test_signal()
    period = 0.1
    sample_step = t[1] - t[0]

    # 测试新 API：显式传入 executor
    with ProcessPoolExecutor(max_workers=2) as executor:
        fitter = AutoTauFitter(
            t, signal,
            sample_step=sample_step,
            period=period,
            window_scalar_min=0.1,
            window_scalar_max=0.4,
            window_points_step=10,
            executor=executor  # ✨ 显式传入并行执行器
        )

        tau_on_popt, tau_on_r2, tau_off_popt, tau_off_r2 = fitter.fit_tau_on_and_off()

    print(f"✓ 并行拟合成功")
    print(f"  Tau On:  τ={tau_on_popt[1]:.4f}s, R²={tau_on_r2:.4f}")
    print(f"  Tau Off: τ={tau_off_popt[1]:.4f}s, R²={tau_off_r2:.4f}")
    print("✓ 测试通过")

    return True


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" AutoTau v0.3.0 架构重构测试套件")
    print("="*70)

    tests = [
        ("AutoTauFitter 串行模式", test_1_autotau_fitter_serial),
        ("CyclesAutoTauFitter 串行模式", test_2_cycles_auto_tau_fitter_serial),
        ("废弃警告", test_3_deprecation_warnings),
        ("AutoTauFitter 并行模式", test_4_parallel_executor),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试失败: {name}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print("\n" + "="*70)
    print(" 测试总结")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有测试通过！架构重构成功！")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
