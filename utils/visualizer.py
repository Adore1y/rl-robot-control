import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def plot_training_results(log_dir, save_dir=None, metrics=None, window_size=10):
    """
    绘制训练结果
    
    Args:
        log_dir: TensorBoard日志目录
        save_dir: 保存图像的目录
        metrics: 需要绘制的指标列表
        window_size: 滑动平均窗口大小
    """
    # 设置默认指标
    if metrics is None:
        metrics = ['episode/reward', 'loss/total', 'loss/policy', 'loss/value', 'eval/mean_reward']
    
    # 创建保存目录
    if save_dir is None:
        save_dir = os.path.join(log_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置样式
    sns.set(style="darkgrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    
    # 加载TensorBoard事件文件
    try:
        import tensorflow as tf
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # 找到最新的事件文件
        event_paths = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_paths.append(os.path.join(root, file))
        
        if not event_paths:
            raise ValueError(f"在{log_dir}中未找到TensorBoard事件文件")
        
        # 按修改时间排序，取最新的
        event_paths.sort(key=os.path.getmtime, reverse=True)
        event_path = event_paths[0]
        
        # 加载事件文件
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        
        # 获取可用的标量指标
        available_metrics = event_acc.Tags()['scalars']
        print(f"可用指标: {available_metrics}")
        
        # 为每个指标创建图表
        for metric in metrics:
            if metric not in available_metrics:
                print(f"指标 {metric} 不可用，跳过")
                continue
            
            # 获取指标数据
            events = event_acc.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            # 计算滑动平均
            if len(values) > window_size:
                values_smooth = pd.Series(values).rolling(window=window_size).mean().iloc[window_size-1:].values
                steps_smooth = steps[window_size-1:]
            else:
                values_smooth = values
                steps_smooth = steps
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 绘制原始数据和滑动平均
            plt.plot(steps, values, alpha=0.3, label='Raw')
            plt.plot(steps_smooth, values_smooth, label=f'Smoothed (window={window_size})')
            
            # 设置标题和标签
            plt.title(f'{metric} over Training Steps')
            plt.xlabel('Training Steps')
            plt.ylabel(metric)
            plt.legend()
            
            # 设置整数步数
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # 保存图表
            metric_name = metric.replace('/', '_')
            plt.savefig(os.path.join(save_dir, f'{metric_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存 {metric} 图表")
        
        # 创建多指标对比图
        if len(metrics) > 1:
            plt.figure(figsize=(12, 8))
            
            for metric in metrics:
                if metric not in available_metrics:
                    continue
                
                # 获取并规范化数据
                events = event_acc.Scalars(metric)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                # 规范化到[0,1]范围
                if len(values) > 0:
                    min_val = min(values)
                    max_val = max(values)
                    if max_val > min_val:
                        values_norm = [(v - min_val) / (max_val - min_val) for v in values]
                    else:
                        values_norm = [0.5 for _ in values]
                    
                    # 计算滑动平均
                    if len(values_norm) > window_size:
                        values_smooth = pd.Series(values_norm).rolling(window=window_size).mean().iloc[window_size-1:].values
                        steps_smooth = steps[window_size-1:]
                    else:
                        values_smooth = values_norm
                        steps_smooth = steps
                    
                    # 绘制规范化后的曲线
                    plt.plot(steps_smooth, values_smooth, label=metric)
            
            # 设置标题和标签
            plt.title('Normalized Metrics Comparison')
            plt.xlabel('Training Steps')
            plt.ylabel('Normalized Value')
            plt.legend()
            
            # 保存图表
            plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("已保存多指标对比图")
    
    except ImportError:
        print("无法导入TensorFlow或TensorBoard，请确保已安装相关依赖")
    except Exception as e:
        print(f"绘图过程中出错: {str(e)}")
    
    print(f"所有图表已保存到 {save_dir}")


def plot_evaluation_results(eval_results, save_path=None):
    """
    绘制评估结果
    
    Args:
        eval_results: 评估结果字典
        save_path: 保存图像的路径
    """
    # 设置样式
    sns.set(style="darkgrid")
    
    # 提取数据
    episodes = list(range(1, len(eval_results['mean_reward']) + 1))
    mean_rewards = eval_results['mean_reward']
    std_rewards = eval_results['std_reward']
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制平均奖励和标准差
    plt.plot(episodes, mean_rewards, marker='o', label='Mean Reward')
    plt.fill_between(
        episodes,
        [m - s for m, s in zip(mean_rewards, std_rewards)],
        [m + s for m, s in zip(mean_rewards, std_rewards)],
        alpha=0.2
    )
    
    # 设置标题和标签
    plt.title('Evaluation Results')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Mean Reward')
    plt.legend()
    
    # 设置整数步数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"评估结果图表已保存到 {save_path}")
    else:
        plt.show() 