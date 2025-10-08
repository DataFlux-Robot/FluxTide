# FluxTide

<div align="center">

**通用人形机器人控制器框架 | Universal Humanoid Robot Controller Framework**

*基于并行环境与采样MPC的重构实现 | Refactored Implementation Based on Parallel Environments and Sample-based MPC*

</div>

---

## 项目简介 | Introduction

FluxTide 是一个通用的人形机器人控制器框架,专注于并行环境模拟与采样式模型预测控制(Sample-based MPC)的结合。本项目基于 [dial-mpc](https://github.com/LeCAR-Lab/dial-mpc) 进行重构,旨在提供更加通用、简洁、易用的控制器实现。

FluxTide is a universal humanoid robot controller framework that combines parallel environment simulation with sample-based Model Predictive Control (MPC). This project is a refactored implementation based on [dial-mpc](https://github.com/LeCAR-Lab/dial-mpc), aiming to provide a more general, concise, and user-friendly controller solution.

## 核心特性 | Key Features

🔧 **解耦设计 | Decoupled Design** - 将 dial-mpc 核心算法与仿真器(如 Brax)解耦,实现更高的通用性 | Separates dial-mpc core algorithms from simulators (e.g., Brax) for enhanced universality

📦 **简化架构 | Simplified Architecture** - 重构项目结构,将相关功能集中到少量 Python 程序中,降低使用门槛 | Refactored project structure consolidates functionality into fewer Python modules

🎨 **可视化界面 | Visual Interface** - 提供可视化配置界面,快速选择和配置控制任务 | Provides visualization tools for quick task selection and configuration

⚡ **并行加速 | Parallel Acceleration** - 充分利用并行环境,提升 MPC 采样效率 | Leverages parallel environments to boost MPC sampling efficiency

## 核心改进 | Core Improvements

相比原始 dial-mpc 项目 | Compared to the original dial-mpc project:

1. **通用性增强 | Enhanced Generality**: 核心控制代码与特定仿真器解耦,可适配多种仿真环境 | Core control code decoupled from specific simulators, adaptable to various simulation environments

2. **结构简化 | Structural Simplification**: 精简项目架构,减少代码冗余,提高可维护性 | Streamlined architecture with reduced code redundancy and improved maintainability

3. **交互优化 | Interaction Optimization**: 新增可视化配置界面,简化任务设置流程 | New visualization interface simplifies task configuration workflow

## 快速开始 | Quick Start

### 安装依赖 | Install Dependencies

```bash
git clone https://github.com/DataFlux-Robot/FluxTide.git
cd FluxTide
pip install -r requirements.txt
运行控制器 | Run Controller
bash
复制
python alone_python_dial_mpc_main.py
技术背景 | Technical Background
DIAL-MPC (Diffusion-Inspired Annealing for Legged MPC) 是一种针对腿足机器人全阶扭矩级控制的采样式 MPC 框架。FluxTide 在此基础上进行重构,使其能够更好地应用于人形机器人控制场景。

DIAL-MPC (Diffusion-Inspired Annealing for Legged MPC) is a sampling-based MPC framework for full-order torque-level control of legged robots. FluxTide refactors this approach to better suit humanoid robot control scenarios.

特别感谢 | Special Thanks
感谢 dial-mpc 项目团队提供的优秀基础工作。

Thanks to the dial-mpc project team for their excellent foundational work.

贡献指南 | Contributing
欢迎提交 Issue 和 Pull Request! | Issues and Pull Requests are welcome!
