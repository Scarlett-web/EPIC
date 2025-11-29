"""
Travel 数据集 EPIC 一键运行脚本
作用：自动完成数据预处理 -> 生成合成数据 -> 评估性能的完整流程
"""
import os
import sys
import subprocess
import time

def run_command(cmd, cwd=None, description=""):
    """运行命令并显示输出"""
    print("\n" + "="*60)
    print(f"🚀 {description}")
    print("="*60)
    print(f"命令: {cmd}")
    print("-"*60)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {description} 完成")
            return True
        else:
            print(f"❌ {description} 失败")
            return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        🚀 Travel 数据集 EPIC 完整流程自动化脚本          ║
    ║                                                          ║
    ║  步骤 1: 数据预处理                                      ║
    ║  步骤 2: 生成合成数据 (EPIC 方法)                        ║
    ║  步骤 3: 评估合成数据质量                                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    syn_gen_dir = os.path.join(project_root, "codes", "SyntheticDataGeneration")
    downstream_dir = os.path.join(project_root, "codes", "DownstreamTasks")
    
    # 询问用户要执行哪些步骤
    print("\n请选择要执行的步骤:")
    print("  1 - 只执行数据预处理")
    print("  2 - 只执行数据生成")
    print("  3 - 只执行性能评估")
    print("  4 - 执行完整流程 (1+2+3)")
    print("  5 - 跳过生成，只评估已有数据 (1+3)")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    start_time = time.time()
    
    # 步骤 1: 数据预处理
    if choice in ['1', '4', '5']:
        success = run_command(
            "python preprocess_travel_data.py",
            cwd=syn_gen_dir,
            description="步骤 1/3: 数据预处理"
        )
        if not success and choice == '4':
            print("\n❌ 数据预处理失败，终止流程")
            return
    
    # 步骤 2: 生成合成数据
    if choice in ['2', '4']:
        print("\n⚠️ 注意: 数据生成可能需要 15-30 分钟")
        confirm = input("是否继续? (y/n): ").strip().lower()
        if confirm == 'y':
            success = run_command(
                "python generate_samples_Travel.py",
                cwd=syn_gen_dir,
                description="步骤 2/3: 生成合成数据 (EPIC)"
            )
            if not success and choice == '4':
                print("\n❌ 数据生成失败，终止流程")
                return
        else:
            print("⏭️ 跳过数据生成步骤")
    
    # 步骤 3: 评估性能
    if choice in ['3', '4', '5']:
        success = run_command(
            "python Classification_Travel.py",
            cwd=downstream_dir,
            description="步骤 3/3: 评估合成数据质量"
        )
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "="*60)
    print(f"✅ 流程完成！总耗时: {minutes} 分 {seconds} 秒")
    print("="*60)
    
    # 显示输出文件位置
    print("\n📁 输出文件位置:")
    print(f"  - 预处理数据: {os.path.join(project_root, 'data', 'realdata', 'Travel')}")
    print(f"  - 合成数据: {os.path.join(project_root, 'data', 'syndata', 'Travel_DeepSeek_EPIC')}")
    print(f"  - 评估结果: {os.path.join(project_root, 'results', 'Travel_EPIC_results.csv')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

