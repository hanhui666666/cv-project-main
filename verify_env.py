"""环境验证脚本 - 第10组工业缺陷检测项目

运行此脚本验证环境配置是否正确。
"""
import sys
import subprocess


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("  [OK] Python版本符合要求 (>=3.10)")
        return True
    else:
        print("  [FAIL] Python版本过低，需要 >=3.10")
        return False


def check_package(package_name, import_name=None, min_version=None):
    """检查Python包是否安装"""
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  {package_name}: {version}", end="")
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) >= pkg_version.parse(min_version):
                print(f" [OK] (>= {min_version})")
                return True
            else:
                print(f" [FAIL] (需要 >= {min_version})")
                return False
        else:
            print(" [OK]")
            return True
    except ImportError:
        print(f"  {package_name}: 未安装 [FAIL]")
        return False


def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA可用: {torch.cuda.is_available()}")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  GPU {i}显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print("  [OK] CUDA环境正常")
            return True
        else:
            print("  CUDA不可用，将使用CPU运行")
            print("  [WARN] 建议配置GPU以加速训练")
            return True  # CPU运行也是可行的
    except Exception as e:
        print(f"  [FAIL] 检查CUDA时出错: {e}")
        return False


def check_yolov8():
    """检查YOLOv8是否可导入"""
    try:
        from ultralytics import YOLO
        print("  YOLOv8导入成功 [OK]")
        return True
    except ImportError:
        print("  YOLOv8导入失败 [FAIL]")
        return False


def main():
    print("=" * 50)
    print("第10组 - 工业缺陷检测项目 环境验证")
    print("=" * 50)
    print()

    results = []

    # 1. Python版本
    print("[1/6] 检查Python版本...")
    results.append(check_python_version())
    print()

    # 2. 核心包
    print("[2/6] 检查核心依赖包...")
    results.append(check_package("torch", "torch", "2.7.0"))
    results.append(check_package("torchvision", "torchvision", "0.24.0"))
    results.append(check_package("ultralytics", "ultralytics", "8.3.0"))
    results.append(check_package("albumentations", "albumentations", "1.4.0"))
    results.append(check_package("opencv-python", "cv2"))
    results.append(check_package("matplotlib", "matplotlib", "3.9.0"))
    results.append(check_package("numpy", "numpy", "1.26.0"))
    results.append(check_package("pandas", "pandas", "2.2.0"))
    results.append(check_package("pytorch-grad-cam", "gradcam"))
    print()

    # 3. CUDA
    print("[3/6] 检查CUDA环境...")
    results.append(check_cuda())
    print()

    # 4. YOLOv8
    print("[4/6] 检查YOLOv8...")
    results.append(check_yolov8())
    print()

    # 5. 数据集路径
    print("[5/6] 检查数据集路径...")
    import os
    data_paths = ["data/aluminum", "data/fabric"]
    for path in data_paths:
        if os.path.exists(path):
            print(f"  {path}: 存在 [OK]")
        else:
            print(f"  {path}: 不存在 [INFO] (数据集需自行下载)")
    print()

    # 6. 项目结构
    print("[6/6] 检查项目结构...")
    required_dirs = ["models", "configs", "scripts", "utils", "experiments", "docs"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  {dir_name}/: 存在 [OK]")
        else:
            print(f"  {dir_name}/: 不存在 [WARN]")
    print()

    # 总结
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"验证结果: {passed}/{total} 项通过")
    if passed == total:
        print("[SUCCESS] 环境配置完成，可以开始项目！")
    else:
        print("[WARN] 部分检查未通过，请根据提示修复")
    print("=" * 50)


if __name__ == "__main__":
    main()
