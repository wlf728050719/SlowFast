import json
import argparse
from pathlib import Path


def generate_action_json(root_dir, output_file):
    """
    生成动作分类的JSON文件

    Args:
        root_dir (str): 要扫描的根目录
        output_file (str): 输出的JSON文件路径
    """
    root_path = Path(root_dir).absolute()

    if not root_path.exists():
        raise ValueError(f"目录不存在: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"路径不是目录: {root_path}")

    actions = []

    # 获取所有直接子目录（不递归）
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    subdirs.sort()  # 按字母顺序排序

    for label, subdir in enumerate(subdirs):
        actions.append({
            "label": label,
            "name": subdir.name,
            "video_folder": str(subdir)
        })

    # 构建完整JSON结构
    result = {
        "actions": actions
    }

    # 写入文件
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"成功生成JSON文件: {output_file}")
    print(f"共找到 {len(actions)} 个动作类别")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成动作分类JSON文件')
    parser.add_argument('--root_dir',default=r'/data/wlf_workspace/UCF-101',help='包含动作类别子目录的根目录')
    parser.add_argument('-o', '--output', default='../config.json',
                        help='输出JSON文件路径')

    args = parser.parse_args()

    try:
        generate_action_json(args.root_dir, args.output)
    except Exception as e:
        print(f"错误: {e}")
        exit(1)
