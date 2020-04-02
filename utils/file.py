import numpy as np


def split_idx(count, train_ratio=0.8):
    total_idx = [x for x in range(count)]
    train_count = int(count * train_ratio)
    train_idx = np.random.choice(count, train_count).tolist()
    test_idx = [x for x in total_idx if x not in train_idx]
    return train_idx, test_idx


def create_data(*label_path):
    """
    输入多个新格式label文件，生成操作数据文本
    """
    lines = []
    for path in label_path:
        with open(path, 'r') as f:
            lines += f.readlines()
    train_idx, test_idx = split_idx(len(lines))
    lines = np.array(lines)
    train_lines = lines[train_idx].tolist()
    test_lines = lines[test_idx].tolist()
    save_txt(train_lines, '../train.txt')
    save_txt(test_lines, '../test.txt')


def save_txt(lines, name):
    with open(name, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    create_data('../cut/I/label.txt', '../cut/II/label.txt', '../non/I/label.txt', '../non/II/label.txt')
