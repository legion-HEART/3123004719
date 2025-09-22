# 3123004719
作业二
"""
论文查重系统 - 优化版
该程序使用高效算法计算两个文本文件之间的相似度
经过性能分析优化，处理速度提升显著
"""

import sys
import re
import math
from collections import defaultdict
import mmap
import os
import time

# 预编译正则表达式以提高性能
WORD_PATTERN = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]+')


def preprocess_text(text):
    """
    文本预处理：分词并过滤非中文字符

    参数:
        text (str): 待处理的文本

    返回:
        list: 分词后的单词列表
    """
    # 使用预编译的正则表达式进行分词
    return WORD_PATTERN.findall(text)


def compute_cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度（优化版）

    参数:
        vec1 (dict): 第一个文本的词频向量
        vec2 (dict): 第二个文本的词频向量

    返回:
        float: 两个向量的余弦相似度（0.0-1.0）
    """
    # 优化点1：只计算两个向量共有的词汇
    common_words = set(vec1.keys()) & set(vec2.keys())

    # 优化点2：提前计算向量模长
    magnitude1 = math.sqrt(sum(count ** 2 for count in vec1.values()))
    magnitude2 = math.sqrt(sum(count ** 2 for count in vec2.values()))

    # 避免除以零
    if magnitude1 * magnitude2 == 0:
        return 0.0

    # 计算点积（只计算共有词汇）
    dot_product = 0
    for word in common_words:
        dot_product += vec1[word] * vec2[word]

    # 计算余弦相似度
    return dot_product / (magnitude1 * magnitude2)


def process_large_file(file_path):
    """
    处理大文件，使用内存映射技术（优化版）

    参数:
        file_path (str): 文件路径

    返回:
        defaultdict: 词频统计字典
    """
    word_counts = defaultdict(int)

    try:
        # 使用UTF-8编码打开文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # 使用内存映射处理大文件
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                # 优化点：直接处理整个内存映射对象
                text = mmap_obj.read().decode('utf-8', errors='ignore')
                words = preprocess_text(text)

                # 优化点：批量处理单词
                for word in words:
                    word_counts[word] += 1

    except OSError as os_error:
        print(f"OS error processing file {file_path}: {os_error}")
        sys.exit(1)
    except Exception as general_error:  # pylint: disable=broad-except
        print(f"Error processing file {file_path}: {general_error}")
        sys.exit(1)

    return word_counts


def process_small_file(file_path):
    """
    处理小文件，直接读取整个文件（优化版）

    参数:
        file_path (str): 文件路径

    返回:
        defaultdict: 词频统计字典
    """
    word_counts = defaultdict(int)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            words = preprocess_text(text)
            for word in words:
                word_counts[word] += 1

    except UnicodeDecodeError:
        try:
            # 尝试使用GBK编码
            with open(file_path, 'r', encoding='gbk') as file:
                text = file.read()
                words = preprocess_text(text)
                for word in words:
                    word_counts[word] += 1
        except Exception as fallback_error:  # pylint: disable=broad-except
            print(f"Error reading file {file_path} with GBK encoding: {fallback_error}")
            sys.exit(1)
    except OSError as os_error:
        print(f"OS error reading file {file_path}: {os_error}")
        sys.exit(1)
    except Exception as general_error:  # pylint: disable=broad-except
        print(f"Error reading file {file_path}: {general_error}")
        sys.exit(1)

    return word_counts


def main():
    """
    主函数：处理命令行参数，计算并输出相似度（性能优化版）
    """
    # 检查命令行参数数量
    if len(sys.argv) != 4:
        print("Usage: python main.py <original_file> <copied_file> <output_file>")
        sys.exit(1)

    # 从命令行参数获取文件路径
    orig_path = sys.argv[1]
    copy_path = sys.argv[2]
    output_path = sys.argv[3]

    # 检查文件是否存在
    if not os.path.exists(orig_path):
        print(f"Error: Original file not found: {orig_path}")
        sys.exit(1)

    if not os.path.exists(copy_path):
        print(f"Error: Copied file not found: {copy_path}")
        sys.exit(1)

    # 开始计时
    start_time = time.time()

    # 检查文件大小
    orig_size = os.path.getsize(orig_path)
    copy_size = os.path.getsize(copy_path)

    # 根据文件大小选择处理方式
    if orig_size < 10 * 1024 * 1024 and copy_size < 10 * 1024 * 1024:  # 小于10MB
        print("Processing small files")
        orig_vector = process_small_file(orig_path)
        copy_vector = process_small_file(copy_path)
    else:
        # 处理大文件
        print(f"Processing large files: original={orig_size / (1024 * 1024):.2f}MB, "
              f"copied={copy_size / (1024 * 1024):.2f}MB")
        orig_vector = process_large_file(orig_path)
        copy_vector = process_large_file(copy_path)

    # 计算相似度
    similarity = compute_cosine_similarity(orig_vector, copy_vector)

    # 确保相似度在0-1范围内
    similarity = max(0.0, min(1.0, similarity))

    # 写入输出文件
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(f"{similarity:.2f}")
        elapsed_time = time.time() - start_time
        print(f"Similarity: {similarity:.4f}, Time: {elapsed_time:.2f}s")
    except OSError as os_error:
        print(f"OS error writing output: {os_error}")
        sys.exit(1)
    except Exception as general_error:  # pylint: disable=broad-except
        print(f"Error writing output: {general_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
