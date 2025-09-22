# test_main.py
import pytest
import os
import tempfile
import shutil
from main import preprocess_text, compute_cosine_similarity, process_small_file


# 测试预处理函数
def test_preprocess_text():
    text = "今天是星期天，天气晴，今天晚上我要去看电影。"
    result = preprocess_text(text)
    expected = ['今天是星期天', '天气晴', '今天晚上我要去看电影']
    assert result == expected

    # 测试英文和数字
    text = "Hello world 123 test. 测试"
    result = preprocess_text(text)
    expected = ['Hello', 'world', '123', 'test', '测试']
    assert result == expected


# 测试余弦相似度计算
def test_compute_cosine_similarity():
    # 完全相同向量
    vec1 = {'a': 1, 'b': 2, 'c': 3}
    vec2 = {'a': 1, 'b': 2, 'c': 3}
    similarity = compute_cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.001

    # 完全不同向量
    vec1 = {'a': 1, 'b': 2}
    vec2 = {'c': 3, 'd': 4}
    similarity = compute_cosine_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 0.001

    # 部分相似向量
    vec1 = {'a': 1, 'b': 2, 'c': 3}
    vec2 = {'a': 1, 'b': 2, 'd': 4}
    similarity = compute_cosine_similarity(vec1, vec2)
    assert 0 < similarity < 1


# 测试小文件处理
def test_process_small_file():
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write("今天是星期天，天气晴。")
        temp_file = f.name

    try:
        result = process_small_file(temp_file)
        expected = {'今天是星期天': 1, '天气晴': 1}
        assert result == expected
    finally:
        os.unlink(temp_file)


# 测试空文件
def test_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write("")
        temp_file = f.name

    try:
        result = process_small_file(temp_file)
        assert result == {}
    finally:
        os.unlink(temp_file)


# 测试文件不存在的情况
def test_file_not_found():
    with pytest.raises(SystemExit):
        process_small_file("nonexistent_file.txt")


# 测试主函数
def test_main_integration():
    # 创建测试文件
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f1:
        f1.write("今天是星期天，天气晴，今天晚上我要去看电影。")
        orig_file = f1.name

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f2:
        f2.write("今天是周天，天气晴朗，我晚上要去看电影。")
        copied_file = f2.name

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f3:
        result_file = f3.name

    try:
        # 模拟命令行参数
        import sys
        original_argv = sys.argv
        sys.argv = ['main.py', orig_file, copied_file, result_file]

        # 导入并运行主函数
        from main import main
        main()

        # 检查结果文件
        with open(result_file, 'r') as f:
            result = float(f.read().strip())
            assert 0 <= result <= 1

    finally:
        # 恢复原始参数
        sys.argv = original_argv
        # 清理临时文件
        os.unlink(orig_file)
        os.unlink(copied_file)
        os.unlink(result_file)


# 测试不同编码
def test_different_encoding():
    # 创建GBK编码文件
    with tempfile.NamedTemporaryFile(mode='w', encoding='gbk', suffix='.txt', delete=False) as f:
        f.write("今天是星期天，天气晴。")
        gbk_file = f.name

    try:
        result = process_small_file(gbk_file)
        expected = {'今天是星期天': 1, '天气晴': 1}
        assert result == expected
    finally:
        os.unlink(gbk_file)


# 测试大文件处理
def test_large_file_processing(monkeypatch):
    # 创建一个临时目录和文件
    temp_dir = tempfile.mkdtemp()
    large_file = os.path.join(temp_dir, "large.txt")

    try:
        # 创建一个大文件（超过10MB阈值）
        with open(large_file, 'w', encoding='utf-8') as f:
            # 写入足够多的内容使其超过10MB
            content = "今天是星期天，天气晴，今天晚上我要去看电影。" * 100000
            f.write(content)

        # 模拟文件大小检查，使其认为文件很大
        original_getsize = os.path.getsize

        def mock_getsize(path):
            if path == large_file:
                return 11 * 1024 * 1024  # 11MB
            return original_getsize(path)

        monkeypatch.setattr(os.path, 'getsize', mock_getsize)

        # 测试处理大文件
        from main import process_large_file
        result = process_large_file(large_file)
        assert len(result) > 0

    finally:
        # 清理
        shutil.rmtree(temp_dir)


# 测试边界情况 - 完全相同的文件
def test_identical_files():
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f1:
        f1.write("测试文本内容")
        file1 = f1.name

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f2:
        f2.write("测试文本内容")
        file2 = f2.name

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f3:
        result_file = f3.name

    try:
        import sys
        original_argv = sys.argv
        sys.argv = ['main.py', file1, file2, result_file]

        from main import main
        main()

        with open(result_file, 'r') as f:
            result = float(f.read().strip())
            assert abs(result - 1.0) < 0.001  # 应该非常接近1

    finally:
        sys.argv = original_argv
        os.unlink(file1)
        os.unlink(file2)
        os.unlink(result_file)


# 测试边界情况 - 完全不同的文件
def test_completely_different_files():
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f1:
        f1.write("这是第一个文本内容")
        file1 = f1.name

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f2:
        f2.write("这是完全不同的第二个文本")
        file2 = f2.name

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f3:
        result_file = f3.name

    try:
        import sys
        original_argv = sys.argv
        sys.argv = ['main.py', file1, file2, result_file]

        from main import main
        main()

        with open(result_file, 'r') as f:
            result = float(f.read().strip())
            assert abs(result - 0.0) < 0.001  # 应该非常接近0

    finally:
        sys.argv = original_argv
        os.unlink(file1)
        os.unlink(file2)
        os.unlink(result_file)
