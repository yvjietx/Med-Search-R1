import re
import random

def extract_solution(solution_str):
    # 1. 移除 "Question:" 之前的内容
    if "Question:" in solution_str:
        solution_str = solution_str.split("Question:", 1)[1]
    else:
        return None

    # 2. 提取所有 <answer> 标签内容
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    
    if not matches:
        return None

    # 3. 取最后一个 <answer> 的内容，并去除首尾空格
    final_answer = matches[-1].group(1).strip().upper()  # 转为大写

    # 4. 检查是否为 A/B/C/D/E
    if final_answer in {'A', 'B', 'C', 'D', 'E'}:
        return final_answer
    else:
        return None

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    print(f"--------------------------------")
    print(f"Solution string: {solution_str}")
    print(f"Ground truth: {ground_truth['target']}\n")
    
    # 提取模型预测的答案
    answer = extract_solution(solution_str)

    
    # 如果没有提取到有效答案
    if answer is None:
        print(f"No valid answer found")
        return 0
    
    # 比较答案
    if answer == ground_truth['target']:
        print(f"Correct answer: {answer}")
        return score
    else:
        print(f"Wrong answer: {answer} | Expected: {ground_truth['target']}")
        return format_score  # 答案格式正确但答案错误