import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime

def load_parquet_data(file_path):
    """加载处理好的parquet格式数据"""
    df = pd.read_parquet(file_path)
    return df

def evaluate_predictions(predictions, targets):
    """评估预测结果"""
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
    
    accuracy = correct / total
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def setup_logger(output_dir, model_name):
    """设置日志记录器"""
    # 创建logger
    logger = logging.getLogger('medqa_inference')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    model_short_name = get_model_short_name(model_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'inference_{model_short_name}_{timestamp}.log')
    
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def extract_answer(response, prompt, logger):
    """从模型响应中提取答案"""
    try:
        logger.debug(f"正在处理响应文本: '{response}'")
        
        # 分离模型的实际输出
        if not response.startswith(prompt):
            logger.warning("无法正确分离模型输出")
            return None
            
        model_output = response[len(prompt):].strip()
        logger.debug(f"模型实际输出: '{model_output}'")
        
        # 在模型输出中查找答案标签
        start_tag = '<answer>'
        end_tag = '</answer>'
        
        start_pos = model_output.find(start_tag)
        if start_pos == -1:
            logger.debug("模型输出中未找到 <answer> 标签")
            return None
            
        start = start_pos + len(start_tag)
        end = model_output.find(end_tag, start)
        
        if end == -1:
            logger.debug("模型输出中未找到 </answer> 标签")
            return None
            
        # 提取标签中的内容
        answer = model_output[start:end].strip()
        logger.debug(f"提取到的原始答案: '{answer}'")
        
        # 校验答案是否为 A-E
        if answer and answer.upper() in ['A', 'B', 'C', 'D', 'E']:
            return answer.upper()
        else:
            logger.debug(f"检测到无效答案格式：'{answer}'")
            return None
            
    except Exception as e:
        logger.error(f"提取答案时出错：{str(e)}")
        return None

def get_model_short_name(model_name):
    """从模型路径中提取简短名称，只保留最后一部分"""
    return model_name.split('/')[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B',
                      help='Hugging Face model name or local path')
    parser.add_argument('--data_path', type=str, default='./data/medqa/test.parquet',
                      help='Path to the processed test data')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                      help='Maximum number of tokens to generate')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(args.output_dir, args.model_name)

    # 获取模型简短名称用于文件命名
    model_short_name = get_model_short_name(args.model_name)

    # 加载模型和分词器
    logger.info(f"正在加载模型 {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 加载测试数据
    logger.info(f"正在加载测试数据: {args.data_path}")
    test_data = load_parquet_data(args.data_path)
    total_samples = len(test_data)
    logger.info(f"共加载 {total_samples} 个测试样本")
    
    # 准备结果存储
    results = []
    predictions = []
    
    # 进行推理
    logger.info("开始推理...")
    invalid_predictions = 0  # 记录无效预测的数量
    
    # 使用tqdm显示进度条
    for idx, row in enumerate(tqdm(test_data.iterrows(), total=total_samples, desc="处理样本")):
        _, row = row  # 解包row
        prompt = row['prompt'][0]['content']
        target = row['reward_model']['ground_truth']['target']
        
        # 对输入进行编码
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=1,
                do_sample=False
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案
        predicted_answer = extract_answer(response, prompt, logger)
        
        if predicted_answer is None:
            invalid_predictions += 1
            logger.debug(f"样本 {idx + 1} 未能提取有效答案")
        else:
            logger.debug(f"样本 {idx + 1} - 预测: {predicted_answer}, 目标: {target}")
        
        # 保存结果
        result = {
            'index': idx,
            'prompt': prompt,
            'target': target,
            'prediction': predicted_answer,
            'full_response': response,
            'correct': predicted_answer == target if predicted_answer else False
        }
        results.append(result)
        if predicted_answer:
            predictions.append(predicted_answer)
        
        # 每处理100个样本保存一次结果
        if (idx + 1) % 100 == 0:
            logger.info(f"已处理 {idx + 1}/{total_samples} 个样本")
            # 保存中间结果
            pd.DataFrame(results).to_json(
                os.path.join(args.output_dir, f'medqa_inference_results_{model_short_name}.json'),
                orient='records',
                lines=True
            )
    
    # 计算并保存最终结果
    if len(predictions) > 0:
        eval_results = evaluate_predictions(predictions, [r['target'] for r in results if r['prediction']])
        logger.info("\n评估结果:")
        logger.info(f"准确率: {eval_results['accuracy']:.4f}")
        logger.info(f"正确数: {eval_results['correct']}/{eval_results['total']}")
        logger.info(f"无效预测数: {invalid_predictions}")
        logger.info(f"总样本数: {total_samples}")
    else:
        logger.warning("没有有效的预测结果！")
    
    # 保存详细结果
    pd.DataFrame(results).to_json(
        os.path.join(args.output_dir, f'medqa_inference_results_{model_short_name}.json'),
        orient='records',
        lines=True
    )
    
    # 保存评估指标
    if len(predictions) > 0:
        eval_results.update({
            'invalid_predictions': invalid_predictions,
            'total_samples': total_samples
        })
        with open(os.path.join(args.output_dir, f'medqa_metrics_{model_short_name}.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
    
    logger.info("评估完成！")

if __name__ == '__main__':
    main() 