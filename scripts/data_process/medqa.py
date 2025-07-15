# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MedQA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question'].strip()
    if not question.endswith("?"):
        question += "?"

    if template_type == 'base':
        # 拼接选项文本
        options = dp.get("options", {})
        if options:
            option_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
            question += f"\n\n{option_text}"

        # 构造提示文本
        prefix = f"""Answer the given medical question. \
You should first have a medical reasoning process in mind and then provide the answer. \
Show your medical reasoning in <think> </think> tags and return the final answer (A-E) in <answer> </answer> tags, for example <answer> E </answer>. \
Medical Question: {question}\n"""
    else:
        raise NotImplementedError

    return prefix

# 主程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/medqa')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'medqa'

    # 加载数据集
    dataset = datasets.load_dataset(
        'json',
        data_files={
            'train': './data/medqa/train.jsonl',
            'validation': './data/medqa/dev.jsonl',
            'test': './data/medqa/test.jsonl'
        }
    )
    
    train_dataset = dataset['train']
    test_dataset = dataset['test'] if 'test' in dataset else dataset['validation']

    print(f"加载数据集完成:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 构造映射函数
    def make_map_fn(split):
        def process_fn(example, idx):
            example['question'] = example['question'].strip()

            # 构造 prompt（含选项 + 思维链格式）
            question = make_prefix(example, template_type=args.template_type)

            # ground truth 使用 answer_idx（如 "E"）
            solution = {
                "target": example['answer_idx'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"\n数据处理完成！")
    print(f"数据已保存到: {local_dir}")
    if hdfs_dir:
        print(f"数据已同步到HDFS: {hdfs_dir}")
