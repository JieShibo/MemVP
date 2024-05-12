from memvp.build import create_model
import torch
import argparse
from dataclasses import dataclass
from memvp.tokenizer import Tokenizer
from util.base_prompt import build_prompt
import json
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import re
from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

@dataclass
class PromptArgs:
    prompt_format = 'QCM-A'
    use_caption = True
    options = ["A", "B", "C", "D", "E"]


def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_file, data_file):
    # read result file
    results = json.load(open(result_file))
    num = len(results)
    assert num == 4241

    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

    # update data
    for index, row in res_pd.iterrows():
        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100

    scores = {
        'acc_natural':
            get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social':
            get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language':
            get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_has_text':
            get_acc_with_contion(res_pd, 'has_text', True),
        'acc_has_image':
            get_acc_with_contion(res_pd, 'has_image', True),
        'acc_no_context':
            get_acc_with_contion(res_pd, 'no_context', True),
        'acc_grade_1_6':
            get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
            get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
            "{:.2f}".format(acc_average),
    }

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1  # return random.choice(range(len(choices)))


@dataclass
class ModelArgs_7B:
    llama_model_path = './data/weights/'
    llm_model = '7B'
    max_seq_len = 512
    hidden_proj = 128
    emb = 320
    cpu_load = False
    adapter_scale = 0.1
    adapter_dim = 12
    gradient_checkpointing = False
    is_train = False
    data_root = './data/'

@dataclass
class ModelArgs_13B:
    llama_model_path = './data/weights/'
    llm_model = '13B'
    max_seq_len = 512
    hidden_proj = 128
    emb = 400
    cpu_load = False
    adapter_scale = 0.1
    adapter_dim = 12
    gradient_checkpointing = False
    is_train = False
    data_root = './data/'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model', default='7B', type=str)
    parser.add_argument('--adapter_path', default='MemVP-SQA-7B', type=str)
    args = parser.parse_args()
    bs = args.batch_size
    adapter_path = args.adapter_path
    if args.model == '7B':
        args = ModelArgs_7B()
    else:
        args = ModelArgs_13B()
    llama = create_model(args)
    adapter = torch.load(os.path.join(adapter_path, 'checkpoint-19.pth'))['model']
    sd = {}
    for k in adapter:
        sd[k.replace('module.', '')] = adapter[k]
    llama.load_state_dict(sd, False)

    tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model'))

    split = 'test'
    print('split: ', split)
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
    captions = json.load(open(os.path.join(args.data_root, 'captions.json')))["captions"]
    image_path = os.path.join(args.data_root, 'images', split)
    qids = pid_splits['%s' % (split)]
    total_items = len(qids)
    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    print('total_items: ', total_items)

    image_transforms = transforms.Compose(
        [transforms.Resize((224, 224), interpolation=Image.BICUBIC), transforms.ToTensor(),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    prompt_args = PromptArgs()

    pattern = re.compile(r'([A-Z])')

    answers = []
    preds = []

    for i in tqdm(range(total_items // bs + 1)):
        batch_qids = qids[i * bs:(i + 1) * bs]
        if len(batch_qids) == 0:
            break
        indicators = []
        prompts = []
        images = []
        for qid in batch_qids:
            prompt, _ = build_prompt(problems, qid, prompt_args)
            prompt += 'The answer is'
            answer = problems[qid]["answer"]
            if problems[qid]['image'] is not None:
                image = Image.open(os.path.join(image_path, qid, 'image.png')).convert('RGB')
                image = image_transforms(image)
                indicator = 1
            else:
                image = torch.Tensor(torch.zeros(3, 224, 224).float())
                indicator = 0
            prompts.append(prompt)
            answers.append(answer)
            images.append(image.unsqueeze(0))
            indicators.append(indicator)
        images = torch.cat(images, 0)

        results = llama.generate(
                prompts, images=images, indicators=indicators, max_gen_len=1, tokenizer=tokenizer, temperature=0.0
        )
        
        for result in results:
            pred = pattern.findall(result)

            if len(pred) >= 1:
                pred = pred[0]  # 'A', 'B', ...
            else:
                print(result)
                pred = "FAILED"
            preds.append(pred)

    # evaluations
    results = {}
    correct = 0
    for i, prediction in enumerate(preds):
        pred_idx = get_pred_idx(prediction, problems[qids[i]]["choices"],
                                prompt_args.options)  # 0, 1, ..., 4
        if pred_idx == answers[i]:
            correct += 1
        results[qids[i]] = pred_idx
    acc = correct / len(results) * 100
    print('overall accuracy: ', acc)

    with open('./preds.json', 'w') as f:
        json.dump(results, f)

    scores = get_scores('./preds.json', os.path.join(args.data_root, 'problems.json'))
    print(scores)
    import time
    with open(str(time.time()) + '.txt', 'w') as f:
        f.write(str(scores))

if __name__ == '__main__':
    main()
