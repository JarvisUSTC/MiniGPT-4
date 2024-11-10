from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config
import torch
from tqdm import tqdm
import json
from PIL import Image
from transformers import StoppingCriteriaList
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
import os

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='mmvet', help="dataset to evaluate")
parser.add_argument("--clean", action='store_true', help="clean image.")
parser.add_argument("--output_file", type=str, default='./result.jsonl',
                    help="Output file.")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_Vicuna0.copy()
# conv_temp.system = ""
model.eval()

##  TODO: expose interface.
mmvet_path = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/llava-mm-vet.jsonl"
if args.clean:
    image_dir = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/images/"
else:
    image_dir = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/noisy_images/"
datasets = []
with open(mmvet_path, "r") as f:
    for line in f:
        datasets.append(json.loads(line))

datasets = datasets[0:]
stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda') for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

outputs = {}
with torch.no_grad():
    for i, data in tqdm(enumerate(datasets), total=len(datasets), desc='Inference'):
        print(f" ----- {i} ----")
        print(" -- prompt: ---")
        question_id = data["question_id"]
        image_path = image_dir + data["image"]
        text_prompt = data["text"]
        # print(text_prompt)
        image = Image.open(image_path).convert('RGB')
        image = vis_processor(image)
        texts = prepare_texts([text_prompt], conv_temp)
        print(texts)
        answers = model.generate(image.unsqueeze(0), texts, max_new_tokens=400, do_sample=False, repetition_penalty=1.05)
        answer = answers[0].split('###')[0]  # remove the stop sign '###'
        answer = answer.split('Assistant:')[-1].strip()
        outputs[f'v1_{question_id}'] = answer
        print(answer)

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, 'w') as f:
    json.dump(outputs, f, indent=4)