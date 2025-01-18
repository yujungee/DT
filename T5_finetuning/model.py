from transformers import AutoTokenizer, T5Model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import argparse
import os
import numpy as np
import random
from tqdm import tqdm 
import pickle

import decision_transformer.method.lora as lora


from torch.utils.data import Dataset, DataLoader
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0)
args = parser.parse_args()



class CustomDataset(Dataset):
    def __init__(self, vis_processor, dataset_file):
        self.vis_processors = vis_processor
        # self.text_processors =text_processor
        self.dataset_file = dataset_file

        # print(self.dataset_file)

        self.data_len = len(dataset_file['samples'])
        print(self.data_len)


        

        self.image_list, self.question_list, self.answer_list, self.weight_list = [], [], [], []


        for sample in self.dataset_file["samples"]:
            self.image_list.append(sample["image"])
            self.question_list.append(sample["input_text"])
            self.answer_list.append(sample["answer"])



    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        image = self.image_list[idx]
        image = Image.open(image).convert("RGB")
        image = self.vis_processors["eval"](image).unsqueeze(0).to(device)
        question = self.question_list[idx]
        answer = self.answer_list[idx]


        return {
            "image": image,
            "text_input": question,
            "text_output": answer
        }

import torch.nn as nn


def setup_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__" :


    data_file = '/home/yujung/offline/offline_baselines_jax/gym/instruct_blip_training_ft.pickle' 
    

    with open(data_file, "rb") as f:
        data_json = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small")
    # vis_processors.to(device)

    model_save_path = ""
    # print(data_json)
    dataset = CustomDataset(tokenizer, data_json)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    init_lr = 1e-5
    min_lr = 0
    warmup_lr =  1e-8
    warmup_steps =  1000
    weight_decay =  0.05
    max_epoch = 100
    num_workers= 4
    accum_grad_iters= 1

    max_len=30
    min_len= 8
    num_beams= 5

    seed= 42

    setup_seeds(seed)


    for i in range(model.n_layer):
        model.h[i].attn.c_attn = lora.MergedLinear(
            model.n_embd, model.n_embd * 3, 
            r=1, 
            lora_alpha=128, 
            lora_dropout=0.0, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
    optimizer = torch.optim.Adam(
        model.Qformer.parameters(),
            lr=init_lr, 
            weight_decay=weight_decay,

        )
    
    

    for name, param in model.named_parameters():
        if "Qformer" not in name:
            param.requires_grad_(False)

    #### double check

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(enabled)

    model.to(device)


    epoch = 100
    best_score=0
    for ep in tqdm(range(epoch)) :

        optimizer.zero_grad()
        # random.shuffle(videos)
        total_macs = 0

        accuracy = 0
        count = 0
        for i, sample in tqdm(enumerate(dataloader)) :

            input_ids = sample


            decoder_input_ids = model._shift_right(decoder_input_ids)

            # forward pass
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            loss = output["loss"]
            last_hidden_states = outputs.last_hidden_state


            sample["image"] = sample["image"].squeeze(1)
        
            output = model(sample)
            # print(loss)

            generate_text = model.predict_answers(samples=sample, max_len=2)
            # print(generate_text)
            answer = sample["text_output"]

            accuracy += sum(1 for gen_ans, ans in zip(generate_text, answer) if gen_ans == ans)/len(answer)
            count += 1
            # if 

            loss.backward()

            # step
            optimizer.step()

        accuracy = accuracy/count
        print("loss: ", loss, "acuuracy: ", accuracy)
        # step
        save_path = f"{model_save_path}/model_{ep}.pth"
        save_path_best = f"{model_save_path}/model_best.pth"
        if accuracy > best_score :
            print("best model saving")
            best_score = accuracy
            torch.save(model.Qformer.state_dict(), save_path_best)

        if ep % 10 == 0 :
            print( "model saving")
            torch.save(model.Qformer.state_dict(), save_path)