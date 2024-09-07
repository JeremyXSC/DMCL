import os
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

#os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import timm
from model_visual import *
from transformers import BertModel, BertTokenizer
from models import Transformer

class process_text(nn.Module):
    
    def __init__(self):
        super(process_text, self).__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained('/home/xuweishi/colo_prp/guanjl/project_text_image_e2e_pretrained/bert-base-uncased') 
    
    def forward(self,text):
        max_len=289+2
        text=text
        text=self.tokenizer.tokenize(text)
        text1=text
        new_tokens1 = ['[CLS]'] + text1 + ['[SEP]']
        new_tokens1 = new_tokens1 + ['[PAD]' for _ in range(max_len - len(new_tokens1))]
        attn_mask1 = [1 if token != '[PAD]' else 0 for token in new_tokens1]
        seg_ids1 = [0 for _ in range(len(new_tokens1))]
        token_ids1 = self.tokenizer.convert_tokens_to_ids(new_tokens1)
        token_ids1 = torch.tensor(token_ids1).unsqueeze(0)
        attn_mask1 = torch.tensor(attn_mask1).unsqueeze(0)
        seg_ids1 = torch.tensor(seg_ids1).unsqueeze(0)
        return token_ids1,attn_mask1,seg_ids1
  
    

class net_self(nn.Module):

    def __init__(self,classes_num=31,choice='resnet'):
        super(net_self, self).__init__()
        self.bert = BertModel.from_pretrained('/home/xuweishi/colo_prp/guanjl/project_text_image_e2e_pretrained/bert-base-uncased') 
        if choice=='resnet':
            self.back=ft_net(class_num=31,circle=True)
        elif choice=='swin':
            self.back=ft_net_swin(class_num=31,circle=True)
            
        self.tran=Transformer(n_layers=4)
        self.lin_down=nn.Linear(768,512)
        self.lin_cla=nn.Linear(512,classes_num)

    def forward(self, image,text,attn_mask,seg_ids):
        text_out=self.bert(text, attention_mask = attn_mask,token_type_ids = seg_ids)
        out1=text_out[0]
        seq_len=out1.size()[-2]
        feature_dim=out1.size()[-1]
        out1=out1.contiguous().view(-1,seq_len,feature_dim)
        _,out2=self.back(image)
        out2=out2.unsqueeze(1)
        logits,ff=self.tran(out2,out1)
        
        return logits,ff
    
    def forward_image(self,image):
        _,out2=self.back(image)
        out2=out2.unsqueeze(1)
        return out2
    
    def forward_text(self,text,attn_mask,seg_ids):
        text_out=self.bert(text, attention_mask = attn_mask,token_type_ids = seg_ids)
        out1=text_out[0]
        seq_len=out1.size()[-2]
        feature_dim=out1.size()[-1]
        out1=out1.contiguous().view(-1,seq_len,feature_dim)
        return out1
    
    def forward_total(self,image_out,text_out):
        logits,ff=self.tran(image_out,text_out)
        return logits,ff
