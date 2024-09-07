from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
import os
import torch
import logging
import random
import pickle
from transformers import BertModel, BertTokenizer
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
transform_train_list = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
        ])

transform_val_list = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        ])

Identity_dict = {}
Identity_dict_key = ['ch1','cxy1','gjh2','gjh4','hdr1','hfq1','hfq2','hlb1','lbl3','mqz1','mwl1','mwl2','mwl3',
                     'plh1','plh2','plh3','qfj2','qzb1','qzb2','sxx1','tsp1','wdm1','xjf1','xmh1','xxy1','yaz1',
                     'yaz2','ybs1','ykf1','zsh1','zsh2']

for i in range(len(Identity_dict_key)):
    Identity_dict[Identity_dict_key[i]] = i
print(Identity_dict)

def get_log(file_name):
    logger = logging.getLogger('*')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

import pandas as pd
class EndoDataset_query(Dataset):
    def __init__(self, file_paths,text_path,transform=None,
                 loader=pil_loader):
        self.file_paths = []
        self.label=[]
        self.text=[]
        self.text_index=[]
        text_file=pd.read_excel(text_path)
        for i in range(len(text_file)):
            self.text.append(text_file['Eng_text'][i])
            self.text_index.append(text_file['id'][i])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for i in file_paths:
            for file in os.listdir(i[0]):
                self.file_paths.append(os.path.join(i[0],file))
        
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        max_len=289+2
        img_names = self.file_paths[index]
        id=img_names.split('/')[-2]
        identity=Identity_dict[img_names.split('/')[-3]]
        imgs = self.loader(img_names)
        img_list=[]
        imgs = self.transform(imgs)
        
        index_text=self.text_index.index(img_names.split('/')[-3])
        text=self.text[index_text]
        text1=self.tokenizer.tokenize(text)
        
        new_tokens1 = ['[CLS]'] + text1 + ['[SEP]']
        new_tokens1 = new_tokens1 + ['[PAD]' for _ in range(max_len - len(new_tokens1))]
        attn_mask1 = [1 if token != '[PAD]' else 0 for token in new_tokens1]
        seg_ids1 = [0 for _ in range(len(new_tokens1))]
        token_ids1 = self.tokenizer.convert_tokens_to_ids(new_tokens1)
        token_ids1 = torch.tensor(token_ids1).unsqueeze(0)
        attn_mask1 = torch.tensor(attn_mask1).unsqueeze(0)
        seg_ids1 = torch.tensor(seg_ids1).unsqueeze(0)
        
        

        return id,imgs,identity,token_ids1,attn_mask1,seg_ids1

    def __len__(self):
        return len(self.file_paths)

class EndoDataset_gallery(Dataset):
    def __init__(self, file_paths,transform=None,
                 loader=pil_loader):
        self.file_paths = []
        self.label=[]
        for i in file_paths:
            for file in os.listdir(i[0]):
                self.file_paths.append(os.path.join(i[0],file))
            for file1 in os.listdir(i[1]):
                self.file_paths.append(os.path.join(i[1],file1))
        
        self.loader = loader
        self.transform=transform
       

    def __getitem__(self, index):
        max_len=289+2
        img_names = self.file_paths[index]
        id=img_names.split('/')[-2]
        identity=Identity_dict[img_names.split('/')[-3]]
        imgs = self.loader(img_names)
        img_list=[]
        imgs = self.transform(imgs)
    
        return id,imgs,identity
    
    def __len__(self):
        return len(self.file_paths)
    

class EndoDataset_Train(Dataset):
    def __init__(self, file_paths,text_path,index,transform=None,
                 loader=pil_loader):
        self.file_paths = []
        self.label=[]
        self.text=text_path
        self.text_index=index
        # text_file=text_path
        # for i in range(len(text_file)):
        #     self.text.append(text_file['Eng_text'][i])
        #     self.text_index.append(text_file['id'][i])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for i in file_paths:
            for file in os.listdir(i[0]):
                self.file_paths.append(os.path.join(i[0],file))
            for file1 in os.listdir(i[1]):
                self.file_paths.append(os.path.join(i[1],file1))
        
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        max_len=289+2
        img_names = self.file_paths[index]
        id=img_names.split('/')[-2]
        identity=Identity_dict[img_names.split('/')[-3]]
        idk=[]
        idk.append(identity)
        idk.append(identity)
        imgs = self.loader(img_names)
        img_list=[]
        imgs1 = self.transform(imgs)
        imgs2 = self.transform(imgs)
        img_list.append(imgs1)
        img_list.append(imgs2)
        idk=torch.tensor([identity, identity])
        final_image=torch.stack(img_list, dim=0).squeeze(0)
        
        index_text=self.text_index.index((img_names.split('/')[-3]))
        text=self.text[index_text]
        text1=self.tokenizer.tokenize(text)
        
        new_tokens1 = ['[CLS]'] + text1 + ['[SEP]']
        new_tokens1 = new_tokens1 + ['[PAD]' for _ in range(max_len - len(new_tokens1))]
        attn_mask1 = [1 if token != '[PAD]' else 0 for token in new_tokens1]
        seg_ids1 = [0 for _ in range(len(new_tokens1))]
        token_ids1 = self.tokenizer.convert_tokens_to_ids(new_tokens1)
        token_ids1 = torch.tensor(token_ids1).unsqueeze(0)
        attn_mask1 = torch.tensor(attn_mask1).unsqueeze(0)
        seg_ids1 = torch.tensor(seg_ids1).unsqueeze(0)
        
        tokens=[]
        attn=[]
        seg=[]
        
        tokens.append(token_ids1)
        tokens.append(token_ids1)

        attn.append(attn_mask1)
        attn.append(attn_mask1)
       
        seg.append(seg_ids1)
        seg.append(seg_ids1)
   

        token_out=torch.stack(tokens)
        attn_out=torch.stack(attn, dim=0)
        seg_out=torch.stack(seg, dim=0)
        



        return id,final_image,idk,token_out,attn_out,seg_out

    def __len__(self):
        return len(self.file_paths)

    

def data_split(data_root,file_paths,fold):
    video=sorted(os.listdir(data_root))
    file= pd.read_excel(file_paths)
    pair=[]
    text=[]
    text_name=[]
    video_name=[]
    for i in range(len(file)):
        text.append(file['Eng_text'][i])
        text_name.append(file['id'][i][:-1])
        
    for j in video:
        video_name.append(j)
        j=os.path.join(data_root,j)
        single_pair=[]
        single_pair.append(os.path.join(j,str(1)))
        single_pair.append(os.path.join(j,str(2)))
        pair.append(single_pair)
    text_new=[]
    index_new=[]
    for name in video_name:
        item=text[text_name.index(name[:-1])]
        text_new.append(item)
        index_new.append(name)
        
    
    # for j in pair:
    #     print("1:",j[0]," ","2:",j[1])
    #     print("*************************")
    print("finish pairing!")
    #assert num_train+num_val==len(pair), "the number of the data split is wrong!!!"
    if fold==1:
        train_pairs=pair[0:24]
        train_video=video[0:24]
        val_video=video[24:31]
        val_pairs=pair[24:31]
        train_text=text_new[0:24]
        val_text=text_new[24:31]
        train_index=index_new[0:24]
        val_index=index_new[24:31]
       
    if fold==2:
        train_pairs=pair[0:16]+pair[24:31]
        train_video=video[0:16]+video[24:31]
        val_video=video[16:24]
        val_pairs=pair[16:24]
        train_text=text_new[0:16]+text_new[24:31]
        val_text=text_new[16:24]
        train_index=index_new[0:16]+index_new[24:31]
        val_index=index_new[16:24]
        
    if fold==3:
        train_pairs=pair[0:8]+pair[16:31]
        train_video=video[0:8]+video[16:31]
        val_video=video[8:16]
        val_pairs=pair[8:16]
        train_text=text_new[0:8]+text_new[16:31]
        val_text=text_new[8:16]
        train_index=index_new[0:8]+index_new[16:31]
        val_index=index_new[8:16]
        
    if fold==4:
        train_pairs=pair[8:31]
        train_video=video[8:31]
        val_video=video[0:8]
        val_pairs=pair[0:8]
        train_text=text_new[8:31]
        val_text=text_new[0:8]
        train_index=index_new[8:31]
        val_index=index_new[0:8]
    
    return train_video,val_video,train_pairs,val_pairs,train_text,val_text,train_index,val_index

Identity_dict_val = {}
Identity_dict_key_val = ['ch1','cxy1','gjh2','gjh4','hdr1','hfq1','hfq2','hlb1','lbl3','mqz1','mwl1','mwl2','mwl3',
                     'plh1','plh2','plh3','qfj2','qzb1','qzb2','sxx1','tsp1','wdm1','xjf1','xmh1','xxy1','yaz1',
                     'yaz2','ybs1','ykf1','zsh1','zsh2']

for i in range(len(Identity_dict_key_val)):
    Identity_dict_val[Identity_dict_key_val[i]] = i
    
    
class TextDataset_Val(Dataset):
    def __init__(self, file_paths):
        file= pd.read_excel(file_paths)
   
        self.text=[]
        self.name=[]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        for i in range(len(file)):
            self.text.append(file['Eng_text'][i])
            self.name.append(file['id'][i])
        
        
        
    
    def __len__(self):
        return len(self.text)

        
    
    def __getitem__(self, index):

        max_len=289+2
        text=self.text[index]
        text=self.tokenizer.tokenize(text)
      
        name=self.name[index]
        identity=Identity_dict_val[name]

        text1=text
       

        new_tokens1 = ['[CLS]'] + text1 + ['[SEP]']
        new_tokens1 = new_tokens1 + ['[PAD]' for _ in range(max_len - len(new_tokens1))]
        attn_mask1 = [1 if token != '[PAD]' else 0 for token in new_tokens1]
        seg_ids1 = [0 for _ in range(len(new_tokens1))]
        token_ids1 = self.tokenizer.convert_tokens_to_ids(new_tokens1)
        token_ids1 = torch.tensor(token_ids1).unsqueeze(0)
        attn_mask1 = torch.tensor(attn_mask1).unsqueeze(0)
        seg_ids1 = torch.tensor(seg_ids1).unsqueeze(0)



        return token_ids1,attn_mask1,seg_ids1,identity
  
if __name__ == '__main__':
    image_root='/home/xuweishi/colo_prp/datasets/resized_frame'
    text_root='/home/xuweishi/colo_prp/guanjl/Bert/process_text.xlsx'
    train_video,val_video,train_list,val_list=data_split(image_root,fold=3)
    data_loader=EndoDataset_Train(train_list,text_root,transform_train_list)
    data_loader.__getitem__(0)
    print("finish")
    
    
