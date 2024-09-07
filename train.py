import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
from datasets import data_split,get_log,EndoDataset_Train,EndoDataset_gallery
#from models_self import resnet50,SupConLoss
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from model_visual import ft_net,ft_net_swin,ft_vit
from models import Transformer
from model_new import net_self,process_text
from pytorch_metric_learning import losses, miners
import scipy.io
import logging
from apex import amp
torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser(description='Colopair training')
parser.add_argument('--train_fold', default=4, type=int, help='')
parser.add_argument('--choice', default='vit', type=str, help='')
parser.add_argument('--batchsize', default=24, type=int, help='batchsize')
parser.add_argument('--max_epoch', default=50, type=int, help='the training max epoch')
parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
parser.add_argument('--output_dir', default='./models_checkpoint', type=str, help='the output dir for model saving')
parser.add_argument('--root_dir', default='/home/xuweishi/colo_prp/datasets/resized_frame', type=str, help='the root for data')
parser.add_argument('--parallel', default=True, type=bool, help='whether multi card')
parser.add_argument('--text_path', default='process_text.xlsx', type=str, help='the root for data')
args = parser.parse_args()
#1 lr 0.0002
fold=args.train_fold
root=args.root_dir
batchsize=args.batchsize
max_epoch=args.max_epoch
lr=args.lr
output_dir=args.output_dir
choice=args.choice
text_path=args.text_path
output_dir =os.path.join(output_dir,str(lr)+'_'+choice)
print("fold:",fold)
print("root_dir:",root)
print("learning rate:",lr)
print("bacth_size:",batchsize)
print("visual network choice:",choice)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1)

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

def evaluate(qf,ql,gf,gl):
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    query_index = np.argwhere(gl==ql)

    good_index = query_index    
    CMC_tmp = compute_mAP(index, good_index)
    return CMC_tmp


def compute_mAP(index, good_index):
    ap = 0
        
    cmc = torch.IntTensor(len(index)).zero_()
    mask_ap=np.in1d(index[:len(good_index)],good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    ngood = len(good_index)
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc



transform_train_list = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])

transform_val_list = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])
        ])
#
#pkl_file_path='/home/xuweishi/colo_prp/guanjl/Bert/text_features/fold_'+str(fold)+'.pkl'
# if choice=='resnet':
#     model_visual=ft_net(class_num=31,circle=True)
#     model_visual.cuda()
#     model_path='/home/xuweishi/colo_prp/guanjl/project_baseline/models_checkpoint/'+'fold_'+str(fold)+'_checkpoint.pth'
#     model_visual.load_state_dict(torch.load(model_path))
#     model_visual.eval()

# elif choice=='swin':
#     output_dir='./models_checkpoint_swin'
#     model_visual=ft_net_swin(class_num=31,circle=True)
#     model_visual.cuda()
#     model_path='/home/xuweishi/colo_prp/guanjl/project_baseline_swin/models_checkpoint/'+'fold_'+str(fold)+'_checkpoint.pth'
#     model_visual.load_state_dict(torch.load(model_path))
#     model_visual.eval()

# elif choice=='vit':
#     output_dir='./models_checkpoint_vit'
#     model_visual=ft_vit(class_num=31,circle=True)
#     model_visual.cuda()
#     model_path='/home/xuweishi/colo_prp/guanjl/project_baseline_Vit/models_checkpoint/'+'fold_'+str(fold)+'_checkpoint.pth'
#     model_visual.load_state_dict(torch.load(model_path))
#     model_visual.eval()
model=net_self(classes_num=31)   

train_video,val_video,train_list,val_list,train_csv,val_csv,train_index,val_index=data_split(root,file_paths=text_path,fold=fold)


Projector=process_text()
        


print("train:",train_video)
print("val:",val_video)
train_data=EndoDataset_Train(train_list,train_csv,train_index,transform_train_list)

train_data_eval=EndoDataset_gallery(train_list,transform_val_list)
train_loader_eval=DataLoader(train_data_eval,batch_size=batchsize)

val_data=EndoDataset_gallery(val_list,transform_val_list)

val_loader=DataLoader(val_data,batch_size=batchsize)
train_loader=DataLoader(train_data,batch_size=batchsize,shuffle=True)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
small_dir='fold_'+str(fold)
logger_path=output_dir+'/'+small_dir+'_record.txt'
logger=get_log(logger_path)
new_dict=['ch1','cxy1','gjh2','gjh4','hdr1','hfq1','hfq2','hlb1','lbl3','mqz1','mwl1','mwl2','mwl3',
                     'plh1','plh2','plh3','qfj2','qzb1','qzb2','sxx1','tsp1','wdm1','xjf1','xmh1','xxy1','yaz1',
                     'yaz2','ybs1','ykf1','zsh1','zsh2']

#loss_function=SupConLoss().cuda()
CE_loss=nn.CrossEntropyLoss()

miner = miners.MultiSimilarityMiner()
criterion_triplet = losses.TripletMarginLoss(margin=0.3)

# optimizer = optim.Adam([
#              {'params': net.parameters(), 'lr': lr},
#          ])
#model.load_state_dict(torch.load('/home/xuweishi/colo_prp/guanjl/project_text_image_e2e_pretrained/models_checkpoint/fold_'+str(fold)+'_checkpoint.pth'))
if args.parallel:
    model=DataParallel(model)
    model.cuda()
    optimizer = optim.Adam([
             {'params': model.module.parameters(), 'lr': 0.1*lr},
         ], lr=lr)
    for param in model.module.bert.parameters():
        param.requires_grad = False
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
best_epoch=-1
best_map=0     
best_model_wts=copy.deepcopy(model.module.state_dict())
model=model.cuda()

for epoch in range(max_epoch):
        torch.cuda.empty_cache()
        model.train()
        train_loss=0
        train_correct=0
        train_start_time = time.time()
    
        for data in tqdm(train_loader):
            
            optimizer.zero_grad()
            
            id,image,label,token_out,attn_out,seg_out= data

            image = image.cuda()
            label = label.cuda().squeeze(0).view(-1)
            token_out=token_out.cuda()
            attn_out=attn_out.cuda()
            seg_out=seg_out.cuda()
            embedding_size=token_out.size()[-1]
            token_out=token_out.view(-1,embedding_size)
            attn_out=attn_out.view(-1,embedding_size)
            seg_out=seg_out.view(-1,embedding_size)
            
            inputs=image.view(-1,3,224,224)
            
            logits,ff=model(inputs,token_out,attn_out,seg_out)
            logits=logits.squeeze(0)
            loss = CE_loss(logits, label)

            hard_pairs = miner(ff, label)
            loss +=  criterion_triplet(ff, label, hard_pairs) 
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        print('train_loss_avg:',train_loss/(len(train_data)//batchsize))

        model.eval()

        
        
        label_total=[]
        label_query=[]
        label_gallery=[]

        id_total=[]
        
        val_label=[]
        val_feature=[]

        pbar = tqdm()
        with torch.no_grad():
            
            for i in range(len(val_index)):
                val_label.append(val_index[i])
                text=val_csv[i]
                token_ids1,attn_mask1,seg_ids1=Projector(text)
                token_ids1=token_ids1.cuda()
                attn_mask1=attn_mask1.cuda()
                seg_ids1=seg_ids1.cuda()
                text_feature=model.module.forward_text(token_ids1,attn_mask1,seg_ids1)
                text_feature=text_feature.squeeze(0)
                val_feature.append(text_feature)
                
            for iter, data in tqdm(enumerate(val_loader)):
                    
                id,inputs,label=data
                for i in id:
                    id_total.append(int(i))
                for j in label:
                    label_total.append(int(j.item()))
                n, c, h, w = inputs.size()
                pbar.update(n)
                ff = torch.FloatTensor(n,512).zero_().cuda()
                inputs = inputs.cuda()
                label = label.cuda().squeeze(0)
                
                inputs=inputs.view(-1,3,224,224)
                outputs= model.module.forward_image(inputs)
                outputs=torch.squeeze(outputs)
                ff += outputs
                # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                # ff = ff.div(fnorm.expand_as(ff))
                if iter == 0:
                        features = torch.FloatTensor( len(val_loader.dataset), ff.shape[1])
                start = iter*batchsize
                end = min( (iter+1)*batchsize, len(val_loader.dataset))
                features[ start:end, :] = ff 

            query_num=id_total.count(1)
            gallery_num=id_total.count(2)
            pbar.close() 
            query_features_visual = torch.FloatTensor( query_num, ff.shape[1])
            gallery_features_visual = torch.FloatTensor( gallery_num, ff.shape[1])
            a=b=0
            for k in range(len(id_total)):
                if id_total[k]==1:
                    label_query.append(label_total[k])
                    query_features_visual[a,:]=features[k]
                    a+=1
                elif id_total[k]==2:
                    label_gallery.append(label_total[k])
                    gallery_features_visual[b,:]=features[k]
                    b+=1
            
            query_features = torch.FloatTensor( query_num, 512)

            c=0
            current=[]
            k=0
            for label_item in val_label:
               
                 
                 
                index=val_label.index(label_item)
                text=torch.tensor(val_feature[index]).unsqueeze(0)

                query_list=[i for i, x in enumerate(label_query) if new_dict[x]==label_item ]
               
                k+=len(query_list)
                for j in query_list:
                
                    print("getting query feature:",c+1,'/',len(label_query),end='\r')
                    visual_feature=query_features_visual[j,:].contiguous().view(1,1,-1)
                    visual_feature=visual_feature.cuda()
                    text_feature=text.cuda()
                    _,output_feature=model.module.forward_total(visual_feature,text_feature)
                    fnorm = torch.norm(output_feature, p=2, dim=1, keepdim=True)
                    output_feature =output_feature.div(fnorm.expand_as(output_feature))
                    query_features[c,:]=output_feature
                    c+=1
                

            print("")

            query_features=query_features.cpu().numpy()

            CMC_1=0
            CMC_5=0
            CMC_10=0
            ap = 0.0
            k=0
            k1=0
            for label_item in val_label:
                k1+=1

                index=val_label.index(label_item)
                text=torch.tensor(val_feature[index]).unsqueeze(0)
                d=0
                gallery_features = torch.FloatTensor( gallery_num, 512)

                for j in range(len(label_gallery)):
                    
                    print("getting gallery feature:",d+1,'/',len(label_gallery),end='\r')
                    visual_feature=gallery_features_visual[j,:].contiguous().view(1,1,-1)
                    visual_feature=visual_feature.cuda()
                    text_feature=text.cuda()
                    _,output_feature=model.module.forward_total(visual_feature,text_feature)
                    fnorm = torch.norm(output_feature, p=2, dim=1, keepdim=True)
                    output_feature = output_feature.div(fnorm.expand_as(output_feature))
                    gallery_features[d,:]=output_feature
                    d+=1
                print("") 
                
                gallery_features=gallery_features.cpu().numpy()
                label_gallery=np.array(label_gallery)
                query_list=[i for i, x in enumerate(label_query) if new_dict[x]==label_item ]
                
                k1=0
                new_label=[]
                rank1=[]
                rank5=[]
                rank15=[]
                map=[]
                num=[]
                num=0
                ap1=0
                
                for i in range(len(query_list)):
                    
                    k+=1
                    print("loading>>>>:",k,'//',len(label_query),end='\r')
                    ap_tmp, CMC_tmp = evaluate(query_features[i],label_query[i],gallery_features,label_gallery)
                    CMC_1+=CMC_tmp[0]
                    CMC_5+=CMC_tmp[1]
                    CMC_10+=CMC_tmp[2]
                    ap += ap_tmp
                    ap1+=ap_tmp
                    num+=1
                print("k1:",k1)
                print("ap:",ap1/num)
                    
                print("")
            CMC_1 = CMC_1/len(label_query)
            CMC_5 = CMC_5/len(label_query)
            CMC_10 = CMC_10/len(label_query) #average CMC
            print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC_1,CMC_5,CMC_10,ap/len(label_query)))
           
            if ap/len(label_query) > best_map:
                best_map=ap/len(label_query)
                best_epoch=epoch
                best_model_wts=copy.deepcopy(model.module.state_dict())
        torch.save(best_model_wts, output_dir+'/'+small_dir+'_checkpoint.pth')

        
        logger.info("train_val_split : %0f" % (fold))
        logger.info("Epoch : %0f" % (epoch))
        logger.info("Train loss: %6f" % (train_loss/(len(train_data)//batchsize)))
        #logger.info("map train: %6f" % (T_ap/len(query_label1)))
        logger.info("Rank1 val: %6f" % (CMC_1))
        logger.info("Rank5 val: %6f" % (CMC_5))
        logger.info("Rank15 val: %6f" % (CMC_10))
        logger.info("map val: %6f" % (ap/len(label_query)))
        logger.info("best_epoch: %1f" % (best_epoch))
        logger.info("\n")





         