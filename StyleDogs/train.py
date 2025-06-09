import numpy as np
import random
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F2
from torchvision import transforms
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
from PIL import Image
from model import Generator,Discriminator
from dataset import RealDataset
from torch import autograd
import matplotlib.pyplot as plt
import config as conf
import torch.nn as nn
from torchvision.utils import save_image
from math import log2
from tqdm import tqdm
from torchvision import datasets
def get_device():
    device=('cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backend.mps.is_available()
            else 'cpu'
            )
    return torch.device(device)


def get_padding(image):
    """Calculate padding to make an image square."""
    _,h,w = image.shape
    # if h>w:
        # plt.imshow(image.cpu().detach().numpy().astype(np.uint8))
    max_dim = max(w, h)
    padding_left = (max_dim - w) // 2
    padding_right = max_dim - w - padding_left
    padding_top = (max_dim - h) // 2
    padding_bottom = max_dim - h - padding_top
    return (padding_left,padding_top,padding_right,padding_bottom)


def get_transforms(image_size):
    """
    Takes numpy uint8 and returns tensor [0,1]
    """
    return v2.Compose([
        v2.Resize((image_size,image_size)),
        v2.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(
            [0.5 for _ in range(conf.num_channels)],
            [0.5 for _ in range(conf.num_channels)],
        )
        # v2.ToTensor(),  
        # v2.Lambda(lambda img:img*2-1),
        # # v2.ToImage(),
        # # v2.ToDtype(torch.float32,scale=True)
    ])

def reverse_transform(input):######## TODO:::: EDIT
    """
    Takes tensor returns numpy uint8
    """
    input=torch.clip(input,-1,1)
    input=input.permute(1,2,0)
    input=(input+1)*255/2
    return input.cpu().detach().numpy()



def print_metrics(metrics:dict):
    outputs=[]
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k,metrics[k]))
    print(outputs)


def get_real_dataloader(image_size):
    transforms=get_transforms(image_size=image_size)
    batch_size=conf.batch_sizes[int(log2(image_size/4))]
    dataset=RealDataset(conf.image_dir,transforms,conf.image_size)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)
    return dataloader,dataset


def check_loader():
    loader,_=get_real_dataloader(4)
    dog=next(iter(loader))
    _,ax=plt.subplots(3,3,figsize=(8,8))
    plt.suptitle('Some real samples')
    ind=0
    for k in range(3):
        for kk in range(3):
            temp=(dog[ind].permute(1,2,0)+1)/2
            ax[k][kk].imshow(temp)
            ind+=1
    
    pass

def generate_examples(gen,steps,epcoh='',n=100):
    device=get_device()
    gen.eval()
    alpha=1.0
    for i in range(n):
        with torch.no_grad():
            noise=torch.randn(1, conf.z_depth).to(device)
            img=gen(noise,alpha,steps)
            if not os.path.exists(os.path.join(conf.examples_save_dir,f'step{steps}')):
                os.makedirs(os.path.join(conf.examples_save_dir,f'step{steps}'))
            if epcoh!='':
                os.makedirs(os.path.join(conf.examples_save_dir,f'step{steps}',epcoh),exist_ok=True)
                save_image(img*0.5+0.5,os.path.join(conf.examples_save_dir,f'step{steps}',epcoh,f'img_{i}.png'))
            else:
                save_image(img*0.5+0.5,os.path.join(conf.examples_save_dir,f'step{steps}',f'img_{i}.png'))
    gen.train()


def calculate_gradient_penalty(critic, real, fake, alpha, train_step):
    device=get_device()
    BATCH_SIZE,C,H,W=real.shape
    beta=torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)

    # eta=torch.FloatTensor(real_images.shape[0],conf.num_channels,1,1).uniform_(0,1)
    # eta=eta.expand(batch_size,real_images.size(1),real_images.size(2),real_images.size(3))
    # eta=eta.to(device)
    # beta i eta su isto samo na razlicite nacine

    # interpolated_images=real*beta+fake.detach()*(1-beta)
    interpolated_images=real*beta+fake.detach()*(1-beta)
    interpolated_images.requires_grad_(True)

    # interpolated=eta*real_images+(1-eta)*fake_images
    # interpolated=interpolated.to(device)
    # interpolated=Variable(interpolated,requires_grad=True)
    #interpolated image sa requires grad na drugi nacin

    mixed_scores=critic(interpolated_images,alpha,train_step)
    gradient=torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # prob_interpolated=discriminator(interpolated)
    # gradient=autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            # grad_outputs=(torch.ones(prob_interpolated.size())).to(device),create_graph=True,retain_graph=True)[0]
    #gradients na drugi nacin

    gradient=gradient.view(gradient.shape[0],-1)
    gradient_norm=gradient.norm(2,dim=1)
    grad_penalty=torch.mean((gradient_norm-1)**2)
    # grad_penalty=((gradients.norm(2,dim=1)-1)**2).mean()*lambda_term
    del beta,interpolated_images,mixed_scores,gradient
    return grad_penalty


def train_fn(
        critic,
        gen,
        loader,
        dataset,
        step,
        alpha,
        opt_critic,
        opt_gen
):
    device=get_device()
    loop=tqdm(loader,leave=True)

    for batch_idx,(real) in enumerate(loop):
        real=real.to(device)
        cur_batch_size=real.shape[0]
        critic_real=critic(real,alpha,step)
        # critic_real=(critic(real,alpha,step).mean()).cpu()
        noise=torch.randn(cur_batch_size,conf.z_depth).to(device)
        fake=gen(noise,alpha,step)
        critic_fake=critic(fake.detach(),alpha,step)
        # critic_fake=critic(fake.detach(),alpha,step).mean()
        gp=calculate_gradient_penalty(critic,real,fake,alpha,step)
        loss_critic=(
            -(torch.mean(critic_real)-torch.mean(critic_fake))
            +conf.lambda_gp*gp
            +(0.001)*torch.mean(critic_real**2)
        )
        # loss_critic=(
        #     -(critic_real-critic_fake)
        #     +conf.lambda_gp*gp
        #     # +(0.001)*torch.mean(critic_real**2)
        # )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake=critic(fake,alpha,step)
        loss_gen=-torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha+=cur_batch_size/(
            conf.progressive_epoches[step]*0.5*len(dataset)
        )
        alpha=min(alpha,1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item()
        )
    return alpha

def run():
    device=get_device()
    gen=Generator(conf.z_depth,conf.w_depth,conf.in_channels,conf.num_channels).to(device)

    critic=Discriminator(conf.in_channels,conf.num_channels).to(device)

    opt_gen=optim.Adam([{'params':[param for name, param in gen.named_parameters() if 'map' not in name]},
                        {'params':gen.map.parameters(),'lr':1e-5}],lr=conf.lr_gen,betas=(conf.beta1,conf.beta2))
    
    opt_critic=optim.Adam(
        critic.parameters(),lr=conf.lr_disc,betas=(conf.beta1,conf.beta2)
    )
    gen.train()
    critic.train()
    step=int(log2(conf.start_train_img_size/4))
    os.makedirs('save',exist_ok=True)
    for num_epoches in conf.progressive_epoches[step:]:
        alpha=1e-7
        loader,dataset=get_real_dataloader(4*2**step)
        print('Curent image size: '+str(4*2**step))

        for epoch in range(num_epoches):
            print(f'Epoch [{epoch+1}/{num_epoches}]')
            alpha=train_fn(critic,gen,loader,dataset,step,alpha,opt_critic,opt_gen)
            generate_examples(gen,step,str(epoch))
        generate_examples(gen,step)
        os.makedirs(os.path.join('save',f'{step}'),exist_ok=True)
        torch.save(gen,os.path.join('save',f'{step}','g.pkl'))
        torch.save(critic,os.path.join('save',f'{step}','c.pkl'))
        step+=1


# def run(Generator:Generator,Discriminator:Discriminator ):
#     num_cycles=conf.num_cycles
#     lr_gen=conf.lr_gen#0.0001
#     lr_disc=conf.lr_disc#0.0001
#     batch_size=conf.batch_size
#     beta1=conf.beta1
#     beta2=conf.beta2
#     device=get_device()
#     generator:nn.Module=Generator(conf.num_channels,conf.image_size,conditional=False)
#     generator.load_state_dict(torch.load('./Gens/Gens5/g400.pkl'))
#     generator.to(device)
#     print(generator)
#     discriminator:nn.Module=Discriminator(conf.num_channels,1,conditional=False)
#     discriminator.load_state_dict(torch.load('./Discs/Discs5/d400.pkl'))
#     discriminator.to(device)
#     print(discriminator)

#     optimizer_generator=optim.RMSprop(
#         filter(lambda p:p.requires_grad,generator.parameters()),
#         lr=lr_gen,
#     )

#     optimizer_discriminator=optim.Adam(
#         filter(lambda p:p.requires_grad, discriminator.parameters()),
#         lr=lr_disc,
#         betas=(beta1,beta2)
#     )

#     # scheduler_generator=lr_scheduler.ReduceLROnPlateau()
#     # scheduler_discriminator=lr_scheduler.ReduceLROnPlateau()

#     r_dataloader,r_dataset=get_real_dataloader(images_path=conf.image_dir,batch_size=batch_size)
#     # f_dataloader,f_dataset=get_fake_dataloader(root_path='./Generated',index=-1,batch_size=batch_size)
#     generator,discriminator,z_save=training(generator,
#                                      discriminator,
#                                      optimizer_generator,
#                                      optimizer_discriminator,
#                                      r_dataloader,
#                                      r_dataset,
#                                     #  f_dataloader,
#                                     #  f_dataset,
#                                      num_cycles,
#                                      batch_size)
#     save_examples_binary(z_save=z_save,generator=generator)


# def load_generator(path):
#     generator=Generator(conf.num_channels)
#     generator.load_state_dict(torch.load(path))
#     return generator

# def generate_examples_binary(generator,num_examples):
#     device=get_device()
#     z_save=torch.randn(num_examples,conf.z_depth,1,1).to(device)
#     os.makedirs(conf.binary_image_generation_dir,exist_ok=True)
#     examples_save_dir=os.path.join(conf.binary_image_generation_dir,f'generated_binary{len(os.listdir(conf.binary_image_generation_dir))}')
#     os.makedirs(examples_save_dir,exist_ok=True)
#     samples=generator(z_save)
#     for i,sample in enumerate(samples):
#         img=np.squeeze(reverse_transform(sample))
#         img[img>128]=255
#         img[img<=128]=0
#         plt.imsave(os.path.join(examples_save_dir,f'{i}.png'),img)

# def save_examples_binary(z_save,generator):
#     os.makedirs(conf.binary_image_save_dir,exist_ok=True)
#     examples_save_dir=os.path.join(conf.binary_image_save_dir,f'examples_binary{len(os.listdir(conf.binary_image_save_dir))}')
#     # temp=os.path.join(examples_save_dir,f'{cycle}')
#     os.makedirs(examples_save_dir,exist_ok=True)
#     samples=generator(z_save)
#     for i,sample in enumerate(samples):
#         img=np.squeeze(reverse_transform(sample))
#         img[img>128]=255
#         img[img<=128]=0
#         plt.imsave(os.path.join(examples_save_dir,f'{i}.png'),img.astype(np.uint8))
#         # plt.imsave(os.path.join(temp,f'{i}.png'),np.squeeze(reverse_transform(sample)))    




# def training(generator:Generator,
#              discriminator:Discriminator,
#              optimizer_generator:optim.RMSprop,
#              optimizer_discriminator:optim.Adam,
#              r_dataloader:DataLoader,
#              r_dataset:RealDataset,
#             #  f_dataloader:DataLoader,
#             #  f_dataset:FakeDataset,
#              num_cycles:int,
#              batch_size:int):
#     generator_save_dir=os.path.join(conf.generators_save_dir,f'Gens{len(os.listdir(conf.generators_save_dir))}')
#     os.makedirs(generator_save_dir,exist_ok=True)
#     discriminator_save_dir=os.path.join(conf.discriminators_save_dir,f'Discs{len(os.listdir(conf.discriminators_save_dir))}')
#     os.makedirs(discriminator_save_dir,exist_ok=True)
#     examples_save_dir=os.path.join(conf.examples_save_dir,f'training_results{len(os.listdir(conf.examples_save_dir))}')
#     os.makedirs(examples_save_dir,exist_ok=True)

#     cycle_save=conf.cycle_save
#     device=get_device()
#     best_generator=copy.deepcopy(generator.state_dict())
#     best_discriminator=copy.deepcopy(discriminator.state_dict())
#     discriminator_epoches=conf.discriminator_epoches
#     one = torch.tensor(1, dtype=torch.float)
#     mone = one * -1
#     z_save=torch.randn(conf.examples_number,conf.z_depth,1,1).to(device)
#     for cycle in range(num_cycles):
#         print(f'Cycle {cycle}/{num_cycles}:')
#         print('-'*15)

#         #Discriminator update
#         since=time.time()
#         for p in discriminator.parameters():
#             p.requires_grad=True
#         for p in generator.parameters():
#             p.requires_grad=False

#         d_loss_real=0
#         d_loss_fake=0
#         W_D=0
        
#         count=0
#         real_loss=0
#         generated_loss=0
#         disc_loss_whole=0
#         gradient_penalty_whole=0
#         for batch_real in r_dataloader:
#             for p in discriminator.parameters():
#                 p.requires_grad=True
#             for p in generator.parameters():
#                 p.requires_grad=False
#             # plt.imshow(reverse_transform(batch_real[0]).astype(np.uint8))
#             # plt.show()
#             discriminator.zero_grad()
#             batch_real=batch_real.to(device)
#             d_loss_real=(discriminator(batch_real)).mean()
#             real_loss+=d_loss_real.item()
#             # d_loss_real_cycle+=d_loss_real

#             # d_loss_real.backward(mone)###OVDEEEEEEEEE
            
#             # if d_loss_real.data>=0:
#                 # d_loss_real.backward(mone)
#             # else:
#                 # d_loss_real.backward(one)

#             # z=(torch.randn((batch_size,conf.z_depth,1,1))).to(device)
#             z=(torch.randn((batch_real.shape[0],conf.z_depth,1,1))).to(device)
#             batch_fake=generator(z)

#             d_loss_fake=(discriminator(batch_fake)).mean()
#             # d_loss_fake_cycle+=d_loss_fake
#             generated_loss+=d_loss_fake.item()

#             # d_loss_fake.backward(one)# OVDEEEEEE



#             # if d_loss_fake.data>=0:
#                 # d_loss_fake.backward(one)
#             # else:
#                 # d_loss_fake.backward(mone)

#             gradient_penalty=calculate_gradient_penalty(batch_real,batch_fake,batch_real.shape[0],device,discriminator,640)
#             gradient_penalty_whole+=gradient_penalty.item()
#             # gradient_penalty.backward()#### OVDEEEEEE

#             # d_loss=d_loss_fake-d_loss_real+gradient_penalty
#             # d_loss_cycle+=d_loss
#             # W_D=d_loss_real-d_loss_fake
#             # W_D_cycle+=W_D
#             disc_loss=d_loss_fake-d_loss_real+gradient_penalty
#             # disc_loss=d_loss_fake-d_loss_real
#             disc_loss_whole+=disc_loss.item()
#             disc_loss.backward()
#             optimizer_discriminator.step()
#             count+=1
#             # if count>=discriminator_epoches:
#                 # break
#             # print("Discriminator step done")
#             del d_loss_fake, d_loss_real, gradient_penalty,z
#             torch.cuda.empty_cache()
#         #Generator update
#         for p in discriminator.parameters():
#             p.requires_grad=False
#         for p in generator.parameters():
#             p.requires_grad=True
#         generator.zero_grad()
#         z=torch.randn(batch_size,conf.z_depth,1,1).to(device)
#         fake_images=generator(z)

#         g_loss=(discriminator(fake_images)).mean()
#             # g_loss_cycle+=g_loss
#         new_gen_loss=g_loss.item()
#         g_loss.backward(mone)

#         # if g_loss.data>=0:
#             # g_loss.backward(mone)
#         # else:
#             # g_loss.backward(one)
        
#             # g_cost=-g_loss
#         optimizer_generator.step()
#         del g_loss,z,fake_images
#         torch.cuda.empty_cache()

#         generated_loss/=count
#         real_loss/=count
#         disc_loss_whole/=count
#         gradient_penalty_whole/=count
#         while new_gen_loss<0.9*real_loss:
#             generator.zero_grad()
#             z=torch.randn(batch_size,conf.z_depth,1,1).to(device)
#             fake_images=generator(z)
#             g_loss=(discriminator(fake_images)).mean()
#             new_gen_loss=g_loss.item()
#             g_loss.backward(mone)
#             optimizer_generator.step()
#             del g_loss,z,fake_images
#             torch.cuda.empty_cache()
#         # print('Generator step done')
#         print(f'real_loss: {real_loss}')
#         print(f'discriminator gen loss: {generated_loss}')
#         print(f'gradient penalty: {gradient_penalty_whole}')
#         print(f'discriminator loss : {disc_loss_whole}')
#         print(f'after generator optimization: {new_gen_loss}')
#         if cycle%cycle_save==0:
#             torch.save(generator.state_dict(),os.path.join(generator_save_dir,f'g{cycle}.pkl'))
#             torch.save(discriminator.state_dict(),os.path.join(discriminator_save_dir,f'd{cycle}.pkl'))
#             print('Models saved')

#             temp=os.path.join(examples_save_dir,f'{cycle}')
#             os.makedirs(temp,exist_ok=True)
#             samples=generator(z_save)
#             for i,sample in enumerate(samples):
#                 temp_img=(np.squeeze(reverse_transform(sample))).astype(np.uint8)
#                 plt.imsave(os.path.join(temp,f'{i}.png'),temp_img)
            
#         # metrics={
#         #     'Wasserstein distance':W_D_cycle/discriminator_epoches,
#         #     'Loss D':d_loss_cycle.data/discriminator_epoches,
#         #     'Loss G':g_loss_cycle.data/discriminator_epoches,
#         #     'Loss D Real':d_loss_real_cycle.data/discriminator_epoches,
#         #     'Loss D Fake':d_loss_fake_cycle.data/discriminator_epoches
#         #     }
#         # print_metrics(metrics)


#         time_elapsed=time.time()-since
#         print(f'{time_elapsed} seconds')

#     # torch.save(generator.state_dict(),f'./Gens/gfinal.pkl')
#     torch.save(generator.state_dict(),os.path.join(generator_save_dir,f'gfinal.pkl'))
#     # torch.save(discriminator.state_dict(),f'./Discs/dfinal.pkl')
#     torch.save(discriminator.state_dict(),os.path.join(discriminator_save_dir,f'dfinal.pkl'))
#     return generator,discriminator,z_save



  
if __name__=='__main__':
    # check_loader()
    run()
    # run(Generator,Discriminator)
    # generator=load_generator('./Gens/Gens6/g7900.pkl')
    # generator=generator.to(get_device())
    # generate_examples_binary(generator,500)