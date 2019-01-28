import dataloader as DL
from config import config
import network as net
from math import floor, ceil
import os, sys
import torch
from torch.autograd import grad
import torchvision.transforms as transforms
from torch.optim import Adam
from tqdm import tqdm
import utils as utils
import numpy as np
from dotenv import load_dotenv
from slack_print import SlackPrint

# environmental variables
load_dotenv('.env')
# slack initialization
sp = SlackPrint(os.environ['ACCESS_TOKEN'], '#dl_prog_tacchan7412')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trainer:
    def __init__(self, config):
        self.config = config
        
        self.nz = config.nz
        self.optimizer = config.optimizer

        self.resl = 2           # we start from 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen':None, 'dis':None}
        self.complete = 0
        self.phase = 'init'
        self.flag_flush = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift

        self.lambda_ = config.lambda_
        
        # network and cirterion
        self.G = net.Generator(config)
        self.D = net.Discriminator(config)
        print ('Generator structure: ')
        print(self.G.model)
        print ('Discriminator structure: ')
        print(self.D.model)
        self.mse = torch.nn.MSELoss()
        if torch.cuda.device_count() > 1:
            self.G = torch.nn.DataParallel(self.G)
            self.D = torch.nn.DataParallel(self.D)
        self.G = self.G.to(device)
        self.D = self.D.to(device)
        
        # define tensors, ship model to cuda, and get dataloader.
        self.renew_everything()
        

    def resl_scheduler(self):
        '''
        this function will schedule image resolution(self.resl) progressively.
        it should be called every iteration to ensure resl value is updated properly.
        step 1. (trns_tick) --> transition in generator and discriminator
        step 2. (stab_tick) --> stabilize.
        each step sees 800k images
        '''
        if floor(self.resl) != 2 :
            self.trns_tick = self.config.trns_tick
            self.stab_tick = self.config.stab_tick
        
        self.batchsize = self.loader.batchsize
        delta = 1.0/(self.trns_tick+self.stab_tick)
        d_alpha = 1.0*self.batchsize/self.trns_tick/self.TICK

        # update alpha if fade-in layer exist.
        if self.fadein['gen'] is not None and self.fadein['dis'] is not None:
            if self.complete < 100:
                self.fadein['gen'].update_alpha(d_alpha)
                self.fadein['dis'].update_alpha(d_alpha)
                self.complete = self.fadein['gen'].alpha*100  # (=self.fadein['dis'].alpha*100)
                self.phase = 'trns'
            else:
                self.phase = 'stab'
            
        prev_kimgs = self.kimgs
        self.kimgs = self.kimgs + self.batchsize
        # when global tick counts up
        if (self.kimgs%self.TICK) < (prev_kimgs%self.TICK):
            self.globalTick = self.globalTick + 1
            # increase linearly every tick, and grow network structure.
            prev_resl = floor(self.resl)
            self.resl = self.resl + delta
            self.resl = max(2, min(10.5, self.resl))        # clamping, range: 4 ~ 1024

            # flush network. (= remove fadein and go to stab)
            if self.flag_flush and self.complete >= 100 and prev_resl!=2:
                self.flag_flush = False
                self.phase = 'stab'
                self.complete = 0.0
                self.G.module.flush_network()   # flush G
                print(self.G.module.model)
                self.G.to(device)
                self.fadein['gen'] = None
                self.D.module.flush_network()   # flush and,
                self.D.to(device)
                print(self.D.module.model)
                self.fadein['dis'] = None

            # grow network. (= add fadein and go to trns)
            if floor(self.resl) != prev_resl and floor(self.resl)<self.max_resl+1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.module.grow_network(floor(self.resl))
                self.G.to(device)
                self.D.module.grow_network(floor(self.resl))
                self.D.to(device)
                self.renew_everything()
                self.fadein['gen'] = self.G.module.model.fadein_block
                self.fadein['dis'] = self.D.module.model.fadein_block
                self.flag_flush = True

            if floor(self.resl) > self.max_resl:
                self.phase = 'final'
                self.resl = self.max_resl


            
    def renew_everything(self):
        # renew dataloader.
        self.loader = DL.dataloader(config)
        self.loader.renew(min(floor(self.resl), self.max_resl))
        
        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz).to(device)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize).to(device)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize).to(device)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1).to(device)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).fill_(0).to(device)

        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
        

    def add_noise(self, x):
        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde) * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        z = (torch.from_numpy(np.random.randn(*x.size()).astype(np.float32)) * strength).to(device)
        return x + z


    def train(self):
        # noise for test.
        self.z_test = torch.FloatTensor(self.loader.batchsize, self.nz).to(device)
        self.z_test.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
        
        # 2^2 -> 2^(max_resl) -> 5 steps to stabilize (?)
        for step in range(2, self.max_resl+1+5):
            print(step)
            for i in range(0, (self.trns_tick+self.stab_tick)*self.TICK, self.loader.batchsize):
                print(i)
                self.globalIter = self.globalIter+1
                self.stack = self.stack + self.loader.batchsize
                if self.stack > ceil(len(self.loader.dataset)):
                    self.epoch = self.epoch + 1
                    self.stack = int(self.stack%(ceil(len(self.loader.dataset))))

                # reslolution scheduler.
                self.resl_scheduler()
                
                # zero gradients.
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator.
                self.x = self.loader.get_batch().to(device)
                if self.flag_add_noise:
                    self.x = self.add_noise(self.x)
                self.z.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0).to(device)
                self.x_tilde = self.G(self.z)
               
                self.fx = self.D(self.x)
                self.fx_tilde = self.D(self.x_tilde.detach())
                # mse loss
                # loss_d = self.mse(self.fx, self.real_label) + self.mse(self.fx_tilde, self.fake_label)
                # WGAN_GP loss
                loss_d_real = -torch.mean(self.fx)
                loss_d_fake = torch.mean(self.fx_tilde)
                # gradient penalty
                alpha = torch.rand((self.loader.batchsize, 1, 1, 1)).to(device)
                x_hat = alpha * self.x.detach() + (1 - alpha) * self.x_tilde.detach()
                x_hat.requires_grad_()
                pred_hat = self.D(x_hat)
                gradients = grad(outputs=pred_hat, inputs=x_hat, 
                                 grad_outputs=torch.ones(pred_hat.size()).to(device),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
                loss_d = loss_d_real + loss_d_fake + gradient_penalty
                loss_d.backward()
                self.opt_d.step()

                # update generator.
                fx_tilde = self.D(self.x_tilde)
                # mse loss
                # loss_g = self.mse(fx_tilde, self.real_label.detach())
                # WGAN_GP loss
                loss_g = -torch.mean(fx_tilde) 
                loss_g.backward()
                self.opt_g.step()

                # logging.
                log_msg = ' [E:{0}][T:{1}][{2:6}/{3:6}]  errD: {4:.4f} | errG: {5:.4f} | [lr:{10:.5f}][cur:{6:.3f}][resl:{7:4}][{8}][{9:.1f}%]'.format(self.epoch, self.globalTick, self.stack, len(self.loader.dataset), loss_d.item(), loss_g.item(), self.resl, int(pow(2,floor(self.resl))), self.phase, self.complete, self.lr)
                sp.print(log_msg)

                # save model.
                self.snapshot('repo/model')

                # save image grid.
                if self.globalIter%self.config.save_img_every == 0:
                    x_test = self.G(self.z_test)
                    os.system('mkdir -p repo/save/grid')
                    utils.save_image_grid(x_test.detach(), 'repo/save/grid/{}_{}_{}.jpg'.format(int(self.globalIter/self.config.save_img_every), self.phase, self.complete))
                    sp.upload('repo/save/grid/{}_{}_{}.jpg'.format(int(self.globalIter/self.config.save_img_every), self.phase, self.complete))
                    os.system('mkdir -p repo/save/resl_{}'.format(int(floor(self.resl))))
                    utils.save_image_single(x_test.detach(), 'repo/save/resl_{}/{}_{}.jpg'.format(int(floor(self.resl)),int(self.globalIter/self.config.save_img_every), self.phase, self.complete))
                    sp.upload('repo/save/resl_{}/{}_{}.jpg'.format(int(floor(self.resl)),int(self.globalIter/self.config.save_img_every), self.phase, self.complete))


    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def snapshot(self, path):
        if not os.path.exists(path):
            os.system('mkdir -p {}'.format(path))
        # save every 100 tick if the network is in stab phase.
        ndis = 'dis_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.globalTick)
        ngen = 'gen_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.globalTick)
        if self.globalTick%50==0:
            if self.phase == 'stab' or self.phase == 'final':
                save_path = os.path.join(path, ndis)
                if not os.path.exists(save_path):
                    torch.save(self.get_state('dis'), save_path)
                    save_path = os.path.join(path, ngen)
                    torch.save(self.get_state('gen'), save_path)
                    print('[snapshot] model saved @ {}'.format(path))


## perform training.
print('----------------- configuration -----------------')
for k, v in vars(config).items():
    print('  {}: {}'.format(k, v))
print('-------------------------------------------------')
torch.backends.cudnn.benchmark = True           # boost speed.
trainer = trainer(config)
trainer.train()
