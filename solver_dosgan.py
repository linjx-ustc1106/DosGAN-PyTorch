from model import *
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import itertools
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(output[0]) < topk[1]:
        topk = (1, len(output[0]))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.ft_num = config.ft_num
        self.c_dim = config.c_dim
        self.d_conv_dim = config.d_conv_dim
        self.d_repeat_num = config.d_repeat_num
        self.n_blocks = config.n_blocks
        self.lambda_rec = config.lambda_rec
        self.lambda_rec2 = config.lambda_rec2
        self.lambda_gp = config.lambda_gp
        self.lambda_fs = config.lambda_fs

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.image_size = config.image_size

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.cls_save_dir = config.cls_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.encoder = ResnetEncoder()
        self.decoder = ResnetDecoder(ft_num=self.ft_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, ft_num = self.ft_num) 
        self.C = Classifier(c_dim = self.c_dim, ft_num = self.ft_num, n_blocks = self.n_blocks)

        self.g_optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)


    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        
        print('Loading the trained models from step {}...'.format(resume_iters))
        encoder_path = os.path.join(self.model_save_dir, '{}-encoder.ckpt'.format(resume_iters))
        decoder_path = os.path.join(self.model_save_dir, '{}-decoder.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))



    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)


   

    def classification_loss(self, logit, target):
        return F.cross_entropy(logit, target)

        #return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
       

    def train(self):
        # Load pre-trained classification network
        cls_iter = 150000
        C_path = os.path.join(self.cls_save_dir, '{}-C.ckpt'.format(cls_iter))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
        
        # Set data loader.
        data_loader = self.data_loader

        # Set learning rate
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        
        empty = torch.FloatTensor(1, 3, self.image_size, self.image_size).to(self.device)
        empty.fill_(1)
        # Calculate domain feature centroid of each domain
        domain_sf_num = torch.FloatTensor(self.c_dim, 1).to(self.device)
        domain_sf_num.fill_(0.00000001)
        domain_sf = torch.FloatTensor(self.c_dim, self.ft_num).to(self.device)
        domain_sf.fill_(0)
        with torch.no_grad():
            for indx, (x_real, label_org) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                label_org = label_org.to(self.device)

                x_ds, x_cls = self.C(x_real)
                for j in range(label_org.size(0)):
                    domain_sf[label_org[j], :] = (domain_sf[label_org[j], :] + x_ds[j] / domain_sf_num[label_org[j], :]) * (
                                domain_sf_num[label_org[j], :] / (domain_sf_num[label_org[j], :] + 1))
                    domain_sf_num[label_org[j], :] += 1
                    
        start_time = time.time()
        # Start training.
        for i in range(start_iters, self.num_iters):

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
                
            x_real = x_real.to(self.device)
            label_org = label_org.to(self.device)
            
            x_ds, x_cls = self.C(x_real) #obtain domain feature for each real image

            #obtain domain feature centroid for each real image
            x_ds_mean = torch.FloatTensor(label_org.size(0), self.ft_num).to(self.device)
            for j in range(label_org.size(0)):
                x_ds_mean[j] = domain_sf[label_org[j]:label_org[j] + 1, :]

            # random target
            rand_idx = torch.randperm(label_org.size(0))

            trg_dst = x_ds_mean[rand_idx] 
            trg_ds = trg_dst.clone()

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_dsrec = torch.mean(
                torch.abs(x_ds.detach() - out_cls)) 

            # Compute loss with fake images.
            x_fake = self.decoder(self.encoder(x_real), trg_ds)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_fs * d_loss_dsrec + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_dsrec'] = d_loss_dsrec.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_di = self.encoder(x_real)

                x_fake = self.decoder(x_di, trg_ds)
                x_reconst1 = self.decoder(x_di, x_ds)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_dsrec = torch.mean(
                    torch.abs(trg_ds.detach() - out_cls))  

                # Target-to-original domain.
                x_fake_di = self.encoder(x_fake)

                x_reconst2 = self.decoder(x_fake_di, x_ds)

                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst1))

                g_loss_rec2 = torch.mean(torch.abs(x_real - x_reconst2))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_rec2 * g_loss_rec2 + self.lambda_fs * g_loss_dsrec  
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_rec2'] = g_loss_rec2.item()
                loss['G/loss_dsrec'] = g_loss_dsrec.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)


            # Translate fixed images for debugging.
            if (i) % self.sample_step == 0:
                with torch.no_grad():
                    out_A2B_results = [empty]

                    for idx1 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx1:idx1 + 1])

                    for idx2 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx2:idx2 + 1])

                        for idx1 in range(label_org.size(0)):
                            x_fake = self.decoder(self.encoder(x_real[idx2:idx2 + 1]), x_ds_mean[idx1:idx1 + 1])
                            out_A2B_results.append(x_fake)
                    results_concat = torch.cat(out_A2B_results)
                    x_AB_results_path = os.path.join(self.sample_dir, '{}_x_AB_results.jpg'.format(i + 1))
                    save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0) + 1,
                               padding=0)
                    print('Saved real and fake images into {}...'.format(x_AB_results_path))
                

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                encoder_path = os.path.join(self.model_save_dir, '{}-encoder.ckpt'.format(i + 1))
                decoder_path = os.path.join(self.model_save_dir, '{}-decoder.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.encoder.state_dict(), encoder_path)
                torch.save(self.decoder.state_dict(), decoder_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
    def train_conditional(self):
        # Load pre-trained classification network
        cls_iter = 150000
        C_path = os.path.join(self.cls_save_dir, '{}-C.ckpt'.format(cls_iter))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
        
        # Set data loader.
        data_loader = self.data_loader

        # Set learning rate
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        
        empty = torch.FloatTensor(1, 3, self.image_size, self.image_size).to(self.device)
        empty.fill_(1)
       
                    
        start_time = time.time()
        # Start training.
        for i in range(start_iters, self.num_iters):

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
                
            x_real = x_real.to(self.device)
            label_org = label_org.to(self.device)
            
            x_ds, x_cls = self.C(x_real) # obtain domain feature for each real image

            # random target
            rand_idx = torch.randperm(label_org.size(0))
            
            
            x_trgt = x_real[rand_idx] 
            x_trg = x_trgt.clone()

            trg_dst = x_ds[rand_idx] 
            trg_ds = trg_dst.clone()
            
            
            

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            out_src_trg, out_cls_trg = self.D(x_trg)
            
            d_loss_real = - torch.mean(out_src) - torch.mean(out_src_trg)
            
            d_loss_dsrec = torch.mean(torch.abs(x_ds.detach() - out_cls)) + torch.mean(torch.abs(trg_ds.detach() - out_cls_trg)) 

            # Compute loss with fake images.
            x_fake = self.decoder(self.encoder(x_real), trg_ds)
            
            x_fake_trg = self.decoder(self.encoder(x_trg), x_ds)
            
            out_src, out_cls = self.D(x_fake.detach())
            
            out_src_trg, out_cls_trg = self.D(x_fake_trg.detach())
            
            d_loss_fake = torch.mean(out_src) + torch.mean(out_src_trg)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_fs * d_loss_dsrec + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_dsrec'] = d_loss_dsrec.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_di = self.encoder(x_real)

                x_fake = self.decoder(x_di, trg_ds)
                x_reconst1 = self.decoder(x_di, x_ds)
                
                x_di_trg = self.encoder(x_trg)

                x_fake_trg = self.decoder(x_di_trg, x_ds)
                x_reconst1_trg = self.decoder(x_di_trg, trg_ds)
                
                out_src, out_cls = self.D(x_fake)
                out_src_trg, out_cls_trg = self.D(x_fake_trg)
                
                g_loss_fake = - torch.mean(out_src)- torch.mean(out_src_trg)
                g_loss_dsrec = torch.mean(torch.abs(trg_ds.detach() - out_cls)) + torch.mean(torch.abs(x_ds.detach() - out_cls_trg))   

                # Target-to-original domain.
                x_fake_di = self.encoder(x_fake)
                x_fake_di_trg = self.encoder(x_fake_trg)

                x_reconst2 = self.decoder(x_fake_di, x_ds)
                x_reconst2_trg = self.decoder(x_fake_di_trg, trg_ds)

                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst1)) + torch.mean(torch.abs(x_trg - x_reconst1_trg))

                g_loss_rec2 = torch.mean(torch.abs(x_real - x_reconst2)) + torch.mean(torch.abs(x_trg - x_reconst2_trg))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_rec2 * g_loss_rec2 + self.lambda_fs * g_loss_dsrec  
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_rec2'] = g_loss_rec2.item()
                loss['G/loss_dsrec'] = g_loss_dsrec.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)


            # Translate fixed images for debugging.
            if (i) % self.sample_step == 0:
                with torch.no_grad():
                    out_A2B_results = [empty]

                    for idx1 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx1:idx1 + 1])

                    for idx2 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx2:idx2 + 1])

                        for idx1 in range(label_org.size(0)):
                            x_fake = self.decoder(self.encoder(x_real[idx2:idx2 + 1]), x_ds[idx1:idx1 + 1])
                            out_A2B_results.append(x_fake)
                    results_concat = torch.cat(out_A2B_results)
                    x_AB_results_path = os.path.join(self.sample_dir, '{}_x_AB_results.jpg'.format(i + 1))
                    save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0) + 1,
                               padding=0)
                    print('Saved real and fake images into {}...'.format(x_AB_results_path))
                

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                encoder_path = os.path.join(self.model_save_dir, '{}-encoder.ckpt'.format(i + 1))
                decoder_path = os.path.join(self.model_save_dir, '{}-decoder.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.encoder.state_dict(), encoder_path)
                torch.save(self.decoder.state_dict(), decoder_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
    def cls(self):
        """Train a domain classifier"""
        # Set data loader.
        data_loader = self.data_loader
       

        # Start training from scratch or resume training.
        start_iters = 0

        # Start training.
        start_time = time.time()
        
        for i in range(start_iters, self.num_iters):

            try:
                x_real, label_org = next(data_iter)
                
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
                
            x_real = x_real.to(self.device)           # Input images.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.


            # =================================================================================== #
            #                             Train the classifier                              #
            # =================================================================================== #

            out_src, out_cls = self.C(x_real)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Backward and optimize.
            d_loss = d_loss_cls
            self.c_optimizer.zero_grad()
            d_loss.backward()
            self.c_optimizer.step()


            # Logging.
            loss = {}
              
            loss['D/loss_cls'] = d_loss_cls.item()
            

         
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                prec1, prec5 = accuracy(out_cls.data, label_org.data, topk=(1, 5))
                loss['prec1'] = prec1
                loss['prec5'] = prec5
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                    

        
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i+1))
                torch.save(self.C.state_dict(), C_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

   

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained networks.
        cls_iter = 150000
        C_path = os.path.join(self.cls_save_dir, '{}-C.ckpt'.format(cls_iter))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.data_loader
        step = 0
        empty = torch.FloatTensor(1, 3,self.image_size,self.image_size).to(self.device) 
        empty.fill_(1)
        with torch.no_grad():

            for indx, (x_real, label_org) in enumerate(data_loader):  
                x_real = x_real.to(self.device)           # Input images.
                label_org = label_org.to(self.device)
                step += label_org.size(0)
                x_ds, x_cls = self.C(x_real)
                torch.manual_seed(789)
                rand_idx = torch.randperm(label_org.size(0))
                trg_ds = x_ds[rand_idx]
                label_trg = label_org[rand_idx]
                x_target = x_real[rand_idx]
                
                out_A2B_results = [empty]

                for j in range(label_org.size(0)):
                    out_A2B_results.append(x_real[j:j+1])

                for i in range(label_org.size(0)):
                    out_A2B_results.append(x_real[i:i+1])
                    
                    for j in range(label_org.size(0)):
                        x_fake = self.decoder(self.encoder(x_real[i:i+1]), x_ds[j:j+1])
                        out_A2B_results.append(x_fake)
                results_concat = torch.cat(out_A2B_results)
                print(results_concat.size())
                x_AB_results_path = os.path.join(self.result_dir, '{}_x_AB_results.jpg'.format(indx+1)) 
                save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0)+1,padding=0)
                print('Saved real and fake images into {}...'.format(x_AB_results_path))
                
  