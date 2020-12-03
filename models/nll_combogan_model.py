import numpy as np
import time
import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import gkern_2d


class NLLComboGANModel(BaseModel):
    def name(self):
        return 'NLLComboGANModel'

    def __init__(self, opt):
        super(NLLComboGANModel, self).__init__(opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None

        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # the log variance is given after tanh. A variance of 25 is enough to mimic a uniform distribution
        #
        self.unc_constant = 5.0

        # load/define networks
        self.netG = networks.define_G(opt.input_nc,
                                      opt.output_nc + 1,  # +1: used for the uncertainties
                                      opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, self.Tensor, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            # this is a huber loss
            self.L1 = torch.nn.SmoothL1Loss()
            self.downsample = torch.nn.AvgPool2d(3, stride=2)
            # Use NLL instead of L1
            self.criterionCycle = lambda generated_image, real_image, log_output_variance: \
                torch.sum((generated_image - real_image)**2 * torch.exp(-log_output_variance) + log_output_variance)
            # self.criterionCycle = self.L1
            self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t))
            self.criterionLatent = lambda y,t : self.L1(y, t.detach())
            self.criterionGAN = lambda r,f,v : (networks.GANLoss(r[0],f[0],v) + \
                                                networks.GANLoss(r[1],f[1],v) + \
                                                networks.GANLoss(r[2],f[2],v)) / 3
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle = [0]*self.n_domains
            # initialize loss multipliers
            self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
            self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward
        else:
            # Init the kernels on gpu
            # dilatation of the mask
            dilation_size = self.opt.blur_dilat_size
            dilation_pad = (dilation_size - 1) // 2
            self.dil_tuple = (dilation_pad, dilation_pad)
            kernel = np.ones([dilation_size, dilation_size])
            self.kernel_tensor = self.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))  # size: (1, 1, 3, 3)
            # gaussian
            blur_size = self.opt.blur_gauss_size
            sigma = self.opt.blur_gauss_sigma
            self.blur_tuple = [(blur_size - 1) // 2] * 2
            gaussian_np = gkern_2d(size=blur_size, sigma=sigma)  # .transpose([1, 0, 2, 3])
            self.gaussian = self.Tensor(gaussian_np)

        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
        self.image_paths = input['path']

    def test(self):
        with torch.no_grad():
            fakes = []
            self.visuals = [self.real_A]
            self.labels = ['real_%d' % self.DA]



            # cache encoding to not repeat it everytime
            encoded = self.netG.encode(self.real_A, self.DA)
            if self.opt.flip_export:
                encoded_flipped = self.netG.encode(torch.flip(self.real_A, [3]), self.DA)
            for d in range(self.n_domains):
                if d == self.DA and not self.opt.autoencode:
                    continue
                fake = self.netG.decode(encoded, d)
                fake_uncertainty = fake[:, -1:, ...] * self.unc_constant
                fake = fake[:, :-1, ...]
                fakes.append(fake)
                self.visuals.append(fake)
                self.labels.append('fake_%d' % d)
                self.visuals.append(self._normalize_unc_img(fake_uncertainty))
                self.labels.append('fake_%d_std' % d)

                if self.opt.flip_export:
                    output_flipped = self.netG.decode(encoded_flipped, d)
                    output_flipped = output_flipped[:, :-1, ...]
                    fake_flip = torch.flip(output_flipped, [3])
                    self.visuals.append(fake_flip)
                    fakes.append(fake_flip)
                    self.labels.append('fake_flip_%d' % d)

                if self.opt.reconstruct:
                    rec = self.netG.forward(fake, d, self.DA)
                    rec_uncertainty = rec[:, -1:, ...] * self.unc_constant
                    rec = rec[:, :-1, ...]
                    self.visuals.append(rec)
                    self.labels.append('rec_%d' % d)
                    self.visuals.append(self._normalize_unc_img(rec_uncertainty))
                    self.labels.append('rec_%d_std' % d)

                    # sum both uncertainties
                    self.visuals.append(self._normalize_unc_img(0.5 * (rec_uncertainty + fake_uncertainty)))
                    self.labels.append('sum_%d_std' % d)

                if self.opt.blur:
                    # blur uncertain regions
                    threshold = self.opt.blur_thresh
                    sum_mask = (rec_uncertainty + fake_uncertainty) > threshold
                    torch_result = torch.nn.functional.conv2d(sum_mask.float(), self.kernel_tensor, padding=self.dil_tuple)
                    sum_mask = torch_result > 0

                    # Blur the synthetic day image
                    fake = (fake/2.0) + 0.5
                    blurred_output = torch.nn.functional.conv2d(fake, self.gaussian, padding=self.blur_tuple, groups=3)
                    blurred_output = torch.clamp(blurred_output, min=0, max=1)

                    # Select blurred pixels
                    blurred_part = blurred_output.masked_fill(~sum_mask, 0.0)
                    sharp_part = fake.masked_fill(sum_mask, 0.0)
                    merged_output = blurred_part + sharp_part
                    merged_output = 2.0*merged_output - 1.0
                    self.visuals.append(merged_output)
                    self.labels.append('blurred_%d' % d)

            fakes = torch.stack(fakes)
            faked_std, fakes_mean = torch.std_mean(fakes, dim=0)
            self.visuals.append(fakes_mean)
            self.labels.append('mean_%d' % d)
            self.visuals.append(faked_std)
            self.labels.append('std_%d' % d)

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, pred_real, fake, domain):
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D = self.criterionGAN(pred_real, pred_fake, True) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.pred_real_B, fake_B, self.DB)
        #D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.pred_real_A, fake_A, self.DA)

    def backward_G(self):
        encoded_A = self.netG.encode(self.real_A, self.DA)
        encoded_B = self.netG.encode(self.real_B, self.DB)

        # Optional identity "autoencode" loss
        if self.lambda_idt > 0:
            # Same encoder and decoder should recreate image
            idt_A = self.netG.decode(encoded_A, self.DA)
            loss_idt_A = self.criterionIdt(idt_A, self.real_A)
            idt_B = self.netG.decode(encoded_B, self.DB)
            loss_idt_B = self.criterionIdt(idt_B, self.real_B)
        else:
            loss_idt_A, loss_idt_B = 0, 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG.decode(encoded_A, self.DB)
        self.fake_B_uncertainty = self.fake_B[:, -1:, ...] * self.unc_constant
        self.fake_B = self.fake_B[:, :-1, ...]

        pred_fake = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(self.pred_real_B, pred_fake, False)
        # D_B(G_B(B))
        self.fake_A = self.netG.decode(encoded_B, self.DA)
        self.fake_A_uncertainty = self.fake_A[:, -1:, ...] * self.unc_constant
        self.fake_A = self.fake_A[:, :-1, ...]
        pred_fake = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(self.pred_real_A, pred_fake, False)
        # Forward cycle loss
        rec_encoded_A = self.netG.encode(self.fake_B, self.DB)
        self.rec_A = self.netG.decode(rec_encoded_A, self.DA)
        self.rec_A_uncertainty = self.rec_A[:, -1:, ...] * self.unc_constant
        self.rec_A = self.rec_A[:, :-1, ...]
        if self.opt.sum_nll_uncertainties:
            self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A,
                                                           self.rec_A_uncertainty + self.fake_B_uncertainty)
        else:
            self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A, self.rec_A_uncertainty)
        # Backward cycle loss
        rec_encoded_B = self.netG.encode(self.fake_A, self.DA)
        self.rec_B = self.netG.decode(rec_encoded_B, self.DB)
        self.rec_B_uncertainty = self.rec_B[:, -1:, ...] * self.unc_constant
        self.rec_B = self.rec_B[:, :-1, ...]
        if self.opt.sum_nll_uncertainties:
            self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B,
                                                           self.rec_B_uncertainty + self.fake_A_uncertainty)
        else:
            self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B, self.rec_B_uncertainty)

        # Optional cycle loss on encoding space
        if self.lambda_enc > 0:
            loss_enc_A = self.criterionLatent(rec_encoded_A, encoded_A)
            loss_enc_B = self.criterionLatent(rec_encoded_B, encoded_B)
        else:
            loss_enc_A, loss_enc_B = 0, 0

        # Optional loss on downsampled image before and after
        if self.lambda_fwd > 0:
            loss_fwd_A = self.criterionIdt(self.fake_B, self.real_A)
            loss_fwd_B = self.criterionIdt(self.fake_A, self.real_B)
        else:
            loss_fwd_A, loss_fwd_B = 0, 0

        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc + \
                 (loss_idt_A + loss_idt_B) * self.lambda_idt + \
                 (loss_enc_A + loss_enc_B) * self.lambda_enc + \
                 (loss_fwd_A + loss_fwd_B) * self.lambda_fwd
        loss_G.backward()

    def optimize_parameters(self):
        self.pred_real_A = self.netD.forward(self.real_A, self.DA)
        self.pred_real_B = self.netD.forward(self.real_B, self.DB)
        # G_A and G_B
        self.netG.zero_grads(self.DA, self.DB)
        self.backward_G()
        self.netG.step_grads(self.DA, self.DB)
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses = extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses)])

    def _normalize_unc_img(self, img):
        # squash the standard deviation between 0 and 1.  1 is the maximum sample variance on the interval [-1,+1]
        img = torch.sqrt(torch.exp(img/5.0))
        img = (img - np.sqrt(np.exp(-1)))
        img = img/(np.sqrt(np.exp(+1)) - np.sqrt(np.exp(-1)))
        return img

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B,
                            self._normalize_unc_img(self.fake_B_uncertainty), self.rec_A,
                            self._normalize_unc_img(self.rec_A_uncertainty),
                            self.real_B, self.fake_A,
                            self._normalize_unc_img(self.fake_A_uncertainty),
                            self.rec_B, self._normalize_unc_img(self.rec_B_uncertainty)]
            self.labels = ['real_A', 'fake_B', 'fake_B_uncertainty', 'rec_A', 'rec_A_uncertainty', 'real_B',
                           'fake_A', 'fake_A_uncertainty',
                           'rec_B', 'rec_B_uncertainty']
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)

        if self.opt.lambda_latent > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_enc = self.opt.lambda_latent * decay_frac
