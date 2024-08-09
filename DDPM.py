import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       

        betas = torch.linspace(beta_1, beta_T, T, device=device)
        beta_t = betas[t_s-1]
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1/torch.sqrt(alpha_t)

        t_s_array = t_s.cpu().numpy()
        if t_s.dim()==0:
            beta_upto_ts = betas[:t_s]
            alpha_t_bar = torch.cumprod(1 - beta_upto_ts, dim=0)[t_s-1]
        else:
            alpha_t_bar = torch.zeros(t_s.shape,device=device)
            for index in range(t_s_array.shape[0]):
                beta_upto_ts = betas[:t_s_array[index,0]]
                alpha_t_bar[index] = torch.cumprod(1 - beta_upto_ts, dim=0)[t_s_array[index,0]-1]
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1-alpha_t_bar)


        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  

        batch = images.size(0)
        conditions = F.one_hot(conditions, num_classes=self.dmconfig.num_classes) # one hot encoding

        drop_batch_idx = torch.rand(batch, device=device) <= self.dmconfig.mask_p 
        conditions[drop_batch_idx,:] = self.dmconfig.condition_mask_value

        t_s = torch.randint(1, T+1, size=(batch,1), device=device)
        noise = torch.randn_like(images, device=device)
        scheduler = self.scheduler(t_s)
        x_t = scheduler['sqrt_alpha_bar'].view(-1,1,1,1) * images + scheduler['sqrt_oneminus_alpha_bar'].view(-1,1,1,1) * noise
        
        t_s_normalized = ((t_s-1.0)/(T-1.0)).view(-1,1,1,1)
        noise_loss = self.loss_fn(self.network(x_t,t_s_normalized,conditions),noise)

        # ==================================================== #

        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        batch = conditions.size(dim=0)
        num_channels = self.dmconfig.num_channels
        input_dim = self.dmconfig.input_dim

        X_t = torch.randn((batch, num_channels, input_dim[0], input_dim[1]), device=device)
        unconditioned = torch.full_like(conditions, self.dmconfig.condition_mask_value, device=device)

        with torch.no_grad():
            for t in torch.arange(T,0,-1):
                t_s = torch.full((batch,1),t,device=device)
                t_s_normalized = (t_s-1) / (T-1)
                z = torch.randn_like(X_t,device=device) if t>1 else torch.zeros_like(X_t,device=device)
                noise_t = (1+omega)*self.network(X_t,t_s_normalized,conditions) - omega*self.network(X_t,t_s_normalized,unconditioned)
                
                scheduler = self.scheduler(t)
                alpha_t = scheduler['alpha_t'].view(-1,1,1,1)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_oneminus_alpha_t = scheduler['sqrt_oneminus_alpha_bar'].view(-1,1,1,1)
                sigma_t = scheduler['sqrt_beta_t'].view(-1,1,1,1)
                X_t = 1/sqrt_alpha_t * (X_t - (1-alpha_t)/sqrt_oneminus_alpha_t * noise_t) + sigma_t*z

        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images