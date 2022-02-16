import torch
import torch.nn as nn

"""
@article{vandeven2019three,
  title={Three scenarios for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1904.07734},
  year={2019}
}

@article{vandeven2018generative,
  title={Generative replay with feedback connections as a general strategy for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1809.10635},
  year={2018}
}
"""

class AutoEncoder(nn.Module):

    def __init__(self, kernel_num, in_channel=1, img_sz=32, hidden_dim=256, z_size=100, bn = False):
        super(AutoEncoder, self).__init__()
        self.BN = bn
        self.in_dim = in_channel*img_sz*img_sz
        self.image_size = img_sz
        self.channel_num = in_channel
        self.kernel_num = kernel_num
        self.z_size = z_size

        # -weigths of different components of the loss function
        self.lamda_rcl = 1.
        self.lamda_vl = 1.

        # Training related components that should be set before training
        # -criterion for reconstruction
        self.recon_criterion = None

        self.encoder = nn.Sequential(
            self._conv(in_channel, 64),
            self._conv(64, 128),
            self._conv(128, 512),
        )

        self.decoder = nn.Sequential(
            self._deconv(512, 256),
            self._deconv(256, 64),
            self._deconv(64, in_channel, ReLU=False),
            nn.Sigmoid()
        )
        self.feature_size = img_sz // 8

        
        self.kernel_num = 512
        self.feature_volume = self.kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)


    def reparameterize(self, mu, logvar):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):

        # encode (forward), reparameterize and decode (backward)
        mu, logvar, hE = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        x_recon = self.decode(z)
        return (x_recon, mu, logvar, z)

    def sample(self, size):

            # set model to eval()-mode
            mode = self.training
            self.eval()
            # sample z
            z = torch.randn(size, self.z_size)
            z = z.cuda()
            with torch.no_grad():
                X = self.decode(z)
            # set model back to its initial mode
            self.train(mode=mode)
            # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor, plus classes-labels
            return X

    def loss_function(self, recon_x, x, dw, mu=None, logvar=None):
        batch_size = x.size(0)

        ###-----Reconstruction loss-----###
        reconL = (self.recon_criterion(input=recon_x.view(batch_size, -1), target=x.view(batch_size, -1))).mean(dim=1)
        reconL = torch.mean(reconL * dw)

        ###-----Variational loss-----###
        if logvar is not None:
            #---- see Appendix B from: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            # -normalise by same number of elements as in reconstruction
            variatL /= self.in_dim
            # --> because self.recon_criterion averages over batch-size but also over all pixels/elements in recon!!

        else:
            variatL = torch.tensor(0.)
            variatL = variatL.cuda()
        
        # Return a tuple of the calculated losses
        return reconL, variatL

    def train_batch(self, x, data_weights, allowed_predictions):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)'''

        # Set model to training-mode
        self.train()

        ##--(1)-- CURRENT DATA --##
        # Run the model
        recon_batch,  mu, logvar, z = self.forward(x)

        # Calculate all losses
        reconL, variatL = self.loss_function(recon_x=recon_batch, x=x, dw = data_weights, mu=mu, logvar=logvar)

        # Weigh losses as requested
        loss_total = self.lamda_rcl*reconL + self.lamda_vl*variatL

        # perform update
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        return loss_total.detach()

    def decode(self, z):
        '''Pass latent variable activations through feedback connections, to give reconstructed image [image_recon].'''
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected)

    def encode(self, x):
        '''Pass input through feed-forward connections, to get [hE], [z_mean] and [z_logvar].'''
        # encode x
        encoded = self.encoder(x)
        # sample latent code z from q given x.
        z_mean, z_logvar = self.q(encoded)
        return z_mean, z_logvar, encoded

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def _conv(self, channel_size, kernel_num, kernel_size_=4, stride_=2):
        if self.BN:
            return nn.Sequential(
                nn.Conv2d(
                    channel_size, kernel_num,
                    kernel_size=kernel_size_, stride=stride_, padding=1,
                ),
                nn.BatchNorm2d(kernel_num),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    channel_size, kernel_num,
                    kernel_size=kernel_size_, stride=stride_, padding=1,
                ),
                nn.ReLU(),
            )

    def _deconv(self, channel_num, kernel_num,ReLU=True, kernel_size_=4, stride_=2):
        if ReLU:
            if self.BN:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                    nn.BatchNorm2d(kernel_num),
                    nn.ReLU(),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                    nn.ReLU(),
                )
        else:
            if self.BN:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                    nn.BatchNorm2d(kernel_num),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)

def CIFAR_GEN(bn = False):
    return AutoEncoder(in_channel=3, img_sz=32, kernel_num=512, z_size=1024)
