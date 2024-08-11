# Main source file for coding up Variational autoencoders

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ToDo: We need to write a code for the encoder, decoder and the prior
# The VAE takes all three objects and then we can optimize and play around with the different components
# Code credits:
# https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# Book: Deep generative modelling, chapter 4: latent variable models

PI = torch.tensor(math.pi)


class Encoder(nn.Module):

    def __init__(self, encoder_nn):
        super().__init__()
        self.encoder_nn = encoder_nn

    def encode(self, x):
        z_mu, z_log_var = self.encoder_nn(x)
        return z_mu, z_log_var

    def sample(self, z_mu, z_log_var):
        x_encoded = self._reparameterize(z_mu, z_log_var)
        return x_encoded

    @staticmethod
    def _reparameterize(z_mu, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        z = z_mu + std * epsilon
        return z

    def get_log_likelihood(self, z_mu, z_log_var):
        # We assume that the encoder parameterizes the normal distribution with zero covariance
        # z_mu, z_var is of dimensionality batch_size, hidden_dimensionality
        # https://www.statlect.com/fundamentals-of-statistics/multivariate-normal-distribution-maximum-likelihood
        # z_log_var is log(variance)
        D = z_mu.shape[1]
        log_likelihood = - 0.5 * D * torch.log(2 * PI) - 0.5 * D * z_log_var - 0.5 * z_mu ** 2
        # Important: check the dimensionality!
        return log_likelihood.sum()


class Decoder(nn.Module):

    def __init__(self, decoder_nn, distribution="categorical"):
        super().__init__()
        self.decoder_nn = decoder_nn
        self.distribution = distribution

    def decode(self, z):
        h_d = self.decoder_nn(z)

        if self.distribution == "categorical":
            # In case of categorical
            # h_d is B, D, L where D is the dimensionality of the input (28 x 28), L is number of categories
            # convert logits to softmax scores
            softmax_probs_d = F.softmax(h_d, dim=2) # Across the last dimension, which is the number of categories
            assert torch.allclose(softmax_probs_d, torch.ones_like(softmax_probs_d)), "not all elements of sum softmax are 1!"

            return softmax_probs_d

        elif self.distribution == "gaussian":
            # As the MNIST data is then in continuous range between 0 and 1
            x_d = torch.sigmoid(h_d)
            return x_d

    def get_log_likelihood(self, x, z):
        # This part is needed for the reconstruction loss
        x_hat = self.decode(z)

        if self.distribution == "categorical":
            one_hot_labels = F.one_hot(x)
            categories_prob = torch.clamp(x_hat, 1.e-15, 1-1.e-15) # make sure we do not run into errors log(0)
            log_p = one_hot_labels * torch.log(categories_prob).sum()

        elif self.distribution == "gaussian":
            # here we can simply take the mean-squared error, as MSE and gaussian loglikelihood are related
            log_p = ((x_hat - x) ** 2).sum()

        else:
            raise ValueError(f"the distribution {self.distribution} is not supported. "
                             f"Only categorical and gaussian")

        return log_p

    def sample(self, z):
        outs = self.decode(z)
        # This is B x D in case of gaussian and B x D x L for categorical

        if self.distribution == "categorical":
            # Draw a sample from for each D x L
            batch_size, num_features, num_categories = outs.shape
            # We sample for each feature one category and reshape it, BxD, L
            reshaped_probabilities = outs.reshape(-1, num_categories)
            sampled_categories = torch.multinomial(reshaped_probabilities, num_samples=1) # 1 per sample

            outs = sampled_categories.reshape(batch_size, num_features)

        elif self.distribution == "gaussian":
            outs = outs

        return outs


class StandardPrior(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def sample(self, batch_size):
        # this samples from standard normal distribution
        z = torch.randn((batch_size, self.latent_dim))
        return z

    def get_log_likelihood(self, z):
        # We need this for training the VAE regularizer term (the KL divergence)
        # z is of shape, batch_size x latent_dim
        # Check this:
        # https://www.statlect.com/fundamentals-of-statistics/multivariate-normal-distribution-maximum-likelihood
        # Here we have a standard multivariate normal, so det(V) is det(I) which is 1, then the log(1) is 0
        #
        log_likelihood = - 0.5 * self.latent_dim * torch.log(2 * PI) - (0.5 * z ** 2).sum()
        return log_likelihood


class VAE(nn.Module):

    def __init__(self, encoder_nn, decoder_nn, latent_dim):
        super().__init__()
        self.encoder = Encoder(encoder_nn)
        self.decoder = Decoder(decoder_nn)
        self.prior = StandardPrior(latent_dim)

    def forward(self, x):
        z_mu, z_log_var = self.encoder.encode(x)
        z = self.encoder.sample(z_mu, z_log_var)

        # ELBO (evidence lower bound, reconstruction loss and regularizer)
        reconstruction_loss = self.decoder.get_log_likelihood(x, z)
        # KL divergence is q(z|x) - p(z)
        log_q_z_given_x = self.encoder.get_log_likelihood(z_mu, z_log_var)
        log_p_z = self.prior.get_log_likelihood(z)

        kl = (log_q_z_given_x - log_p_z)

        # In pytorch we always minimize the loss and take the batch average
        elbo_loss = - (reconstruction_loss - kl).mean()

        return elbo_loss

    def sample(self, batch_size):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)


