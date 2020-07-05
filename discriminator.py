import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torch.distributions import Normal

class Discriminator(nn.Module):

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss()

        # adapted from DCGAN model tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
        self.ndf = 19
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(1, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def gan_loss(self, logits, labels):
        # expand label value and compute loss
        labels = torch.full(logits.shape, labels, dtype=logits.dtype)
        loss = torch.mean(self.criterion(logits, labels))

        return loss

    def forward(self, states, actions, next_states):
        # build trajectories and run discriminator
        trajectories = torch.cat((actions, states, next_states), 2)
        self.ndf = states.shape[2]
        logits = self.main(trajectories)
        return logits

    def loss(self, states, actions, predicted_next_states, labels):
        # get logits from forward pass of model
        logits = self.forward(states, actions, predicted_next_states)
        # check if the answers were right
        loss = self.gan_loss(logits, labels)

        return loss

    def d_loss(self, states, actions, next_states, predicted_next_states):
        # get logits from forward pass of model
        logits_true = self.forward(states, actions, next_states)
        logits_gen = self.forward(states, actions, predicted_next_states)

        # compute model loss for generated and true trajectories
        loss_on_gen = self.gan_loss(logits_gen, 1.0)
        loss_on_true = self.gan_loss(logits_true, 0.0)

        loss = loss_on_gen + loss_on_true

        # compute fraction_correct
        num_correct_gen = torch.count_nonzero(torch.gt(logits_gen, 0.5))
        num_correct_true = torch.count_nonzero(torch.lt(logits_true, 0.5))
        fraction_correct = (num_correct_gen + num_correct_true) / logits.shape[0]

        # threshold loss values
        loss = np.where(fraction_correct < self.threshold, loss, 0)

        return loss
