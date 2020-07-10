import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torch.distributions import Normal

# TODO
# tune m_loss_weight and a_loss_weight
# run trials to get varying random seeds
# double check signs on all loss values and utilities
# threshold for loss suppression in discrimator - try not using and also varying values
# try batch /instance norms
# action norms are much smaller for discrim than max - fix this

# check model is not using variance somewhere that causes problems for ensemble size 1
# double check why novelty measures are not zero or erroring for discriminator
# log utility from discrimator and fraction_correct for discrimator
# what are the relative sizes of states, next states, and actions?
# return zero only and return one only as reward from discriminator and see effect
# scale reward to be larger

class Discriminator(nn.Module):

    def __init__(self, threshold, device):
        super().__init__()
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.device = device

    def gan_loss(self, logits, labels):
        # expand label value
        labels = torch.full(logits.shape, labels, dtype=logits.dtype)
        # put the data on the target compute device
        labels = labels.to(self.device)
        # compute loss
        loss = self.criterion(logits, labels)

        return torch.squeeze(loss.T)

    def forward(self, states, actions, next_states):
        raise NotImplementedError

    def utility(self, states, actions, predicted_next_states, labels):
        # get logits from forward pass of model
        logits = self.forward(states, actions, predicted_next_states)
        # check if the answers were right
        loss = self.gan_loss(logits, labels)
        # make sure we always return as tensor with dim >= 1
        loss = torch.unsqueeze(loss, dim=0) if loss.dim() == 0 else loss

        return loss

    def loss(self, states, actions, predicted_next_states, labels):
        loss = -1 * torch.mean(self.utility(states, actions, predicted_next_states, labels))
        return loss

    def d_loss(self, states, actions, next_states, predicted_next_states):
        # get logits from forward pass of model
        logits_true = self.forward(states, actions, next_states)
        logits_gen = self.forward(states, actions, predicted_next_states)

        # compute model loss for generated and true trajectories
        loss_on_gen = torch.mean(self.gan_loss(logits_gen, 1.0))
        loss_on_true = torch.mean(self.gan_loss(logits_true, 0.0))

        loss = loss_on_gen + loss_on_true

        # compute fraction_correct
        num_correct_gen = np.count_nonzero(torch.gt(logits_gen, 0.5).cpu())
        num_correct_true = np.count_nonzero(torch.lt(logits_true, 0.5).cpu())
        fraction_correct = (num_correct_gen + num_correct_true) / logits_true.shape[0]

        # threshold loss values
        loss = loss if (fraction_correct > self.threshold) else torch.tensor([0.0], dtype=loss.dtype, requires_grad=True)

        return loss

class ConvDiscriminator(Discriminator):

    def __init__(self, threshold, device):
        super().__init__(threshold=threshold, device=device)

        # TODO update this code to work for robot sim, currently unused
        # adapted from DCGAN model tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
        self.ndf = 44
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
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
        self.to(self.device)

    def forward(self, states, actions, next_states):
        # put the data on the target compute device
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)

        # remove empty dimensions
        # note this code will not work for batch_size = 1
        states = torch.squeeze(states)
        actions = torch.squeeze(actions)
        next_states = torch.squeeze(next_states)

        # build trajectories and run discriminator - shape (batch_size, in_features)
        trajectories = torch.cat((actions, states, next_states), -1)
        self.ndf = trajectories.shape[-1]

        # input shape is (batch_size, in_features)
        # output shape is (batch_size, 1)
        logits = self.main(trajectories)
        return logits

class NonconvDiscriminator(Discriminator):

    def __init__(self, threshold, device):
        super().__init__(threshold=threshold, device=device)

        # discriminator architecture for trajectories with non-image states
        self.in_features = 44
        self.main = nn.Sequential(
            # 1st layer
            nn.Linear(self.in_features, self.in_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Linear(self.in_features * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.to(self.device)

    def forward(self, states, actions, next_states):
        # put the data on the target compute device
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)

        # for predicted next states (not true next states) take mean over
        # the ensemble of models
        next_states = next_states.mean(dim=1) if len(next_states.shape) == 3 else next_states

        # remove empty dimensions
        states = torch.squeeze(states)
        actions = torch.squeeze(actions)
        next_states = torch.squeeze(next_states)

        # add back a dimension if batch size was 1
        states = torch.unsqueeze(states, dim=0) if len(states.shape) == 1 else states
        actions = torch.unsqueeze(actions, dim=0) if len(actions.shape) == 1 else actions
        next_states = torch.unsqueeze(next_states, dim=0) if len(next_states.shape) == 1 else next_states

        # build trajectories and run discriminator - shape (batch_size, in_features)
        trajectories = torch.cat((actions, states, next_states), -1)
        self.in_features = trajectories.shape[-1]

        # input shape is (batch_size, in_features)
        # output shape is (batch_size, 1)
        logits = self.main(trajectories)
        return logits
