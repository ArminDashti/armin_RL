# From https://github.com/aviralkumar2907/BEAR/blob/master/algos.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleCritic(nn.Module):
    """ Critic which does have a network of 4 Q-functions"""
    def __init__(self, num_qs, state_dim, action_dim):
        super(EnsembleCritic, self).__init__()
        
        self.num_qs = num_qs

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        # self.l7 = nn.Linear(state_dim + action_dim, 400)
        # self.l8 = nn.Linear(400, 300)
        # self.l9 = nn.Linear(300, 1)

        # self.l10 = nn.Linear(state_dim + action_dim, 400)
        # self.l11 = nn.Linear(400, 300)
        # self.l12 = nn.Linear(300, 1)

    def forward(self, state, action, with_var=False):
        all_qs = []
        
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)   # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    def q_all(self, state, action, with_var=False):
        all_qs = []
        
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs
