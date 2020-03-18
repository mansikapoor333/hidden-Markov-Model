from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for q in range(S):
            alpha[q, 0] = self.pi[q] * self.B[q, self.obs_dict[Osequence[0]]]
  
        for r in range(1, L):
            for q in range(S):
                alpha[q, r] = self.B[q, self.obs_dict[Osequence[r]]] * sum([self.A[i, q] * alpha[i, r - 1]for i in range(S)])
        ###################################################
        return alpha

    def backward(self, Osequence):
        
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        for q in range(S):
            beta[q, L - 1] = 1
  
        for r in reversed(range(L - 1)):
            for i in range(S):
                beta[i, r] = sum([beta[q, r + 1] * self.A[i, q] * self.B[q, self.obs_dict[Osequence[r + 1]]] for q in range(S)])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):

        prob = 0
        ###################################################
        # Edit here
        theta=self.forward(Osequence)
        prob = sum(theta[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):

        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        abc=self.backward(Osequence)
        theta=self.forward(Osequence)
        prob1 = sum(theta[:, -1])
        i=0
        while i<S:
        #for i in range(S):
            for j in range(L):
                xyz=theta[i,j]*abc[i,j]
                prob[i,j]=xyz/prob1 
            i+=1    
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):

        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha =self.forward(Osequence)
        beta =self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        
        for t in range(L - 1 ):
            for j in range(S):
                for k in range(S):
                    prob[j][k][t]= [self.A[j , k].T * alpha[j, t] * self.B[k , self.obs_dict[Osequence[t+1]]] * beta[ k, t + 1]]/ seq_prob
            
        
        ###################################################
        return prob

    def viterbi(self, Osequence):

        path = []
        ###################################################
        # Q3.3 Edit here
        th = np.zeros([len(self.pi), len(Osequence)])
        de = np.zeros([len(self.pi), len(Osequence)])

        i = 0
        while i<len(self.pi):
            tempo = self.obs_dict[Osequence[0]]
            th[i][0] = self.B[i][tempo]*self.pi[i]
            i=i+1

        for i in range(1, len(Osequence)):
            for j in range(len(self.pi)):

                t= [th[k][i - 1]*self.A[k][j] for k in range(len(self.pi))]

                t = np.array(t)

                tempo = self.obs_dict[Osequence[i]]
                th[j][i] = np.max(t) * self.B[j][tempo]
                de[j][i] = np.argmax(t)

        integral_max = int(np.argmax(th.T[len(Osequence) - 1]))
        path.append(integral_max)

        for i in range(len(Osequence) - 1,0,-1):
            path.insert(0, int(de[path[0]][i]))

        inv_map = {}
        for k, v in self.state_dict.items():
            inv_map[v] =k

        for p in range(len(path)):
            path[p] = inv_map[path[p]]
        ###################################################
        return path
