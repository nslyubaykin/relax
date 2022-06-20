import numpy as np
import matplotlib.pyplot as plt

from math import log
from random import shuffle


def learner_id(learner):
    return f'{type(learner).__name__}_{id(learner)}'


def sample_unique(size, sample):

    sample_size = 0
    out_sample = []

    if sample > size:
        raise AssertionError('Sample size cannot be greater than total size')

    while sample_size < sample:
        out_sample += list(np.random.choice(size, sample, replace=True))
        out_sample = list(set(out_sample))
        sample_size = len(out_sample)
        
    shuffle(out_sample)

    return np.array(out_sample[:sample])


class TreeNode:
    
    def __init__(self,
                 p=None,
                 p_alpha=0,
                 n_visits=0,
                 buffer_index=None,
                 is_leaf=False):
        
        self.p = p
        self.p_alpha = p_alpha
        self.n_visits = n_visits
        self.buffer_index = buffer_index
        self.is_leaf = is_leaf
        
    def reset(self):
        self.p = None
        self.p_alpha = 0
        self.n_visits = 0
        self.buffer_index = None
        self.is_leaf = False
        
        
class UniformReference:

    def __init__(self, size):
        
        self.size = size
        self.tree = []
        
        self.n_transitions = 0
        self.n_paths_evicted = 0
        
        self.buffer_reference = []
        
    def append(self, 
               value=None,
               buffer_index=None,
               return_index=False):
        
        add_index = self.n_transitions
        
        node = TreeNode()
        node.is_leaf = True
        node.buffer_index = buffer_index
        self.tree.append(node)
        
        self.n_transitions += 1
        
        if return_index:
            return add_index
    
    def evict(self, i):
        
        if not isinstance(i, tuple):
        
            # Swap places with the last and delete it
            self.tree[i] = self.tree.pop()

            # Reflect changes in buffer reference
            p_ind, tr_ind = self.tree[i].buffer_index
            p_ind -= self.n_paths_evicted

            self.buffer_reference[p_ind][tr_ind] = i

            self.n_transitions -= 1
    
    def sample(self, k):
        
        tree_ind = sample_unique(sample=k, 
                                 size=self.n_transitions)
        
        buffer_ind_out = []
        
        for ti in tree_ind:
            node = self.tree[ti]
            if node.is_leaf:
                p_ind, tr_ind = node.buffer_index
                buffer_ind_out.append((p_ind-self.n_paths_evicted, tr_ind))
                node.n_visits += 1
            
        return buffer_ind_out
        
    def add_batch(self, 
                  batch_size):
        
        pushed, searched = 0, 0
        p_ind, tr_ind = -1, -1
        ref_len = len(self.buffer_reference)
        
        while pushed < batch_size:

            ref_p = self.buffer_reference[p_ind]
            ref_i = ref_p[tr_ind]
            ref_p_len = len(ref_p)

            if isinstance(ref_i, tuple):
                self.buffer_reference[p_ind][tr_ind] = self.append(
                    buffer_index=ref_i,
                    return_index=True
                )
                pushed += 1
                
            searched += 1

            # print(pushed, searched)

            if abs(tr_ind) < ref_p_len:
                tr_ind -= 1
            else:
                if abs(p_ind) < ref_len: # no paths left in a reference
                    p_ind -= 1
                    tr_ind = -1
                else:
                    break
                    
    def evict_last_paths(self, n_paths):
        if n_paths > 0:
            for p in range(n_paths):
                indices_to_evict = self.buffer_reference[p]
                for i_to_evict in indices_to_evict:
                    self.evict(i=i_to_evict)
            # remove from reference also
            self.buffer_reference = self.buffer_reference[n_paths:]
            self.n_paths_evicted += n_paths
            
    def plot_priority(self):
        
        n_visits = [n.n_visits for n in self.tree]

        never_seen = ((np.array(n_visits)) == 0).sum()
        never_seen_perc = round(
            never_seen / self.n_transitions * 100,
            2
        )

        plt.hist(n_visits, 
                 bins=min(50, 
                          int(self.n_transitions / 100)))
        plt.title("Number of visits. "
                  f"Never seen: {never_seen}, "
                  f"({never_seen_perc} %)")
        plt.show()
        
        
class SumTree(UniformReference):
    
    def __init__(self, size, 
                 init_max=1,
                 init_alpha=1,
                 recalc_tree_every=int(3e6)):
        
        self.size = size
        self.last_parent = size-1
        self.tree_size = 2 * size
        
        self.tree = [TreeNode() for _ in range(self.tree_size)]
        self.empty = list(range(self.size, self.tree_size))
        
        self.max_val = init_max
        self.last_alpha = init_alpha
        
        self.n_transitions = 0
        self.n_propagations = 0
        self.n_paths_evicted = 0
        self.recalc_tree_every = recalc_tree_every
        
        self.buffer_reference = []
        
    def _propagate(self, i, change):     
        
        while i // 2 > 0:
            self.tree[i // 2].p_alpha += change
            i = i // 2
            
    def _retrieve(self, s):
        
        assert s <= self.total
        
        i=1
        while i <= self.last_parent:
            left_child = self.tree[2*i].p_alpha
            if s < left_child:
                i = 2 * i
            else:
                s -= left_child
                i = 2 * i + 1
                
        return i
    
    def _recalc_sums(self):
            
        if self.n_propagations > 0 and self.n_propagations % self.recalc_tree_every == 0:
        
            for i in range(self.size):
                self.tree[i].reset()

            for i in range(self.size, self.tree_size):
                node = self.tree[i]
                if node.is_leaf:
                    self._propagate(i=i,
                                    change=node.p_alpha)
    
    def append(self, 
               value=None,
               buffer_index=None,
               return_index=False):
        
        if value is not None:
            if value > self.max_val:
                self.max_val = value
        else:
            value = self.max_val
        
        add_index = self.empty.pop()
        value_alpha = value**self.last_alpha
        
        node = self.tree[add_index]
        
        node.p = value
        node.p_alpha = value_alpha
        node.is_leaf = True
        node.buffer_index = buffer_index
        
        self._propagate(i=add_index, change=value_alpha)
        self.n_transitions += 1
        
        # reset
        self.n_propagations += 1
        
        # recalc if needed
        self._recalc_sums()
        
        if return_index:
            return add_index
    
    def change(self, i, 
               newvalue,
               alpha=1):
        
        assert i > self.last_parent # only leafs can be changed
        
        if newvalue > self.max_val:
            self.max_val = newvalue
        
        newvalue_alpha = newvalue**alpha
        oldvalue_alpha = self.tree[i].p_alpha
        change = newvalue_alpha - oldvalue_alpha
        
        self.tree[i].p = newvalue
        self.tree[i].p_alpha += change
        
        self._propagate(i=i, change=change)
        
        # reset
        self.n_propagations += 1
        
        # recalc if needed
        self._recalc_sums()
        
    def evict(self, i):
        # check this also
        # assert i not in self.empty
        
        if not isinstance(i, tuple):
            
            self.change(i=i, 
                        newvalue=0,
                        alpha=1)
            self.tree[i].reset()
            self.empty.append(i)
        
            self.n_transitions -= 1
        
    def sample(self, k):
        
        samples = np.random.uniform(self.total, size=k)        
        tree_ind = [self._retrieve(s=s) for s in samples]
        
        buffer_ind_out, tree_ind_out, p_alpha_out = [], [], []
        
        for ti in tree_ind:
            node = self.tree[ti]
            if node.is_leaf:
                p_ind, tr_ind = node.buffer_index
                buffer_ind_out.append((p_ind-self.n_paths_evicted, tr_ind))
                tree_ind_out.append(ti)
                p_alpha_out.append(node.p_alpha)
                node.n_visits += 1
            
        return tree_ind_out, buffer_ind_out, p_alpha_out
    
    def update_priorities(self, tree_ind,
                          p, alpha):
        
        for ti, pi in zip(tree_ind, p):
            self.change(i=ti,
                        newvalue=pi,
                        alpha=alpha)
        
        self.last_alpha = alpha
    
    def plot_priority(self, figsize=(12,12), nbins=50):
        
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)

        # nbins = min(p_bins, int(self.n_transitions / 1000))
        alpha, p_alpha, p, n_visits = [], [], [], []
        never_seen = 0
        
        for n in self.tree:
            
            if n.is_leaf and n.p is not None:
                
                p.append(n.p)
                p_alpha.append(n.p_alpha)
                
                if n.p != 0 and n.p != 1:
                    alpha.append(log(n.p_alpha, n.p))
                    
                n_visits.append(n.n_visits)
                
                if n.n_visits == 0:
                    never_seen += 1
                
        mean_alpha = np.mean(alpha)
        mean_alpha = round(mean_alpha, 3)
        
        never_seen_perc = round(
            never_seen / self.n_transitions * 100,
            2
        )

        ax1.hist(p_alpha, bins=nbins, 
                 label='p_alpha', 
                 alpha = 0.5)
        
        ax1.hist(p, 
                 bins=nbins, 
                 label='p', 
                 alpha = 0.5)
        
        ax1.legend()
        ax1.title.set_text("Prioritization weights. "
                           f"Total trans.: {len(p)}, "
                           f"Mean alpha: {mean_alpha}")
        ax1.set_xlabel('Priority value')
        ax1.set_ylabel('Frequency')


        ax2.hist(n_visits, 
                 bins=nbins, 
                 label='n_visits')
        
        ax2.legend()
        ax2.title.set_text("Number of visits. "
                           f"Never seen: {never_seen}, "
                           f"({never_seen_perc} %)")
        ax2.set_xlabel('Number of visits')
        ax2.set_ylabel('Frequency')


        corr = round(np.corrcoef(p_alpha, n_visits)[0, 1], 3)
        z = np.polyfit(p_alpha, n_visits, 1)
        p = np.poly1d(z)

        ax3.scatter(p_alpha, n_visits)
        ax3.title.set_text("Priority VS Number "
                           f"of visits. Corr: {corr}")
        ax3.set_xlabel('Priority (p_alpha)')
        ax3.set_ylabel('Number of visits')
        ax3.plot(p_alpha, p(p_alpha),"r--")

        plt.show()
        
    @property
    def total(self):
        return self.tree[1].p_alpha
    