import numpy as np
import torch
from algo.client.rater.RaterBase import Client as Rater
from collections import defaultdict


class Client(Rater):
    def __init__(self, args):
        super().__init__(args)

        self.sensitivity = 1
        self.delta = 1 / (self.args.real_volume_per_label * self.args.num_labels)
        self.sigma = self.sensitivity * np.sqrt(2 * np.log(
            1.25 / self.delta)) / self.args.epsilon


    @torch.no_grad()
    def rater(self, filtered_dataset):
        self.ref_imgs_prob = defaultdict(list)
        self.model.eval()
        assert self.args.histogram_threshold < self.args.real_volume_per_label
        histogram = [
            [
                0 for _ in range(self.args.volume_per_label - self.current_volume_per_label[i])
            ] for i in range(self.args.num_labels)
        ]
        
        train_vecs = [[] for _ in range(self.args.num_labels)]
        train_loader = self.load_train_dataset(is_shuffle=False, batch_size=1)
        for x, y in train_loader:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            vecs = self.encoder(x).detach()
            for vec, yy in zip(vecs, y):
                yc = yy.item()
                train_vecs[yc].append(vec.detach().data)

        # get histogram
        for vecs in self.real_vecs:
            for vec in vecs:
                self.update_histogram(yc, histogram, vec, train_vecs)
        
        # Gaussian mechanism
        # Post-processing (threshold)
        for i in range(self.args.num_labels):
            noise = np.random.normal(size=len(histogram[i])) * self.sigma
            for j in range(len(histogram[i])):
                histogram[i][j] = histogram[i][j] + noise[j].item()
                histogram[i][j] = max(histogram[i][j] - self.args.histogram_threshold, 0)

        # rater
        # get filtered_dataset
        last_yc = None
        idxs = [0 for _ in range(self.args.num_labels)]
        for x, y in train_loader:
            for xx, yy in zip(x, y):
                yc = yy.item()
                if self.current_volume_per_label[yc] < self.args.volume_per_label and histogram[yc][idxs[yc]] > 0:
                    filtered_dataset[yc].append((xx, yy))
                    self.current_volume_per_label[yc] += 1
                    last_yc = yc
                    imp = histogram[yc][idxs[yc]]
                    self.ref_imgs_prob[yc].append(imp)
                else:
                    self.ref_imgs_prob[yc].append(0)
                idxs[yc] += 1
            if self.check_done():
                self.done = True
                break

        for k, v in self.ref_imgs_prob.items():
            if sum(v) == 0:
                self.ref_imgs_prob[k] = [1 for _ in v]
                print(f'Random selection for class {k}.')

        return filtered_dataset, last_yc
        

    def update_histogram(self, yc, histogram, real_vec, train_vecs):
        min_idx = -1
        min_dist = float('inf')
        for idx, vec in enumerate(train_vecs[yc]):
            dist = self.calculate_dist(real_vec, vec)
            if dist < min_dist:
                min_idx = idx
                min_dist = dist
        if min_idx >= 0:
            histogram[yc][min_idx] += 1
