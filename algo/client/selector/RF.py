import torch
from algo.client.selector.SelectorBase import Client as Selector
from collections import defaultdict


class Client(Selector):
    def __init__(self, args):
        super().__init__(args)


    @torch.no_grad()
    def selector(self, filtered_dataset):
        self.model.eval()

        train_loader = self.load_train_dataset(is_shuffle=False, batch_size=1)
        last_yc = None
        for x, y in train_loader:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            vecs = self.encoder(x).detach()
            for vec, xx, yy in zip(vecs, x, y):
                yc = yy.item()
                if self.current_volume_per_label[yc] < self.args.volume_per_label and self.check_close(vec, yc):
                    filtered_dataset[yc].append((xx, yy))
                    self.current_volume_per_label[yc] += 1
                    last_yc = yc
            if self.check_done():
                self.done = True
                break
        return filtered_dataset, last_yc
    
    def check_close(self, vec, label):
        for l in range(self.args.num_labels):
            if l != label:
                for v in self.real_vecs[l]:
                    if self.calculate_dist(vec, v) < self.args.dist_threshold:
                        return False
        return True