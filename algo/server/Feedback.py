import copy
import gc
import os
import numpy as np
import torch
import ujson
from algo.server.Filter import Server as Filter
from utils.dataset import inv_normalize
from collections import defaultdict
from utils.dataset import preprocess_image 
from utils.generator import get_generator 


class Server(Filter):
    def __init__(self, args):
        args.random_gen = True
        super().__init__(args)

        self.ref_imgs_prob = defaultdict(list)
        self.previous_volume_per_label = []


    def receive(self):
        assert self.args.task_mode == 'I2I'
        if self.it > 0:
            self.ref_imgs_prob = defaultdict(list)
            previous_dir = os.path.join(self.train_dataset_dir, f'{self.it-1}')
            self.previous_volume_per_label = copy.deepcopy(self.current_volume_per_label)
            self.current_volume_per_label, self.done, self.ref_imgs_prob = self.client.send('rated')
            with open(os.path.join(previous_dir, 'ref_imgs_prob.json'), 'w') as f:
                ujson.dump(self.ref_imgs_prob, f)
            
            if self.it == 1:
                print('\nReload I2I generator...')
                del self.Gen
                torch.cuda.empty_cache()
                gc.collect()
                self.args.random_gen = False
                self.Gen = get_generator(self.args)


    def get_img(self, label_name):
        label_id = self.args.label_names.index(label_name)
        imgs_prob = self.ref_imgs_prob[label_id]
        if len(imgs_prob) > 0:
            previous_dir = os.path.join(self.generated_dataset_dir, f'{self.it-1}')
            sampled_idx = np.random.choice(len(imgs_prob), 1, p=imgs_prob)[0].item()
            offset = self.previous_volume_per_label[label_id]
            file_name = f'[{label_name}]-{offset + sampled_idx}.jpg'
            if self.args.online_api:
                with open(os.path.join(previous_dir, 'image_urls.json'), 'r') as f:
                    image_urls_dict = ujson.load(f)
                return image_urls_dict[file_name]
            else:
                file_path = os.path.join(previous_dir, file_name)
                random_img = preprocess_image(self.args, file_path)
                if self.args.do_norm:
                    random_img = inv_normalize(random_img)
                return random_img
        else:
            return None
        
    def callback_per_iter(self):
        self.args.i2i_strength -= self.args.i2i_strength_anneal
        self.args.i2i_strength = max(
            self.args.i2i_strength, self.args.i2i_strength_threshold)
        print('Next I2I strength:', self.args.i2i_strength)