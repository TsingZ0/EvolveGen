import os
import numpy as np
import torch
from algo.server.ServerBase import ServerBase
from algo.client.ClientBase import ClientBase
from utils.prompts import *
from utils.captioner import get_captioner
from utils.dataset import inv_normalize
from collections import defaultdict


class Server(ServerBase):
    def __init__(self, args):
        args.Client = ClientBase
        super().__init__(args)

        if not args.use_generated:
            self.base_prompt = exemple_prompt
            self.captions = defaultdict(list)
            self.Cap = get_captioner(args)


    def receive(self):
        assert self.args.task_mode == 'T2I'
        self.current_volume_per_label, self.done, data = self.client.send('real')
        for img, y in data:
            if self.args.do_norm:
                img = inv_normalize(img)
            text = self.Cap(img)
            label_name = self.args.label_names[y.item()]
            self.captions[label_name].append(text)
        caption_file_path = os.path.join(self.generated_dataset_dir, 'captions.pt')
        torch.save(self.captions, caption_file_path)
        print('Captioner done.')

    def get_prompt(self, label_name):
        random_idx = np.random.choice(len(self.captions[label_name]), 1)[0]
        prompt = self.base_prompt.format(
            DOMAIN=self.args.domain, 
            LABEL=label_name, 
            EXAMPLE=self.captions[label_name][random_idx]
        )[:self.args.prompt_max_length]
        return prompt
    
    def callback(self):
        if not self.args.use_generated:
            del self.Gen
            del self.Cap