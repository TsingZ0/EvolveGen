from algo.server.ServerBase import ServerBase
from algo.client.ClientBase import ClientBase
from utils.llm import get_llm
from utils.prompts import *


class Server(ServerBase):
    def __init__(self, args):
        args.Client = ClientBase
        super().__init__(args)

        if not args.use_generated:
            self.refine_prompt = refine_prompt
            self.LLM = get_llm(args)


    def get_prompt(self, label_name):
        prompt = self.base_prompt.format(
            DOMAIN=self.args.domain, 
            LABEL=label_name
        )[:self.args.prompt_max_length]
        prompt_refined = self.LLM(self.refine_prompt + prompt)
        print(f'Label name: {label_name}', f'\t\trefined prompt: {prompt_refined}')
        return prompt_refined
    
    def callback(self):
        if not self.args.use_generated:
            del self.Gen
            del self.LLM