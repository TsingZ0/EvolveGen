from algo.server.ServerBase import ServerBase
from algo.client.ClientBase import ClientBase


class Server(ServerBase):
    def __init__(self, args):
        args.Client = ClientBase
        super().__init__(args)

        
