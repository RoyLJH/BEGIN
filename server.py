class Server(object):
    def __init__(self, args):
        pass

    def optimize(self):
        # for each iteration, send current input to each worker and wait all workers update their gradients
        print("Optimize finished")
