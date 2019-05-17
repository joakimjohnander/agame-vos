
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.clear()

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 'nan'

    def new_epoch(self):
        self.history.append(self.avg)
        self.reset()

