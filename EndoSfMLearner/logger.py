import progressbar
import sys


class TermLogger:
    def __init__(self, n_epochs, train_size, valid_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size

        print("\n" * 3)  # Espacio visual
        self.epoch_bar = progressbar.ProgressBar(max_value=n_epochs)
        self.reset_train_bar()
        self.reset_valid_bar()

    def reset_train_bar(self):
        print("\nTrain:")
        self.train_bar = progressbar.ProgressBar(max_value=self.train_size)

    def reset_valid_bar(self):
        print("\nValidation:")
        self.valid_bar = progressbar.ProgressBar(max_value=self.valid_size)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert len(val) == self.meters
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)
