import torch
from torch import nn

class Scholar(nn.Module):

    def __init__(self, generator, solver, stats = None, class_idx = None, temp = None):

        super().__init__()
        self.generator = generator
        self.solver = solver

        # get class keys
        if class_idx is not None:
            self.class_idx = list(class_idx)
            self.layer_idx = list(self.stats.keys())
            self.num_k = len(self.class_idx)


    def sample(self, size, allowed_predictions=None, return_scores=False):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # sample images
        x = self.generator.sample(size)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        # set model back to its initial mode
        self.train(mode=mode)

        return (x, y, y_hat) if return_scores else (x, y)

    def generate_scores(self, x, allowed_predictions=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat