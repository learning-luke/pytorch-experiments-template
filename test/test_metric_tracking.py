import torch

torch.manual_seed(0)


def test_compute_accuracy():
    """
    def compute_accuracy(logits, targets):
        acc = (targets == logits.argmax(-1)).float().detach().cpu().numpy()
        return np.mean(acc)
    """
    from utils.metric_tracking import compute_accuracy

    batch_size = 128
    n_classes = 1000
    logits = torch.randint(low=0, high=batch_size, size=(batch_size, n_classes))
    targets = torch.zeros(batch_size)

    for i, logit in enumerate(logits):
        targets[i] = torch.argmax(logit)

    assert compute_accuracy(logits, targets) == 100.0


def test_push():
    """
    def push(self, epoch, iteration, logits, targets):
        self.metrics["epochs"].append(epoch)
        self.metrics["iterations"].append(iteration)
        for k, fnc in self.metrics_to_track.items():
            self.metrics[k].append(fnc(logits, targets))
    """
    pass


def test_metric_tracker():
    """
    A neat test to do here would be to feed a sequence of predictions to the
    metric tracker and verify that the histories are correct.
    """
    pass
