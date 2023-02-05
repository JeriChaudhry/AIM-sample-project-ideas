import numpy as np


class MyEarlyStopper:
    def __init__(self, patience=1, tolerance=0):
        self.patience = (
            patience  # How many epochs in a row the model is allowed to underperform
        )
        self.tolerance = tolerance  # How much leeway the model has (i.e. how close it can get to underperforming before it is counted as such)
        self.epoch_counter = 0  # Keeping track of how many epochs in a row were failed
        self.max_validation_acc = np.NINF  # Keeping track of best metric so far

    def should_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.epoch_counter = 0
        elif validation_acc < (self.max_validation_acc - self.tolerance):
            self.epoch_counter += 1
            if self.epoch_counter >= self.patience:
                return True
        return False
