from logger import logger


class EarlyStopping:
    """Simple early stopper"""

    def __init__(self, patience: int = 30, best_score: float = 0.) -> None:
        """Initialize Early stopping class
        :param patience: early stopping patience
        :param best_score: current best score to update (larger is better)
        """
        self.best_score = best_score  # i.e. F1, F2, mAP or any other scalar metric
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after score stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch: int, score: float) -> bool:
        """Decide stop early or not
        :param epoch: training epoch
        :param score: score of current epoch
        :returns: bool to whether finish training or not
        """
        if score >= self.best_score:  # >= 0 to allow for early zero-score stage of training
            logger.info(f"New best score {score} from last best {self.best_score}")
            self.best_epoch = epoch
            self.best_score = score
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= self.patience - 1  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        logger.info(f"Early stopper patience counter: {delta}/{self.patience}. Current best: {self.best_score}")
        if stop:
            logger.info(f"EarlyStopping patience {self.patience} exceeded, stopping training.")
            logger.info(f"Best score: {self.best_score}")
        return stop
