
from dataclasses import dataclass


@dataclass(frozen=True)
class Binomial:
    n_pos_: int
    total_: int

    def __post_init__(self):
        assert self.n_pos_ >= 0
        assert self.total_ >= 0
        assert self.n_pos_ <= self.total_

    def n(self):
        return self.total_

    def pos(self):
        return self.n_pos_

    def neg(self):
        return self.total_ - self.n_pos_

    def p(self):
        return self.n_pos_ / self.total_


