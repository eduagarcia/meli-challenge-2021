from models.uniform import Uniform
from models.simple_first_30_days_fixed_spike import SimpleFirst30DaysFixedSpike
from models.voted_shifted_padded_gaussian_probs import VotedShiftedPaddedGaussianProbs

models = {
    "uniform": Uniform,
    "simple_first_30_days_fixed_spike": SimpleFirst30DaysFixedSpike,
    "voted_shifted_padded_gaussian_probs": VotedShiftedPaddedGaussianProbs,
}