from models.uniform import Uniform
from models.simple_first_30_days_fixed_spike import SimpleFirst30DaysFixedSpike
from models.voted_shifted_padded_gaussian_probs import VotedShiftedPaddedGaussianProbs
from models.normal import Normal
from models.tweedie import Tweedie
from models.xgboost_v1 import XGBoostV1
from models.xgboost_features_v1 import XGBoostFeaturesV1
from models.xgboost_features_v2_classification import XGBoostFeaturesV2
from models.xgboost_features_v2_1_classification import XGBoostFeaturesV2_1
from models.xgboost_features_v2_2_all59 import XGBoostFeaturesV2_2
from models.xgboost_features_v3_tsfresh import XGBoostFeaturesV3


models = {
    "uniform": Uniform,
    "simple_first_30_days_fixed_spike": SimpleFirst30DaysFixedSpike,
    "voted_shifted_padded_gaussian_probs": VotedShiftedPaddedGaussianProbs,
    "normal": Normal,
    "tweedie": Tweedie,
    "xgboost_v1": XGBoostV1,
    "xgboost_features_v1": XGBoostFeaturesV1,
    "xgboost_features_v2_classification": XGBoostFeaturesV2,
    "xgboost_features_v2_1_classification": XGBoostFeaturesV2_1,
    "xgboost_features_v2_2_all59": XGBoostFeaturesV2_2,
    "xgboost_features_v3_tsfresh": XGBoostFeaturesV3
}