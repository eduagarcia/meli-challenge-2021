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
from models.xgboost_features_v1_2_30models import XGBoostFeaturesV1_2
from models.xgboost_features_v3_1_tsfresh_all import XGBoostFeaturesV3_1
from models.xgboost_features_v3_2_tsfresh_plus_manual import XGBoostFeaturesV3_2
from models.xgboost_features2_v4 import XGBoostFeaturesV4
from models.xgboost_features2_v4_1_target_features import XGBoostFeaturesV4_1
from models.xgboost_features2_v4_2_normalize_features import XGBoostFeaturesV4_2
from models.xgboost_features2_v4_3_eliminate_features import XGBoostFeaturesV4_3
from models.xgboost_features2_v4_4_select_3 import XGBoostFeaturesV4_4
from models.xgboost_features2_v4_5_normalize import XGBoostFeaturesV4_5
from models.xgboost_features2_v4_6_more_target_feature import XGBoostFeaturesV4_6
from models.lightgbm_v1 import LightGBM_V1
from models.xgboost_features2_v4_5_1_fix_dup import XGBoostFeaturesV4_5_1

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
    "xgboost_features_v3_tsfresh": XGBoostFeaturesV3,
    "xgboost_features_v1_2_30models": XGBoostFeaturesV1_2,
    "xgboost_features_v3_1_tsfresh_all": XGBoostFeaturesV3_1,
    "xgboost_features_v3_2_tsfresh_plus_manual": XGBoostFeaturesV3_2,
    "xgboost_features2_v4": XGBoostFeaturesV4,
    "xgboost_features2_v4_1_target_features": XGBoostFeaturesV4_1,
    "xgboost_features2_v4_2_normalize_features": XGBoostFeaturesV4_2,
    "xgboost_features2_v4_3_eliminate_features": XGBoostFeaturesV4_3,
    "xgboost_features2_v4_4_select_3": XGBoostFeaturesV4_4,
    "xgboost_features2_v4_5_normalize": XGBoostFeaturesV4_5,
    "xgboost_features2_v4_6_more_target_feature": XGBoostFeaturesV4_6,
    "lightgbm_v1": LightGBM_V1,
    "xgboost_features2_v4_5_1_fix_dup": XGBoostFeaturesV4_5_1,
}