from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


def get_mae(X, y):
    #multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring='neg_mean_absolute_error').mean()


predictors_without_categoricals = train_predictors.select_dtypes(exclude=[
                                                                 'object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' +
      str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))1
