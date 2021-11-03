from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from xgboost import XGBRegressor
from termcolor import colored
from TaxiFareModel.encoders import DistanceToCenter

class Trainer():
    
    ESTIMATOR = "Lasso"
    EXPERIMENT_NAME = "TaxifareModel"
    
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.kwargs = kwargs
        
    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif estimator == "xgboost":
            model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10, learning_rate=0.05,
                                 gamma=3)
            self.model_params = {'max_depth': range(10, 20, 2),
                                 'n_estimators': range(60, 220, 40),
                                 'learning_rate': [0.1, 0.01, 0.05]
                                 }
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    # pipe_time_features = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),
    #                                     OneHotEncoder(handle_unknown='ignore'))
    # pipe_distance = make_pipeline(DistanceTransformer(distance_type=dist, **DIST_ARGS), StandardScaler())
    # pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())
    # pipe_direction = make_pipeline(Direction(), StandardScaler())
    # pipe_distance_to_center = make_pipeline(DistanceToCenter(), StandardScaler())

    # # Define default feature engineering blocs
    # feateng_blocks = [
    #     ('distance', pipe_distance, list(DIST_ARGS.values())),
    #     ('time_features', pipe_time_features, ['pickup_datetime']),
    #     ('geohash', pipe_geohash, list(DIST_ARGS.values())),
    #     ('direction', pipe_direction, list(DIST_ARGS.values())),
    #     ('distance_to_center', pipe_distance_to_center, list(DIST_ARGS.values())),
    # ]
    
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
    ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        dist_to_center_pipe = make_pipeline(DistanceToCenter())
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('distance_to_center', dist_to_center_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude'])
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def run(self, X_train, y_train):
        """set and train the pipeline"""
        return pipe.fit(X_train,y_train)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        rmse_train = self.compute_rmse(self.X_train, self.y_train)
        self.mlflow_log_metric("rmse_train", rmse)
        return rmse
    
    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
        
    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


if __name__ == "__main__":
    # get data
    data = get_data()
    # clean data
    data = clean_data(data)
    # set X and y
    y = data["fare_amount"]
    X = data.drop("fare_amount", axis=1) 
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X,y)
    pipe = trainer.set_pipeline()
    trainer.run(X_train, y_train)
    # evaluate
    print(trainer.evaluate(X_test, y_test))
    # save 
    trainer.save_model()
    
