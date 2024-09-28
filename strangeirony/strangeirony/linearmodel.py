# MODEL MODULE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class LinearRegressionModelSelector:
    def __init__(self, X, y, test_size=0.15,validation_size = 0.10, random_state=42):
        self.X = X # feature data
        self.y = y # target data
        self.X_train = None # training split
        self.X_test = None # test split
        self.y_train = None
        self.y_test = None
        self.y_pred = None # predictions
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state
        self.model = None # save the best model here
        self.score = None

    def split_data(self):
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        return None

    def train_and_evaluate(self, model_class, param_grid, cv = 5, verbose = 1):
        # model = model_class()
        grid_search = GridSearchCV(model_class, param_grid, cv=cv, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        self.y_pred = best_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return best_model, np.sqrt(mse), r2

    def select_best_model(self, cv = 15, verbose = 1):
        models = [LinearRegression(),
                  Ridge(),
                  Lasso(),
                  DecisionTreeRegressor(),
                  RandomForestRegressor(),
                  GaussianProcessRegressor(),
                ]
        param_grids = [
            {},  # Default parameters for LinearRegression
            {'alpha': [0.01, 0.1, 1, 10, 100]},  # Hyperparameters for Ridge
            {'alpha': [0.01, 0.1, 1, 10, 100]}, # Hyperparameters for Lasso
            {}, # Default parameters for DecisionTreeRegressor
            {}, # Default parameters for RandomForestRegressor
            {}, # Default parameters for GaussianProcessRegressor
        ]
        best_model = None
        best_mse = float('inf')
        best_r2 = 0
        for model_class, param_grid in zip(models, param_grids):
            model, mse, r2 = self.train_and_evaluate(model_class, param_grid, cv = cv, verbose = verbose)
            if mse < best_mse:
                best_model = model
                best_mse = mse
                best_r2 = r2
        self.model = best_model
        self.score = {"mean-square-error": best_mse, "r2-score":best_r2}
        return None

    def save_model(self,name=None):
        # Save the model
        if name is None:
            raise ValueError('Must provide a name for the model to be saved.')
        else:
            joblib.dump(self.model, f'{name}.pkl')

if __name__ == "__main__":
    from lineardata import LinearModelData
    #data
    lmd = LinearModelData(data_path='strangeirony/tests/california_housing.csv')
    lmd.load_data()
    lmd.preprocess(debug_print=False)
    lmd.write_report(naming_sugar="Cali_data", map=False, colormap_on='housing_median_age')
    # model
    model_selector = LinearRegressionModelSelector(lmd.features_data, lmd.target_data)
    model_selector.split_data()
    model_selector.select_best_model(cv=2, verbose =1)
    model_selector.save(name='california_housing')
    #report
    print("Best Model:", lmd.best_model)
    print("Best MSE:", lmd.score["mean-square-error"])
    print("Best R2 Score:", lmd.score["r2-score"])