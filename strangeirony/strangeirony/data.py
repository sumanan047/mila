from sklearn.datasets import fetch_california_housing
import pandas as pd

# methods
# 1. generate test data
# 2. 

def gen_test_data(data_type=None):
    """
    Description:
    ============
        Generates test data for machine learning
    
    Parameters:
    ============
        data_type: None
    """
    if data_type == "lr":
        housing_data = fetch_california_housing()
        df_features = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
        df_prices = pd.DataFrame(housing_data.target, columns=['Price'])
        df = pd.concat([df_features,df_prices],axis=1)
        df.to_csv('california_housing.csv')
    else:
        raise NotImplementedError('No such data can be created!')




if __name__ == "__main__":
    # Explore the data
    gen_test_data(data_type='lr')