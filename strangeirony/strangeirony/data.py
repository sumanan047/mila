from sklearn.datasets import fetch_california_housing
import pandas as pd
import os
import matplotlib.pyplot as plt

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

def load_csv(file):
    """
    Description:
    ============
        For now just load the csv simply into a dataframe
    
    Keyword arguments:
     -- description
    Return: return_description
    """
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        raise FileExistsError('Could not find the file provided')


def invalid_count(df):
    na_count = (df.isna().sum(axis=0)).sum()
    null_count = (df.isnull().sum(axis=0)).sum()
    if (na_count == 0) and (null_count == 0):
        print("CLEAN DATA !!")
        print("==============")
        print('No null or nan in the dataframe')
        return 0
    else:
        raise ValueError('The data has invalid nan or null values')

def gen_histogram(df,save=True, figsize=(10,35),filename='histogram.pdf'):
    fig, axes = plt.subplots(nrows=len(df.columns),ncols=1,figsize=figsize)
    # Plot each column
    for i, col in enumerate(df.columns):
        df[col].plot(kind='hist', ax=axes[i], title=col, alpha=0.3,label=col, bins=100,range=(df[col].min(), df[col].max()))
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}',zorder=15)
        axes[i].axvline(df[col].median(), color='green', linestyle='--', label=f'Mean: {df[col].median():.2f}',zorder=15)
        axes[i].legend()
    plt.tight_layout()
    if save ==True:
        plt.savefig(filename)
    else:
        pass
    plt.show()
    return None

if __name__ == "__main__":
    # Explore the data
    gen_test_data(data_type='lr')
    # load data
    df = load_csv(file='california_housing.csv')
    # check if data has na or null data, if it does do pre-processing seprate and then load
    # only good data
    invalid_count(df=df)