# DATA MODULE
import folium
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class LinearModelData:
    """
    This class will load a csv just for the purpose of providing data in a dataframe form to a
    linear regression model.

    The class should be strict and should have mainly theree methods:
    - load_data
    - preprocess (this includes any splitting and scaling)
    - report
    """

    def __init__(self, data_path=None):
        self.data_path = data_path # csv path, must be provided or the class fails, lets be strict
        self.data_df = None # dataframe
        self.data_types = None # data types of the columns
        self.features = None # features columns
        self.target = None # target column
        self.features_data = None # input data
        self.features_data_unscaled = None
        self.target_data = None # target data
        self.report = None # report on the data after pre-processing it

    def load_data(self):
        """
        Description
        -----------
            Load the csv file into a dataframe by default last column will be the target value,
            but it can be set using the features column if desired.
        """
        if self.data_path is None:
            raise ValueError("Data path must be provided")
        else:
            self.data_df = pd.read_csv(self.data_path)
            self.data_df.columns = [c.lower() for c in self.data_df.columns]
        return None

    def preprocess(self, split_target_features=True,
                   set_features = None,
                   set_target = None,
                   scale=True,
                   customize = False,
                   debug_print= False,
                   granular_preproces_dict = {"fill_strategy":None,
                                            "categorical_strategy":None,
                                            "scale_strategy":None}):
        """
        Description
        -----------
            Preprocess the data by default doing the following:
            - 1.0 Convert categorical data to numerical data
            - 2.0 Filling missing data
            - 3.0 Splitting it into input and target data, NOT train and test data!!!
            - 4.0 Scaling the input data for gradient descent method of training
        """
        if customize == False:
            # Step 1.0: Find categorical columns (usually object or category dtype) and then convert them to numerical data
            self.data_types = self.data_df.dtypes # get the data types of the columns
            categorical_columns = self.data_types[self.data_types == 'object']
            if len(categorical_columns) > 0:
                # Convert the categorical columns to numerical data
                self.data_df[categorical_columns.index] = self.data_df[categorical_columns.index].astype('category')
                self.data_df[categorical_columns.index] = self.data_df[categorical_columns.index].apply(lambda x: x.cat.codes)
                if debug_print:
                    print("After categorical data conversion")
                    print(self.data_df.head())


            # Step 2.0: fill the missing data with mean values by default, we will add more here later
            self.data_df.fillna(self.data_df.mean(), inplace=True) # fill the missing data with mean for the column

            # Step 3.0: Split the data into input and target data
            if split_target_features:
                if set_features is None: # if you want to set features then use this
                    self.features = self.data_df.columns[:-1]
                    self.features_data = self.data_df[self.features]
                else:
                    self.features_data = self.data_df[set_features]
                if set_target is None:
                    self.target = self.data_df.columns[-1]
                    self.target_data = self.data_df[self.target]
                else:
                    self.target_data = self.data_df[set_target]
            else:
                raise ValueError("Not splitting target and features is not supported yet")

            # Step 4.0 Scale the data using the StandardScalar by default and give other options
            self.features_data_unscaled = self.features_data.copy()
            if scale:
                scaler = StandardScaler() # scaling for making mean to be zero, good for normal distribution.
                cols = self.features_data.columns
                # pop features like latitude and longitudes from it because those do not need scaling
                if 'latitude' and 'longitude' in cols:
                    self.features_data = self.features_data.drop('latitude', axis=1)
                    self.features_data = self.features_data.drop('longitude',  axis=1)
                    cols = self.features_data.columns
                    self.features_data = scaler.fit_transform(self.features_data)

                self.features_data = pd.DataFrame(self.features_data, columns=cols) # convert to dataframe again
                # add latitude and longitude from original
                self.features_data['latitude'] = self.features_data_unscaled['latitude']
                self.features_data['longitude'] = self.features_data_unscaled['longitude']

        else:
            raise NotImplementedError("Customize preprocessing is not implemented yet")
        return None

    def write_report(self, histogram=True, correlation=True, summary_stats= True, mode='pdf', naming_sugar=None, map=False, colormap_on = None, display=False):
        """
        Description
        -----------
            Write a report on the data.
        """
        if histogram == True:
            self.features_data.hist(bins=50, figsize=(20,15))
            plt.savefig(f'{naming_sugar}_data_histogram.'+mode)
            plt.close()
        if correlation == True:
            corr_matrix = self.features_data.corr()
            # Calculate the correlation matrix
            corr = self.features_data.corr()
            # Create the heatmap
            sns.heatmap(corr, annot=True, cmap="viridis")
            plt.savefig(f'{naming_sugar}_data_correlation_matrix.'+mode)
            plt.close()
        if summary_stats == True:
            # Write the DataFrame to a PDF
            # Convert DataFrame to HTML
            html_string = self.features_data.describe().to_html()
            # Save the HTML string to a file
            with open(f'{naming_sugar}_stat.html', 'w') as f:
                f.write(html_string)
        if map:
            # Create a base map
            m = folium.Map(location=[self.features_data_unscaled['latitude'].mean(), self.features_data_unscaled['longitude'].mean()], zoom_start=12)
            # Add scatter markers with colormap
            for index, row in self.features_data_unscaled.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"Location: {row['latitude']} {row['longitude']}",
                    fill=True,
                    color='blue',
                    fill_color='blue'
                ).add_to(m)

            # Display the map
            m.save('map.html')
            if display:
                display(m)

        return None

if __name__ == "__main__":
    lmd = LinearModelData(data_path='california_housing.csv')
    lmd.load_data()
    lmd.preprocess(debug_print=False)
    lmd.write_report(naming_sugar="Cali_data", map=True, colormap_on='housing_median_age')
