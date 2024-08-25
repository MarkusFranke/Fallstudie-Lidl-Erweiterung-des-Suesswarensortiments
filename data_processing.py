import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Self, Tuple, List

class CandyDataProcessor:
    def __init__(self, file_path:str, price_percent_weight:float):
        """
        Initialize the CandyDataProcessor with the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing the candy data.
        """
        
        df = pd.read_csv(file_path)
        # Scale to 0-1:
        df['winpercent'] = (df['winpercent'] - min(df['winpercent']))/(max(df['winpercent']) - min(df['winpercent']))
        df['pricepercent'] = (df['pricepercent'] - min(df['pricepercent']))/(max(df['pricepercent']) - min(df['pricepercent']))

    
        assert 0 <= price_percent_weight and price_percent_weight <= 1
        df['desirability'] = (1-df['pricepercent'])**price_percent_weight * df['winpercent']**(1 - price_percent_weight)
        self.df = df
        self.original_df = self.df.copy()

    def sort_by_win_percent(self, ascending=False) -> Self:
        """Sort the DataFrame by 'winpercent' in descending order."""
        self.df = self.df.sort_values(by='winpercent', ascending=ascending)
        return self

    @staticmethod
    def get_predictor_cols():
        """Get a list of the colnames of the characterinstics, i.e. of the features"""
        return ['chocolate','fruity','hard','nougat','crispedricewafer',
                'peanutyalmondy', 'caramel','bar','pluribus']

    def filter_ingredient_columns(self) -> Self:
        """Filter out non-ingredient columns and retain only ingredient-related features."""
        
        self.df = self.df[self.get_predictor_cols()]
        return self

    def get_num_norm_cols(self) -> np.ndarray:
        """Get normalized numerical columns using StandardScaler."""
        scaler = StandardScaler()
        df = self.df.select_dtypes(include=[float, int])
        YX = scaler.fit_transform(df)
        return YX

    def prepare_features_and_target(self, feature_columns: List[str], target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate the DataFrame into features (X) and target (y).
        
        Args:
            feature_columns (array): Array of feature column names.
            target_column (str): Name of the target column.
        
        Returns:
            pd.DataFrame, pd.Series: Features (X) and target (y).
        """
        assert target_column not in feature_columns
        X = self.df[feature_columns]
        y = self.df[target_column]
        return X, y

    def reset(self):
        """Reset the DataFrame to its original state."""
        self.df = self.original_df.copy()
        return self

    def perform_clustering(self, n_clusters:int):
        """Perform clustering on the DataFrame and add cluster labels."""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        distance_matrix = pdist(self.df, metric='euclidean')
        linkage_matrix = linkage(distance_matrix, method='ward')
        self.df['Cluster'] = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return self
