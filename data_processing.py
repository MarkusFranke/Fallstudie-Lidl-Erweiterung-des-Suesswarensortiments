import pandas as pd
from sklearn.preprocessing import StandardScaler

class CandyDataProcessor:
    def __init__(self, file_path):
        """
        Initialize the CandyDataProcessor with the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing the candy data.
        """
        df = pd.read_csv(file_path)
        df['winpercent'] = df['winpercent']/100
        self.df = df
        self.original_df = self.df.copy()


    def sort_by_win_percent(self, ascending=False):
        """Sort the DataFrame by 'winpercent' in descending order."""
        return self.df.sort_values(by='winpercent', ascending=ascending)

    @staticmethod
    def get_predictor_cols():
        return ['chocolate','fruity','hard','nougat','crispedricewafer',
                'peanutyalmondy', 'caramel','bar','pluribus']

    def filter_ingredient_columns(self):
        """Filter out non-ingredient columns and retain only ingredient-related features."""
        return self.df[self.get_predictor_cols()]

    def get_num_norm_cols(self):
        """Normalize specified numerical columns using StandardScaler."""
        scaler = StandardScaler()
        df = self.df.select_dtypes(include=[float, int])
        df = scaler.fit_transform(df)
        return df

    def prepare_features_and_target(self, feature_columns, target_column):
        """
        Separate the DataFrame into features (X) and target (y).
        
        Args:
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

    def perform_clustering(self, n_clusters):
        """Perform clustering on the DataFrame and add cluster labels."""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        distance_matrix = pdist(self.df, metric='euclidean')
        linkage_matrix = linkage(distance_matrix, method='ward')
        self.df['Cluster'] = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return self
