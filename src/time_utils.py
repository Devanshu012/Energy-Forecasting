import pandas as pd

class TimeCleaner:
    """
    Utility class for cleaning and standardizing time columns.
    """
    
    def __init__(self, df: pd.DataFrame, time_col: str = "Time"):
        self.df = df.copy()
        self.time_col = time_col
        
    def fix_24_hour_format(self) -> None:
        """
        Fix timestamps containing ', 24:' by replacing with ', 00:'.
        """
        mask = (
            self.df[self.time_col].astype(str).str.contains(", 24:", regex=False)
        )
        self.df.loc[mask, self.time_col] = (
            self.df.loc[mask, self.time_col].astype(str).str.replace(", 24:", ", 00:", n=1, regex=False)
        )
        
    def to_datetime(self, fmt:str = "%d/%m/%Y, %H:%M:%S") -> None:
        """
        Convert time column to pandas datetime.
        """
        self.df[self.time_col] = pd.to_datetime(
            self.df[self.time_col],
            format=fmt,
            errors="coerce"
        )
        
    def sort_by_time(self) -> None:
        """
        Sort dataframe by time column.
        """
        self.df = self.df.sort_values(self.time_col)
        
    def clean(self) -> pd.DataFrame:
        """
        Run full time cleaning pipeline
        """
        self.fix_24_hour_format()
        self.to_datetime()
        self.sort_by_time()
        return self.df
        