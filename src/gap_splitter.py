import pandas as pd
import numpy as np

class LargeGapSplitter:
    """
    Split large time gap creating a new mid point row for time and redistributing KWH_diff values.
    """
    INVALIDATE_COLS = [
        "AVG_CURRENT",
        "AVG_V_LL",
        "AVG_V_LN",
        "FREQUENCY",
    ]
    
    def __init__(self, df: pd.DataFrame, threshold_hours: float = 1.0):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a Dattime index")
        
        self.df = df.copy()
        self.threshold = threshold_hours
        
    def _split_once(self) -> bool:
        """
        Perform ONE pass of large-gap splitting.
        Returns True if any change was made.
        """
        self.df = self.df.sort_index()
        
        gap_hours = (
            self.df.index.to_series()
            .diff()
            .dt.total_seconds()
            .div(3600)
        )
        
        rows_to_add = []
        
        for i in range(1, len(self.df)):
            gap = gap_hours.iloc[i]
            
            if gap > self.threshold:
                prev_time = self.df.index[i - 1]
                curr_time = self.df.index[i]
                
                midpoint = prev_time + (curr_time - prev_time) / 2
                
                energy_half = self.df.iloc[i]["KWH_diff"] / 2
                
                # halved current row energy
                self.df.at[curr_time, "KWH_diff"] = energy_half
                
                # create midpoint row
                new_row = self.df.loc[curr_time].copy()
                new_row.name = midpoint
                new_row["KWH_diff"] = energy_half
                
                # invalidate non-energy feratures
                for col in self.INVALIDATE_COLS:
                    if col in new_row:
                        new_row[col] = np.nan
                    
                rows_to_add.append(new_row)
                
        if not rows_to_add:
            return False
        
        self.df = (
            pd.concat([self.df, pd.DataFrame(rows_to_add)])
            .sort_index()
        )
        
        return True
    
    def run(self) -> pd.DataFrame:
        """
        Repeatedly split gaps until no large gaps remain.
        """
        while True:
            changed = self._split_once()
            if not changed:
                break
            
        return self.df