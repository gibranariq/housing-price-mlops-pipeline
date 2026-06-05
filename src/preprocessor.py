import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Setup Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AmesPreprocessor")

class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Definisikan mapping untuk ordinal encoding sesuai dengan notebook
        self.qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        self.bsmt_exp_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        self.bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
        self.gar_fin_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        self.shape_map = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
        self.slope_map = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
        self.fence_map = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
        
        self.ordinal_configs = {
            'ExterQual': self.qual_map, 'ExterCond': self.qual_map, 'BsmtQual': self.qual_map, 
            'BsmtCond': self.qual_map, 'HeatingQC': self.qual_map, 'KitchenQual': self.qual_map, 
            'FireplaceQu': self.qual_map, 'GarageQual': self.qual_map, 'GarageCond': self.qual_map, 
            'PoolQC': self.qual_map, 'BsmtExposure': self.bsmt_exp_map, 'BsmtFinType1': self.bsmt_fin_map,
            'BsmtFinType2': self.bsmt_fin_map, 'GarageFinish': self.gar_fin_map, 'LotShape': self.shape_map, 
            'LandSlope': self.slope_map, 'Fence': self.fence_map
        }
        
        self.intentional_na_cols = [
            'PoolQC', 'Alley', 'Fence', 'MiscFeature', 'MasVnrType', 'GarageType', 
            'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
        ]

    def fit(self, X, y=None):
        # Karena ini manual data cleaning & rule-based engineering, 
        # kita gak butuh belajar parameter apa pun dari data train.
        logger.info("Initializing fit step for AmesFeatureEngineer (No parameters to learn).")
        return self

    def transform(self, X):
        logger.info(f"Starting data transformation pipeline for {X.shape[0]} rows.")
        df = X.copy()
        
        # 1. Drop Fitur Bising
        cols_to_drop = ['Id', 'Utilities']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        logger.info("Dropped unnecessary columns (Id, Utilities).")
        
        # 2. Handle Intentional NA
        for col in self.intentional_na_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')
        logger.info("Handled missing values for intentional NA categorical features.")
        
        # 3. Sinkronisasi Data Hilang (Ranjau MasVnr, Basement, & Garage)
        if 'MasVnrType' in df.columns and 'MasVnrArea' in df.columns:
            df.loc[df['MasVnrType'] == 'None', 'MasVnrArea'] = 0
            
        if 'GarageType' in df.columns:
            if 'GarageCars' in df.columns:
                df.loc[df['GarageType'] == 'None', 'GarageCars'] = 0
            if 'GarageArea' in df.columns:
                df.loc[df['GarageType'] == 'None', 'GarageArea'] = 0

        bsmt_numeric_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
        for col in bsmt_numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if 'Functional' in df.columns:
            df['Functional'] = df['Functional'].fillna('Typ')
        logger.info("Synchronized structural dependencies (Garages, Basements, MasVnr).")

        # 4. Manual Ordinal Encoding
        for col, mapping in self.ordinal_configs.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)
        logger.info("Executed manual ordinal encoding mappings.")

        # 5. Advanced Feature Engineering (Age Calculation)
        # Handle kondisi jika input dari FastAPI (single input) mengirim 'Age_House' langsung dari form
        if 'Age_House' not in df.columns and 'YrSold' in df.columns and 'YearBuilt' in df.columns:
            df['Age_House'] = df['YrSold'].astype(int) - df['YearBuilt'].astype(int)
            
        if 'Age_Garage' not in df.columns:
            if 'YrSold' in df.columns and 'GarageYrBlt' in df.columns:
                df['Age_Garage'] = df.apply(
                    lambda r: -1 if pd.isna(r['GarageYrBlt']) or int(r['GarageYrBlt']) == -1 
                    else int(r['YrSold']) - int(r['GarageYrBlt']), axis=1
                )
            else:
                df['Age_Garage'] = -1
                
        # Drop YearBuilt & GarageYrBlt karena sudah direpresentasi oleh fitur Age
        df = df.drop(columns=['YearBuilt', 'GarageYrBlt'], errors='ignore')

        if 'YrSold' in df.columns:
            df['YrSold'] = df['YrSold'].astype(str)

        # 6. Sparse Binary Flags
        df['Has_Pool'] = (df['PoolQC'] != 0).astype(int) if 'PoolQC' in df.columns else 0
        df['Has_Misc'] = (df['MiscFeature'] != 'None').astype(int) if 'MiscFeature' in df.columns else 0
        df['Has_Alley'] = (df['Alley'] != 'None').astype(int) if 'Alley' in df.columns else 0
        
        logger.info("Feature engineering and flag generation completed successfully.")
        return df