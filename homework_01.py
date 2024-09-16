import pandas as pd
import numpy as np

def get_pandas_v():
    print(f"I'm using Pandas == {pd.__version__}")
          
def load_data(file_path, sep):
    return pd.read_csv(file_path, encoding='utf-8', sep=sep)

def mode_imputation(df, column):
    value = df[column].mode()
    df.fillna({column: value}, inplace=True)
    return df

def lin_reg(df, columns, y):
    X = df[columns].to_numpy()
    X_T = X.T
    XTX = X_T @ X
    XTX_inv = np.linalg.inv(XTX)
    weights = XTX_inv @ X_T @ y
    return weights

if __name__ == "__main__":
    # print current pandas version
    print("Q1")
    get_pandas_v()

    # load csv into pandas df
    file_path = './datasets/laptops.csv'
    df = load_data(file_path, ',')

    # questions from Q2 to Q5
    print(f"\nQ2\nOur dataset has {df.shape[0]} rows")
    
    print(f"\nQ3\nThere are {df['Brand'].nunique()} laptop brands presented in the dataset")

    print(f"\nQ4\nOur dataset has {df.isna().any().sum()} columns with missing values")

    print(f"\nQ5\nThe maximum final price of Dell notebooks in the dataset is ${df.loc[df['Brand'] == 'Dell', 'Final Price'].max()}")

    # Q6
    pre_median_screen = df['Screen'].median()
    
    # use mode imputation and calculate median once again
    new_df = mode_imputation(df, 'Screen')
    post_median_screen = new_df['Screen'].median()

    if pre_median_screen == post_median_screen:
        changed = "\nNo, it didn't change.\n"
    else:
        changed = "\nYes, it changed.\n"

    print(f"\nQ6{changed}The median screen value was {pre_median_screen} and after imputation it's {post_median_screen}")

    # Q7
    y = [1100, 1300, 800, 900, 1000, 1100]
    reg_df = df.query('`Brand` == "Innjoo"')

    weights = lin_reg(reg_df, ['RAM', 'Storage', 'Screen'], y)

    print(f"\nQ7\nThe sum of weights is {np.sum(weights):.2f}")
