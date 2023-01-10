# Contains functions for scaling copied from analyze_data_stability.py
# but with inverse and saving of scaler
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd
import numpy as np


def mm_scale(train_scaled, test_scaled, col, thresh=3, border=2, return_mm=False):
    # project outliers to [-thresh-border, -thresh] and [thresh, thresh+border] respectively
    if test_scaled is not None:
        test_not_na = test_scaled[col].notna()
    train_not_na = train_scaled[col].notna()
    ix_train_p = train_scaled.index[(train_scaled[col] > thresh) & (train_not_na)]
    mm_pos = None
    if len(ix_train_p) > 0:
        mm_pos = MinMaxScaler(feature_range=(thresh, thresh + border))
        outliers_values = train_scaled.loc[ix_train_p, col].values
        mm_pos.fit(np.r_[[thresh], outliers_values].reshape(-1,
                                                            1))  # we need this to fix left border for scaler (min value = 3 sigma)
        train_scaled.loc[ix_train_p, col] = mm_pos.transform(outliers_values.reshape(-1, 1)).flatten()

        # scale reference
        if test_scaled is not None:
            ix_test_p = test_scaled.index[(test_scaled[col] > thresh) & (test_not_na)]
            if len(ix_test_p) > 0:
                test_scaled.loc[ix_test_p, col] = mm_pos.transform(
                    test_scaled.loc[ix_test_p, col].values.reshape(-1, 1)).flatten()

    ix_train_n = train_scaled.index[(train_scaled[col] < -thresh) & (train_not_na)]
    mm_neg = None
    if len(ix_train_n) > 0:
        mm_neg = MinMaxScaler(feature_range=(-thresh - border, -thresh))
        outliers_values = train_scaled.loc[ix_train_n, col].values
        mm_neg.fit(np.r_[[-thresh], outliers_values].reshape(-1, 1))
        train_scaled.loc[ix_train_n, col] = mm_neg.transform(outliers_values.reshape(-1, 1)).flatten()

        if test_scaled is not None:
            # scale reference
            ix_test_n = test_scaled.index[(test_scaled[col] < -thresh) & (test_not_na)]
            if len(ix_test_n) > 0:
                test_scaled.loc[ix_test_n, col] = mm_neg.transform(
                    test_scaled.loc[ix_test_n, col].values.reshape(-1, 1)).flatten()
    if return_mm:
        return mm_pos, mm_neg


def scale_minmax(test_data_scaled, col, thresh, border):
    not_na_test = test_data_scaled[col].notna()
    ix_test_notna = test_data_scaled.index[not_na_test]
    not_na_test_col = test_data_scaled.loc[ix_test_notna, col]

    outlier_test_col = not_na_test_col.loc[not_na_test_col.abs() > thresh + border]

    if outlier_test_col.shape[0] > 0:
        mm_scalers_test = mm_scale(test_data_scaled, None, col, thresh + border, thresh + border, return_mm=True)
        return mm_scalers_test
    return None


# todo: split into scale train and scale test
def scale_data(train_df, test_df=None, thresh=3, border=2, quantile_range=(10,90), scale_test=False):
    train_data_scaled = train_df.copy()
    test_data_scaled = None
    if test_df is not None:
        test_data_scaled = test_df.copy()
    scalers = {}
    for col in train_data_scaled.columns:
        rs, ix_train = rs_scale(train_data_scaled, col, return_rs=True, qr=quantile_range)
        if test_df is not None:
            rs_scale(test_data_scaled, col, rs)

        not_na_train_col = train_data_scaled.loc[ix_train, col]
        #
        outlier_train_col = not_na_train_col.loc[not_na_train_col.abs() > thresh + border]
        mm_scalers = None

        if outlier_train_col.shape[0] > 0:
            mm_scalers = mm_scale(train_data_scaled, test_data_scaled, col, thresh, border, return_mm=True)

        if test_df is not None and scale_test:
            mm_scalers_test = scale_minmax(test_data_scaled, col, thresh, border)

            scalers[col] = {'rs': rs, 'mm': mm_scalers, 'mm_test': mm_scalers_test}
        else:
            scalers[col] = {'rs': rs, 'mm': mm_scalers}

    train_data_scaled.fillna(0, inplace=True)
    test_data_scaled = test_data_scaled.fillna(0) if (test_df is not None) else None
    scalers['_extra'] = {'thresh': thresh, 'border': border}

    return train_data_scaled, test_data_scaled, scalers


def rs_scale(data, feature_name, rs: RobustScaler = None, qr=(10, 90), return_rs=False):
    not_na = data[feature_name].notna()
    ix = data.index[not_na]
    if rs is None:
        rs = RobustScaler(quantile_range=qr)
        rs.fit(data.loc[ix, feature_name].values.reshape(-1, 1))
        return_rs = True
    data.loc[ix, feature_name] = rs.transform(data.loc[ix, feature_name].values.reshape(-1, 1)).flatten()
    if return_rs:
        return rs, ix
    else:
        return ix


def scale_test_data(test_df, scalers, scale_test_minmax=True):
    test_data_scaled = test_df.copy()
    if '_extra' in scalers and 'thresh' in scalers['_extra']:
        thresh = scalers['_extra']['thresh']
    else:
        thresh = 3
    if '_extra' in scalers and 'border' in scalers['_extra']:
        border = scalers['_extra']['border']
    else:
        border = 2

    for col in test_data_scaled.columns:
        if not col in scalers:
            raise Exception(f"Column {col} will not be scaled, scaler not found")

        scalers_col = scalers[col]
        if 'rs' not in scalers_col:
            raise Exception(f"Column {col} will not be scaled, robust scaler not found")

        scalers_rs = scalers_col['rs']
        rs_scale(test_data_scaled, col, scalers_rs)

        if 'mm' not in scalers_col or scalers_col['mm'] is None:
            if scale_test_minmax:
                scale_minmax(test_data_scaled, col, thresh, border)
            continue

        scalers_mm = scalers_col['mm']
        scalers_mm_pos, scalers_mm_neg = scalers_mm

        mm_scale_test(test_data_scaled, col, mm_pos=scalers_mm_pos, mm_neg=scalers_mm_neg, thresh=thresh)
        if scale_test_minmax:
            scale_minmax(test_data_scaled, col, thresh, border)

    return test_data_scaled


def inverse_scale_data(df, scalers):
    df = df.copy()
    thresh = scalers['_extra']['thresh']
    for col in df.columns.values:
        if not col in scalers:
            continue
        rs = scalers[col]['rs']
        mm_pos, mm_neg = scalers[col]['mm'] if scalers[col]['mm'] else (None, None)

        not_na = df[col].notna()
        ix = df.index[not_na]

        outliers_pos_idx = df.loc[df.loc[ix, col] > thresh, col].index
        outliers_neg_idx = df.loc[df.loc[ix, col] < -thresh, col].index

        if mm_pos is not None and len(outliers_pos_idx) > 0:
            df.loc[outliers_pos_idx, col] = mm_pos.inverse_transform(
                df.loc[outliers_pos_idx, col].values.reshape(-1, 1)).flatten()

        if mm_neg is not None and len(outliers_neg_idx) > 0:
            df.loc[outliers_neg_idx, col] = mm_neg.inverse_transform(
                df.loc[outliers_neg_idx, col].values.reshape(-1, 1)).flatten()

        df.loc[ix, col] = rs.inverse_transform(df.loc[ix, col].values.reshape(-1, 1)).flatten()

    return df


def mm_scale_test(test_df, col, mm_pos=None, mm_neg=None, thresh=3):
    test_not_na = test_df[col].notna()
    ix_test_p = test_df.index[(test_df[col] > thresh) & (test_not_na)]
    if len(ix_test_p) > 0 and mm_pos is not None:
        test_df.loc[ix_test_p, col] = mm_pos.transform(
            test_df.loc[ix_test_p, col].values.reshape(-1, 1)).flatten()

    ix_test_n = test_df.index[(test_df[col] < -thresh) & (test_not_na)]
    if len(ix_test_n) > 0 and mm_neg is not None:
        test_df.loc[ix_test_n, col] = mm_neg.transform(
            test_df.loc[ix_test_n, col].values.reshape(-1, 1)).flatten()
    return test_df


def test_scale_inverse():
    df_test = pd.DataFrame([[1, 1, 1, 1, 1, 1.2], [0, 10, 500, 600, -100, -200]]).T

    df_train = pd.DataFrame(
        [[1, 1, 1., 1, 1, 1, 1, 1, 1, 1, 1, 1.000001, -1], [1, 1, 2., 3., 4., 5, 1, 1, 1, 1, 1, 1, 1]]).T

    df_train_scaled, df_test_scaled, scalers = scale_data(df_train, df_test, scale_test=True)
    df_inverse = inverse_scale_data(df_test_scaled, scalers)

    print("df_train")
    print(df_train)
    print("\ndf_test")
    print(df_test)
    print("\ndf_train_scaled")
    print(df_train_scaled)
    print("\ndf_test_scaled")
    print(df_test_scaled)
    print("\nscalers")
    print(scalers)
    print("df_inverse")
    print(df_inverse)


def test_scale():
    df_test = pd.DataFrame([[1, 1, 1, 1, 1, 1.2], [0, 10, 500, 600, -100, -200]]).T

    df_train = pd.DataFrame(
        [[1, 1, 1., 1, 1, 1, 1, 1, 1, 1, 1, 1.000001, 100], [1, 1, 2., 3., 4., 5, 1, 1, 1, 1, 1, 1, 1]]).T

    df_train_scaled, df_test_scaled, scalers = scale_data(df_train, df_test)
    df_test_scaled2 = scale_test_data(df_test, scalers)

    print("df_train")
    print(df_train)
    print("\ndf_test")
    print(df_test)
    print("\ndf_train_scaled")
    print(df_train_scaled)
    print("\ndf_test_scaled")
    print(df_test_scaled)
    print("\nscalers")
    print(scalers)
    print("df_test_scaled2")
    print(df_test_scaled2)
