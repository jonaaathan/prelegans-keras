import numpy as np
import pandas as pd
import sys
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler

import io_tools as io

# refractors to callable modules [ PCA, padding ...]
# make sample more positive dataset too.
def scaling(ndarray, method='minmax'):
    scaler = MinMaxScaler()
    # return scaler(quantile_range=(25, 75)).fit_transform(ndarray)
    scaled = scaler.fit_transform(ndarray)
    print ('scaling', scaled)
    # assert all(np.amin(scaled)) >= 0, 'scaled data is smaller than 0'
    # assert all(np.amax(scaled)) <= 1, 'scaled data is larger than 1'
    return scaled

def retrieve_past(x_data, y_data, time_forward, sample=False):

    start_time = time.time()
    print ('retrieve past data')
    use_index = []
    # full = np.stack([x_data[:time_forward], x_data[1:time_forward+1]])
    all_index = x_data.index
    index_set = all_index
    all_things = [] # for keeping all the frames to do numpy stack
    retained_row = [] # saving retained_row for retrieving y

    if sample:
        pos = y_data[y_data == 1]
        neg = y_data[y_data == 0]
        # print ('type index', pos.index.values.shape, np.random.choice(neg.index.values, len(pos.index) * 4, replace=False).shape)
        neg_sampled = np.random.choice(neg.index.values, len(pos.index) * 4, replace=False)
        index_set = np.concatenate([pos.index.values, neg_sampled])
        print ('\n', pos.value_counts(), pos.index[:20], '\n', neg_sampled[:20], 'index length', index_set.shape)
        index_set = np.sort(index_set)

    N = 0
    for index in index_set:
        past_index = [index - i for i in range(time_forward)]
        if all(i in all_index for i in past_index):
            current_frame = np.asarray(x_data.loc[past_index])
            all_things.append(current_frame)
            retained_row.append(index)
            # print (N, index)
            N += 1

            # full = np.concatenate([full, np.expand_dims(current_frame, axis=0)], axis=0)

    full_x = np.stack(all_things)
    full_y = y_data.loc[retained_row]
    assert len(full_x.shape) == 3, 'data should have 3 dimensions, (sample, time, features)'
    # assert type(full_x) == ndarray, 'data should be in ndarray format'
    print ('final x shape', full_x.shape)
    # print ('bincount 0, 1 ', full_y.value_counts())# np.bincount(fully))
    print('data read time for {0} rows. Total time: {1:.3f}s'.format( full_x.shape[0],
                                                                      time.time() - start_time))
    return full_x, full_y

def preprocess_dataset(x_data, y_data=None, padded=True, pad_y_data=False):
    normalized_curve = x_data.iloc[:, 4:]
    # print ('nan', normalized_curve[normalized_curve.isnull().any(axis=1)])
    normalized_curve.dropna(inplace=True)
    keepIndex = normalized_curve.index.tolist()
    # align
    mean_vec = normalized_curve.sum(axis=1)
    num_col = normalized_curve.shape[1]
    norm_vec = mean_vec.div(num_col)
    normalized_curve = normalized_curve.sub(norm_vec, axis=0)
    assert normalized_curve.shape[1] ==  num_col
    assert normalized_curve.mean(axis=1).all() == 0
    # print ('norm_mean_row',  normalized_curve.iloc[1:10,:].sum())

    # normalized for pca
    normalized_curve = normalized_curve.sub(normalized_curve.mean(axis=1), axis=0)
    df_norm = normalized_curve.div(normalized_curve.std(axis=1), axis=0)
    # pca
    pca = PCA(n_components=3)
    pca_curves = pca.fit_transform(df_norm.values)
    projected_curves = pd.DataFrame(data=pca_curves)
    projected_curves.index = keepIndex

    # replace NaN

    # normalize non-curve
    x_dropna_non_curve = x_data.loc[keepIndex].iloc[:, :4]
    print ('x dropna index', x_dropna_non_curve[['length', 'comspeed1']].isnull())
    scaled = scaling(x_dropna_non_curve[['length']]) # pass array of 1 label to get the 2-d ndarray requirement
    print ('scaled length', scaled)

    # combine x
    assert x_dropna_non_curve.shape[0] == projected_curves.shape[0]
    x = pd.concat([x_dropna_non_curve, projected_curves], axis=1)
    y = None

    if y_data is not None:
        y = y_data.loc[keepIndex]
        assert x.shape[0] == y.shape[0]
    return x, y

def load_dataset(infile, codes=None, code_key=None, columns=None, columns_name=None):
    filename = infile
    filetype = filename.split('.')[-1]

    if filetype == 'csv':
        x_data, y_data = io.csv_parser(filename, codes, code_key, columns, columns_name, sep=', ')
    elif filetype == 'tsv':
        x_data, y_data = io.csv_parser(filename, codes, code_key, columns, columns_name, sep='\t')
    elif filetype == 'img':
        raise NotImplementedError('Image parser is not supported yet.')
    else:
        raise NotImplementedError('No parser for current file type.')
    return x_data, y_data

if __name__ == '__main__':
    col_names = ['length', 'comSpeed1', 'bodyAxisSpeed1', 'pumpingRate', 'unnamed:_2', 'unnamed:_3', 'unnamed:_4', 'unnamed:_5', 'unnamed:_6', 'unnamed:_7', 'unnamed:_8', 'unnamed:_9', 'unnamed:_10', 'unnamed:_11', 'unnamed:_12', 'unnamed:_13', 'unnamed:_14', 'unnamed:_15']
    # 'DMP_event'
    filepath = '20170818_AM_N2_comp1_fullDataTable.tsv'
    # 431747
    x, y = load_dataset(filepath, columns=True, columns_name=col_names, codes=True, code_key='DMPevent')
    x, y = preprocess_dataset(x, y)
    # retrieve_past(x, y, 10, sample=True)
    # selected = x.loc[1000:1010,:]
    # a = map(lambda x: x, normalized)
    # print(map(lambda x:x.mean(), selected))
