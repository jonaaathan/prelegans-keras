import numpy as np
import pandas as pd
##########################
# todo: normalize each columns (by .describe())
#       make a hook back to original images
#       or make a function to query image file to get it back
#
##########################
def csv_parser(filename, codes=False, code_key=None, columns=False, columns_name=None, sep='\t'):
    '''
    try:
        raw_data = pd.read_hdf(filename.split('.')[0] + '.h5', 'raw_data')
    except FileNotFoundError:
        raw_data = pd.read_csv(filename, sep=sep, header='infer')
        raw_data.columns = raw_data.columns.str.strip().str.lower().str.replace(' ', '_')
        try:
            raw_data['fam'] = raw_data['protein_families'].apply(fam)
            raw_data['sup'] = raw_data['protein_families'].apply(supfam)
            raw_data['sub'] = raw_data['protein_families'].apply(subfam)
        except KeyError:
            pass

        raw_data.to_hdf(filename.split('.')[0] + '.h5', 'raw_data')
    '''
    try:
        raw_data = pd.read_csv(filename, sep=sep, header='infer')
        raw_data.columns = raw_data.columns.str.strip().str.lower().str.replace(' ', '_')
    except FileNotFoundError:
        raise IOError('File not found in path')
    # need to refractor for columns, codes and None
    if columns:
        if type(columns_name) == list:
            # check input columns_name are in the df columns
            print (raw_data.columns.values.tolist())
            columns_name = [col.strip().lower().replace(' ', '_') for col in columns_name]
            assert set(columns_name) < set(raw_data.columns.values.tolist())
            # print ('length', len(set(columns_name)), len(set(raw_data.columns.values.tolist())))
            # raise IndexError('No columns %s found in datafile %s' % (columns_name, filename))

            x_data = raw_data.loc[:, columns_name]
            # print('x_data shape', x_data.shape)
        else:
            raise TypeError('columns_name must be wrapped with a list')
    else:
        x_data = raw_data

    if codes:
        if type(code_key) == str:
            try:
                print(code_key.strip().lower().replace(' ', '_'))
                code_key = code_key.strip().lower().replace(' ', '_')
                code_vc = raw_data.loc[:, code_key].value_counts()
            except KeyError:
                # search for nearest keys
                raise KeyError('err')
            # print('code_vc', code_vc)
            y_data = raw_data.loc[:, code_key]
            # select pos data and neg data
            '''
            pos_data = raw_data[raw_data[code_key] != 'Unassigned'][
                ~raw_data[code_key].isin(code_vc[code_vc == 1].index.tolist())]
            code_cats = pos_data[code_key].astype('category').cat.codes
            y_data = code_cats.tolist()
            y_data = [y + 1 for y in y_data]
            '''
        else:
            pass
            '''
            pf = raw_data[code_key]
            pos_data = raw_data[pf[code_key[0]].notnull() & pf[code_key[1]].notnull()]
            y_data = [pos_data[k].tolist() for k in code_key]
            x_data = pos_data.sequence
            '''
    else:
        y_data = None

    return x_data, y_data
