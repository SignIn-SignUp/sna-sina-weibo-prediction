
import time
import pandas as pd
from sklearn import model_selection

import jieba
import jieba.analyse
import math
import numpy as np
import time

# from ItemCF import ItemCF
class ItemCF:
    def __init__(self, data):
        self.data = data

    def user_item(self):  # Training set converted to dict
        data = self.data
        self.traindata = {}
        for user, item, time in np.array(data).tolist():
            self.traindata.setdefault(user, {})
            self.traindata[user][item] = 1  # time

    def ItemSimilarity(self, alpha=1):
        train = self.traindata
        # calculate co-rated users between items
        C = dict()
        N = dict()
        for u, items in train.items():
            for i, tui in items.items():
                N.setdefault(i, 0)
                N[i] += 1
                for j, tuj in items.items():
                    C.setdefault(i, {})
                    if i == j:
                        continue
                    C[i].setdefault(j, 0)
                    C[i][j] += 1 / (1 + alpha * abs(tui - tuj))
        # calculate finial similarity matrix W
        W = dict()
        for i, related_items in C.items():
            W.setdefault(i, {})
            for j, cij in related_items.items():
                W[i].setdefault(j, 0)
                W[i][j] = cij / math.sqrt(N[i] * N[j] * 1.0)
        self.itemSim = W

    def recommend(self, user, tid=1, k=1, beta=1):
        W = self.itemSim
        train = self.traindata
        rank = dict()
        ru = train.get(user, {})
        for i, tui in ru.items():
            for j, wj in W[i].items():
                if j in ru.keys():
                    continue
                rank.setdefault(j, 0)
                rank[j] += wj / (1 + beta * abs(tid - tui))
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:k])



class cWeibo:

    def __init__(self, path):
        self.path = path

    def importData(self):
        path = self.path
        # import sample set
        data = pd.read_csv(path,
                           encoding='utf8', sep='\t',
                           names=['luid', 'mid', 'time', 'fcs', 'ccs', 'lcs', 'cont'])  # nrows=1000
        data['fcs'] = data['fcs'].astype('int')  # one week after repost，weight 0.5
        data['ccs'] = data['ccs'].astype('int')  # one week after comments,  0.25
        data['lcs'] = data['lcs'].astype('int')  # one week after likes 0.25
        train, test = model_selection.train_test_split(data, test_size=0.2)
        self.traindata = pd.DataFrame(data)  # train set
        self.testdata = pd.DataFrame(test)  # test set

        data = pd.read_csv(
            path,
            encoding='utf8', sep='\t', names=['luid', 'mid', 'time', 'cont'])  # nrows=100
        self.predata = data  # predict set

    def ETL(self):

        # Convert time to 0-23 numbers, taking into account special holidays if necessary.
        self.traindata['time'] = self.traindata.apply(lambda x: (time.strptime(x['time'], "%Y-%m-%d %H:%M:%S")).tm_hour,
                                                      axis=1)
        self.traindata.rename(columns=lambda x: x.replace('time', 'tid'), inplace=True)  # rename tid
        self.testdata['time'] = self.testdata.apply(lambda x: (time.strptime(x['time'], "%Y-%m-%d %H:%M:%S")).tm_hour,
                                                    axis=1)
        self.testdata.rename(columns=lambda x: x.replace('time', 'tid'), inplace=True)
        self.predata['time'] = self.predata.apply(lambda x: (time.strptime(x['time'], "%Y-%m-%d %H:%M:%S")).tm_hour,
                                                  axis=1)
        self.predata.rename(columns=lambda x: x.replace('time', 'tid'), inplace=True)
        # content analysis , consider special meaning words likes red poket
        # jieba.suggest_freq('@', True)
        self.traindata['cont'] = self.traindata['cont'].astype(str).fillna('')
        self.traindata['cont'] = self.traindata.apply(lambda x: ",".join(jieba.analyse.extract_tags(x['cont'], topK=50, \
                                                                                                    allowPOS=(
                                                                                                    'n', 'nr', 'ns',
                                                                                                    'nt', 'nz', 'a',
                                                                                                    'ad', 'an', 'f',
                                                                                                    's', 'i', 't', 'v',
                                                                                                    'vd', 'vn'))),
                                                      axis=1)
        self.traindata = self.traindata.drop('cont', axis=1).join(
            self.traindata['cont'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))

        self.testdata['cont'] = self.testdata['cont'].astype(str).fillna('')
        self.testdata['cont'] = self.testdata.apply(lambda x: ",".join(jieba.analyse.extract_tags(x['cont'], topK=50, \
                                                                                                  allowPOS=(
                                                                                                  'n', 'nr', 'ns', 'nt',
                                                                                                  'nz', 'a', 'ad', 'an',
                                                                                                  'f', 's', 'i', 't',
                                                                                                  'v', 'vd', 'vn'))),
                                                    axis=1)
        self.testdata = self.testdata.drop('cont', axis=1).join(
            self.testdata['cont'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))

        self.predata['cont'] = self.predata['cont'].astype(str).fillna('')
        self.predata['cont'] = self.predata.apply(lambda x: ",".join(jieba.analyse.extract_tags(x['cont'], topK=50, \
                                                                                                allowPOS=(
                                                                                                'n', 'nr', 'ns', 'nt',
                                                                                                'nz', 'a', 'ad', 'an',
                                                                                                'f', 's', 'i', 't', 'v',
                                                                                                'vd', 'vn'))), axis=1)
        self.predata = self.predata.drop('cont', axis=1).join(
            self.predata['cont'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))
        # Create indexing
        ft_train = set(self.traindata.iloc[:, 6])
        ft_pred = set(self.predata.iloc[:, 3])
        ft = list(ft_train.symmetric_difference(ft_pred))
        df_ft = pd.DataFrame(ft, columns=['tag'])
        df_ft['fid'] = df_ft.index
        self.traindata = pd.merge(self.traindata, df_ft, on=['tag'], how='left')
        self.traindata = self.traindata[['luid', 'mid', 'tid', 'fid', 'fcs', 'ccs', 'lcs']]
        self.traindata = self.traindata.dropna(axis=0, how='any')
        self.traindata['fid'] = self.traindata['fid'].astype('int')
        print(self.traindata.shape)
        self.testdata = pd.merge(self.testdata, df_ft, on=['tag'], how='left')
        self.testdata = self.testdata[['luid', 'mid', 'tid', 'fid', 'fcs', 'ccs', 'lcs']]
        self.testdata = self.testdata.dropna(axis=0, how='any')
        self.testdata['fid'] = self.testdata['fid'].astype('int')
        print(self.testdata.shape)
        self.predata = pd.merge(self.predata, df_ft, on=['tag'], how='left')
        self.predata = self.predata[['luid', 'mid', 'tid', 'fid']]
        self.predata = self.predata.dropna(axis=0, how='any')
        self.predata['fid'] = self.predata['fid'].astype('int')
        print(self.predata.shape)

    def callItemCF(self):
        data = self.traindata

        data_f = data[['fid', 'fcs', 'tid']]
        data_c = data[['fid', 'ccs', 'tid']]
        data_l = data[['fid', 'lcs', 'tid']]
        # Training the Retweet Count Recommendation Model
        ic_f = ItemCF(data_f)
        ic_f.user_item()  # Transform into dict and generate training and test sets
        ic_f.ItemSimilarity()  # Generate item similarity matrix
        self.ic_f = ic_f

        ic_c = ItemCF(data_c)
        ic_c.user_item()
        ic_c.ItemSimilarity()
        self.ic_c = ic_c

        ic_l = ItemCF(data_l)
        ic_l.user_item()
        ic_l.ItemSimilarity()
        self.ic_l = ic_l

        test = self.testdata
        test_f = test[['mid', 'fid', 'tid', 'fcs']]
        test_c = test[['mid', 'fid', 'tid', 'ccs']]
        test_l = test[['mid', 'fid', 'tid', 'lcs']]
        test_f['pfcs'] = test_f.apply(
            lambda x: list(ic_f.recommend(x['fid']).keys())[0] if ic_f.recommend(x['fid']) else 0, axis=1)

        test_c['pccs'] = test_c.apply(
            lambda x: list(ic_c.recommend(x['fid']).keys())[0] if ic_c.recommend(x['fid']) else 0, axis=1)

        test_l['plcs'] = test_l.apply(
            lambda x: list(ic_l.recommend(x['fid']).keys())[0] if ic_l.recommend(x['fid']) else 0, axis=1)

        # 计算准确率
        precision = self.precision(test_f[['mid', 'fcs', 'pfcs']], test_c[['mid', 'ccs', 'pccs']],
                                   test_l[['mid', 'lcs', 'plcs']])
        print(precision)

    def precision(self, test_f, test_c, test_l):  # calculate precision
        test_f = test_f.groupby('mid').mean()
        test_c = test_c.groupby('mid').mean()
        test_l = test_l.groupby('mid').mean()

        # print(test_f.columns.tolist)
        test_f['dev_f'] = abs(test_f['fcs'] - test_f['pfcs']) / np.clip(test_f['fcs'] + 5, 1, None)
        test_c['dev_c'] = abs(test_c['ccs'] - test_c['pccs']) / np.clip(test_c['ccs'] + 3, 1, None)
        test_l['dev_l'] = abs(test_l['lcs'] - test_l['plcs']) / np.clip(test_l['lcs'] + 3, 1, None)

        test = test_f
        test = pd.merge(test, test_c, left_index=True, right_index=True)  # 以索引连接
        test = pd.merge(test, test_l, left_index=True, right_index=True)

        test['prec'] = 1 - 0.5 * test['dev_f'] - 0.25 * test['dev_c'] - 0.25 * test['dev_l']

        test['count'] = np.where(test['pfcs'] + test['pccs'] + test['plcs'] > 100, 100,
                                 test['pfcs'] + test['pccs'] + test['plcs'])


        test['sgn'] = np.where(test['prec'] > 0.8, 1, 0)
        test['on'] = (test['count'] + 1) * test['sgn']
        test['down'] = test['count'] + 1

        prec_df = test[['on', 'down']]

        on = prec_df['on'].sum()
        down = prec_df['down'].sum()
        return (on / down)

    def predict(self):
        '''
        Name of document：weibo_result_data.txt
        '''


        predata = self.predata[['luid', 'mid', 'tid', 'fid']]

        predata['recommend_f'] = predata['fid'].apply(lambda x: self.ic_f.recommend(x))
        predata['recommend_c'] = predata['fid'].apply(lambda x: self.ic_c.recommend(x))
        predata['recommend_l'] = predata['fid'].apply(lambda x: self.ic_l.recommend(x))

        predata['pfcs'] = predata['recommend_f'].apply(lambda x: list(x.keys())[0] if x else 0)
        predata['pccs'] = predata['recommend_c'].apply(lambda x: list(x.keys())[0] if x else 0)
        predata['plcs'] = predata['recommend_l'].apply(lambda x: list(x.keys())[0] if x else 0)

        predata['results'] = predata.apply(
            lambda x: f"{int(x['pfcs'])},{int(x['pccs'])},{int(x['plcs'])}", axis=1)

        result_data = predata[['luid', 'mid', 'results']]
        result_data = result_data.drop_duplicates(subset=['luid', 'mid'])

        result_data.to_csv('weibo_result_data.txt', sep='\t', index=False, header=False)

        print("File has been saved as 'weibo_result_data.txt'.")
        print(result_data.shape)


if __name__ == "__main__":
    start = time.time()

    wb = cWeibo('D:\\xxx\\weibo')
    wb.importData()  # import data
    wb.ETL()  # feature extracting
    wb.callItemCF()  # recommendation algorithm
    wb.predict()

    end = time.time()
    print('finish all in %s' % str(end - start))
