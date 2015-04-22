__author__ = 'alicebenziger'

import pandas as pd
import numpy as np
import nltk
from collections import Counter
import string
from sklearn.cluster import MiniBatchKMeans

def lat_long_manipulation(train, test):

    esti = 125
    km = MiniBatchKMeans(n_clusters=esti, random_state=1377, init_size=esti*100)
    km.fit(train.ix[:, ['latitude', 'longitude']])
    train['loc_clust_125'] = km.predict(train.ix[:,['latitude','longitude']])
    test['loc_clust_125'] = km.predict(test.ix[:,['latitude', 'longitude']])
    return train, test

if __name__ == '__main__':
    data_bs_train = pd.read_table("yelp_training_set_business.csv", sep=",")
    data_ck_train = pd.read_table("yelp_training_set_checkin.csv", sep=",")
    data_rev_train = pd.read_table("yelp_training_set_review.csv", sep=",")
    data_user_train = pd.read_table("yelp_training_set_user.csv", sep=",")

    data_bs_test = pd.read_table("yelp_test_set_business.csv", sep=",")
    data_ck_test = pd.read_table("yelp_test_set_checkin.csv", sep=",")
    data_rev_test = pd.read_table("yelp_test_set_review.csv", sep=",")
    data_user_test = pd.read_table("yelp_test_set_user.csv", sep=",")

    # adding columns row wise for data_checkin
    data_ck_train["total_checkins"] = data_ck_train.sum(axis=1)
    data_ck_test["total_checkins"] = data_ck_train.sum(axis=1)

    #replacing NA's with zeroes in check -in
    data_ck_train = data_ck_train.fillna(0)
    data_ck_test = data_ck_test.fillna(0)

    # new check in dataframe with just total check-in's
    data_ck_train = data_ck_train[["business_id", "total_checkins"]]
    data_ck_test = data_ck_test[["business_id", "total_checkins"]]


    # freshness of a review
    data_rev_train["freshness"] = pd.to_datetime('2013-01-19')-pd.to_datetime(data_rev_train["date"])
    data_rev_train["freshness"] = data_rev_train["freshness"]/np.timedelta64(1, 'D')

    data_rev_test["freshness"] = pd.to_datetime('2013-03-12')-pd.to_datetime(data_rev_train["date"])
    data_rev_test["freshness"] = data_rev_test["freshness"]/np.timedelta64(1, 'D')


    # features from review text--Text feature extraction takes 6.5 hours
    adj = []
    punct = []
    count_sent = []
    data_rev_train["text_length"] = data_rev_train["text"].str.len()
    data_rev_train = data_rev_train.fillna(0)

    for i in range(data_rev_train.shape[0]):
        #print i
        if data_rev_train["text"][i] == 0:
            adj.append(0)
            punct.append(0)
        else:

            tokens = nltk.tokenize.word_tokenize(data_rev_train["text"][i])
            text = nltk.Text(tokens)
            sentences = nltk.tokenize.sent_tokenize(data_rev_train["text"][i])
            count_sent.append(len(sentences))
            tags = nltk.tag.pos_tag(text)
            text_feat = Counter([k if k not in string.punctuation else "PUNCT" for k in [j for i, j in tags]])
            adj.append(text_feat['JJ'])
            punct.append(text_feat['PUNCT'])
    data_rev_train["count_adj"] = pd.DataFrame(adj)
    data_rev_train["count_punct"] = pd.DataFrame(punct)
    data_rev_train["count_sentences"] = pd.DataFrame(count_sent)

    adj_test = []
    punct_test = []
    count_sent_test = []
    data_rev_test["text_length"] = data_rev_test["text"].str.len()
    data_rev_test = data_rev_test.fillna(0)

    for i in range(data_rev_test.shape[0]):
        if data_rev_test["text"][i] == 0:
            adj_test.append(0)
            punct_test.append(0)
        else:
            tokens = nltk.tokenize.word_tokenize(data_rev_test["text"][i])
            text = nltk.Text(tokens)
            sentences = nltk.tokenize.sent_tokenize(data_rev_train["text"][i])
            count_sent_test.append(len(sentences))
            tags = nltk.tag.pos_tag(text)
            text_feat = Counter([k if k not in string.punctuation else "PUNCT" for k in [j for i, j in tags]])
            adj.append(text_feat['JJ'])
            punct.append(text_feat['PUNCT'])

    data_rev_test["count_adj"] = pd.DataFrame(adj)
    data_rev_test["count_punct"] = pd.DataFrame(punct)
    data_rev_test["count_sentences"] = pd.DataFrame(count_sent_test)
    data_rev_train.to_csv("yelp_rev_train_feat.csv", index=False)
    data_rev_test.to_csv("yelp_rev_tst_feat.csv", index=False)

####---- Text feature creation ends -----##

    #Extracting zip code from addresses
    ZIP = []
    for i in range(data_bs_train.shape[0]):
        last_five_chars = data_bs_train["full_address"][i][-5:]
        if last_five_chars.isdigit():
            ZIP.append(last_five_chars)
        else:
            ZIP.append("12345")
    data_bs_train["zip_code"] = pd.DataFrame(ZIP)

    ZIP_test = []
    for i in range(data_bs_test.shape[0]):
        last_five_chars = data_bs_test["full_address"][i][-5:]
        if last_five_chars.isdigit():
            ZIP_test.append(last_five_chars)
        else:
            ZIP_test.append("12345")

    data_bs_test["zip_code"] = pd.DataFrame(ZIP_test)

    #including longitude and latitude clusters
    data_bs_train, data_bs_test = lat_long_manipulation(data_bs_train, data_bs_test)

    # filling NA in categories with other
    all_categories = data_bs_train['categories']
    all_categories[pd.isnull(all_categories)] = 'Other'
    data_bs_train['categories'] = all_categories

    all_categories = data_bs_test['categories']
    all_categories[pd.isnull(all_categories)] = 'Other'
    data_bs_test['categories'] = all_categories

    data_rev_train_1 = pd.read_csv("yelp_rev_train_feat.csv",sep=',')
    data_rev_test_1 = pd.read_csv("yelp_rev_tst_feat.csv", sep =',')

    #getting training data
    data_merge1 = pd.merge(data_rev_train_1, data_bs_train, on ='business_id',how='left',suffixes=('_rev','_bs'))
    data_merge2 = pd.merge(data_merge1, data_ck_train,on='business_id',how='left')
    data_merge2['total_checkins'] = data_merge2['total_checkins'].fillna(0)
    data_train = pd.merge(data_merge2,data_user_train,on = 'user_id',how='left',suffixes=('_rev','_user'))
    data_train['average_stars'] = data_train['average_stars'].fillna(4)
    data_train['review_count_user'] = data_train['review_count_user'].fillna(6)
    data_train['votes_cool_user'] = data_train['votes_cool_user'].fillna(0)
    data_train['votes_funny_user'] = data_train['votes_funny_user'].fillna(0)
    # data_train['votes_useful_user'] = data_train['votes_useful_user'].fillna(0)

    data_train = data_train
    data_train_copy = data_train.copy(deep=True)

    del data_train['business_id'],data_train['date'],data_train['text'],data_train['type_rev'],\
        data_train['user_id'],data_train['votes_cool_rev'],data_train['votes_funny_rev'], \
        data_train['city'],data_train['full_address'],data_train['name_rev'],data_train['neighborhoods'],\
        data_train['state'],data_train['type_bs'],\
        data_train['name_user'],data_train['type'],data_train['latitude'], data_train['longitude'],data_train['votes_useful_user']
        # data_train['votes_cool_user'], data_train['votes_funny_user']#,

    data_train.to_csv('train_new.csv',index=False)

    #getting test data
    del data_bs_test["neighborhoods"], data_bs_train["neighborhoods"]
    test_train_bs = data_bs_test.copy(deep=True)
    train_test_diff_bs = data_bs_train[~data_bs_train["business_id"].isin(data_bs_test["business_id"])].dropna()
    test_train_bs= test_train_bs.append(train_test_diff_bs)

    test_train_user = data_user_test.copy(deep=True)
    train_test_diff_usr = data_user_train[~data_user_train["user_id"].isin(data_user_test["user_id"])].dropna()
    test_train_user= test_train_user.append(train_test_diff_usr)

    del test_train_user["votes_useful"]


    ##------merging test-----------##
    data_mergea =  pd.merge(data_rev_test_1, test_train_bs, on='business_id', how='left',suffixes=('_rev','_bs'))
    data_mergeb = pd.merge(data_mergea, data_ck_train,on='business_id',how='left')
    data_mergeb['total_checkins'] = data_mergeb['total_checkins'].fillna(0)
    data_test = pd.merge(data_mergeb,test_train_user,on='user_id',how='left',suffixes=('_rev','_user'))
    data_test['average_stars'] = data_test['average_stars'].fillna(4)
    data_test['review_count_user'] = data_test['review_count_user'].fillna(10)
    data_test['votes_cool'] = data_test['votes_cool'].fillna(0)
    data_test['votes_funny'] = data_test['votes_funny'].fillna(0)

    data_test = data_test
    data_test_copy = data_test.copy(deep=True)

    del data_test['business_id'],data_test['date'],data_test['text'],data_test['type_rev'],\
        data_test['user_id'], \
        data_test['city'],data_test['full_address'],data_test['name_rev'],\
        data_test['state'],data_test['type_bs'],\
        data_test['name_user'],data_test['type'],data_test['latitude'], data_test['longitude']

    data_test.to_csv('test_new.csv', index=False)


