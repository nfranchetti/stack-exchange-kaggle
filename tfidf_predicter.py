import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

def create_clean_columns(infile):
    df = pd.read_csv('datasets/%s' % infile)
    
    df['clean content'] = df['content'].apply(cleanhtml)
    df['clean title'] = df['title'].apply(cleanhtml)
    return df

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = cleantext.replace('\n', '')
    cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
    cleantext = cleantext.lower()
    return cleantext

def do_tfidf(df):
    tvec = TfidfVectorizer(stop_words='english')
    tvec.fit(df['clean content'])

    words_df = pd.DataFrame(tvec.transform(df['clean content']).todense(),
                   columns=tvec.get_feature_names())
    return words_df

def get_best_word(df, index):
    row = df.iloc[index, :]
    row_tuple = zip(row.index, row.values)
    return max(row_tuple, key=lambda x: x[1])

def get_list_of_tags(df):
    list_of_tags = []

    #########
    # This is the part that takes absurdly long
    for i in range(len(df)):
        list_of_tags.append(get_best_word(df, i))
    better_list_of_tags = []
    #########

    for i in range(len(list_of_tags)):
        better_list_of_tags.append(list_of_tags[i][0])
    
    return better_list_of_tags

def create_prediction_column(df, words_df):
    df['predicted_tag'] = get_list_of_tags(words_df)




df = create_clean_columns('test.csv')
words_df = do_tfidf(df)
create_prediction_column(df, words_df)

sub_df = df[['id', 'predicted_tag']]
sub_df.columns = ['id', 'tags']
sub_df.to_csv('submission.csv', encoding='utf-8', index=False)