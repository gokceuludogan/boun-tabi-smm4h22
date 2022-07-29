import pandas as pd
import preprocessor as p
import click
import re 
import emoji
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv

def read_data(data_dir):
    tweet_col_names = ['tweet_id', 'tweet']
    if data_dir.name.lower() == 'task1a':
        output_col_names = ['tweet_id', 'label']
        output_file = 'class'
    elif data_dir.name.lower() == 'task1b':
        output_col_names = ['tweet_id', 'type', 'begin', 'end', 'label']
        output_file = 'spans'

    train_tweets = pd.read_csv(data_dir / 'train' / 'train_tweets.tsv', sep='\t', names=tweet_col_names, header=None, quoting=csv.QUOTE_NONE)
    train_labels = pd.read_csv(data_dir / 'train' / f'train_{output_file}.tsv', sep='\t',  names=output_col_names, header=None)
    df_train = pd.merge(train_tweets, train_labels, on='tweet_id', how='left').drop_duplicates('tweet_id')


    valid_tweets = pd.read_csv(data_dir / 'valid' / 'tweets.tsv', sep='\t', names=tweet_col_names, header=None, quoting=csv.QUOTE_NONE)
    valid_labels = pd.read_csv(data_dir / 'valid' / f'{output_file}.tsv', sep='\t',  names=output_col_names, header=None)
    df_valid = pd.merge(valid_tweets, valid_labels, on='tweet_id', how='left').drop_duplicates('tweet_id')

    if data_dir.name.lower() == 'task1a':
        df_train['label'].fillna('NoADE', inplace=True)
        df_valid['label'].fillna('NoADE', inplace=True)
    elif data_dir.name.lower() == 'task1b':
        df_train['label'].fillna('none', inplace=True)
        df_valid['label'].fillna('none', inplace=True)

    return df_train, df_valid

def preprocess_tweet(raw_str):
    # p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)
    p.set_options(p.OPT.URL)
    
    try:
        # Decode emojis
        raw_str = emoji.demojize(raw_str)
        clean_str = p.clean(raw_str)
    except:
        #print(raw_str)
        clean_str = raw_str

    # Removes _ at the end of placeholders 
    clean_str = re.sub(r"[_]+", r"", clean_str)

    return clean_str

def preprocess_frame(df, prefix, targets):
    df['input_text'] = df['tweet'].apply(preprocess_tweet)
    df['prefix'] = prefix

    if type(targets) == dict:
        df['target_text'] = df['label'].apply(lambda l: targets[l]) 
    else:
        df['target_text'] = df['label']

    return df


def prepare_data(data_dir, task_name, targets):
    df_train, df_test = read_data(data_dir)

    df_train = preprocess_frame(df_train, task_name, targets)
    df_test = preprocess_frame(df_test, task_name, targets)

    # Split train instances into train and validation sets. 
    df_train, df_val = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=25)

    # TODO: Perform augmentation and under/oversampling. 

    # Export data splits. 
    df_train[['tweet_id', 'tweet', 'prefix', 'input_text', 'target_text']].to_csv(data_dir / 'train.csv', index=False)
    df_val[['tweet_id', 'tweet', 'prefix', 'input_text', 'target_text']].to_csv(data_dir / 'val.csv', index=False)
    df_test[['tweet_id', 'tweet', 'prefix', 'input_text', 'target_text']].to_csv(data_dir / 'test.csv', index=False)



@click.command()
@click.argument('data_dir', default='data/')
def main(data_dir):
    path = Path(data_dir)
    # TODO: fill targets dictionary
    task1a_targets = {'ADE': 'adverse event problem', 'NoADE': 'healthy okay'}
    prepare_data(path / 'Task1a', 'assert ade', task1a_targets)
    prepare_data(path / 'Task1b', 'ner ade', 'span')

if __name__ == "__main__":
    main()

