import pandas as pd
import click
import difflib
import random
import json
from pathlib import Path
from itertools import combinations

def majority_voting(frame_list):
	df = pd.concat(frame_list)
	df_counts = df.groupby(['tweet_id', 'predicted_text']).size().reset_index(name='counts')
	return df_counts.sort_values('counts', ascending=False).drop_duplicates('tweet_id').sort_index()

def find_intersection(span_list):
	matches = []
	for seq1, seq2 in combinations(span_list, 2):
		blocks = difflib.SequenceMatcher(None, seq1.lower(), seq2.lower()).get_matching_blocks()
		max_length =  0 
		match = None
		for s1_ix, s2_ix, length in blocks:
			if length > max_length:
				match = seq1[s1_ix:(s1_ix+length+1)]
		if match is not None and len(match) >= 5:
			matches.append(match)

		print('Seq1:', seq1, '\nSeq2:', seq2, '\nMatch:', match)
	if len(matches) == 0:
		print('\n', span_list, 'no intersection')
		return random.choice(span_list)
	match_counts = {}
	for match in matches:
		match_counts[match] = 0
		for span in span_list:
			match_counts[match] += match in span
	print('\n', span_list, max(match_counts, key=match_counts.get))
	return max(match_counts, key=match_counts.get)

def span_ensemble(frame_list):
	df = pd.concat(frame_list)
	df_counts = df.groupby(['tweet_id', 'predicted_text']).size().reset_index(name='counts')
	# df_counts.sort_values(['counts', 'tweet_id'], ascending=False).to_csv('ner_counts.csv')
	df_majority = df_counts[df_counts['counts'] >= len(frame_list) / 2].reset_index()
	df_majority_ade = df_majority[df_majority['predicted_text'] != 'none']
	df_intersection = df_counts[~df_counts['tweet_id'].isin(df_majority['tweet_id'].tolist())]
	df_intersection = df_intersection[df_intersection['predicted_text'] != 'none']
	df_spans = df_intersection.groupby('tweet_id')['predicted_text'].apply(list).reset_index(name='all_spans')
	df_spans['predicted_text'] = df_spans['all_spans'].apply(find_intersection)
	return pd.concat([df_majority[['tweet_id', 'predicted_text']], df_spans[['tweet_id', 'predicted_text']]])

@click.command()
@click.argument('config_path')
def main(config_path):
	with open(config_path) as f:
		config = json.loads(f.read())
	input_dir = Path(config['input_dir'])
	predictions = {}
	for folder in input_dir.iterdir():
		if folder.name in config['predictions']:
			df = pd.read_csv(folder / 'test_predictions.csv', index_col=False)
			df['predicted_text'] = df['predicted_text'].apply(lambda s: s.strip('.').lower())
			predictions[folder.name] = df
			task = 'classification' if folder.name.startswith('assert') else 'detection'

	all_frames = list(predictions.values())
	if task == 'classification':
		df_majority = majority_voting(all_frames)
		df_output = pd.merge(df_majority, all_frames[0][['tweet_id', 'tweet']], how='left')
	else:
		df_output = span_ensemble(all_frames)
		df_output = pd.merge(df_output, all_frames[0][['tweet_id', 'tweet']], how='left')

	output_folder = input_dir / config['output'] 
	output_folder.mkdir(exist_ok=True, parents=True)
	df_output.to_csv(output_folder / 'test_predictions.csv', index=False)


if __name__ == "__main__":
    main()
