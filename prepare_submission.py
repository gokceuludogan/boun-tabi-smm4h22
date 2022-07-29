import pandas as pd
import click
from pathlib import Path


def format_submission(input_path, output_path, task_type='classification'):
	df = pd.read_csv(input_path, index_col=False)
	df['predicted_class'] = 'ADE'
	df['predicted_text'] = df['predicted_text'].apply(lambda s: s.strip('.'))
	if task_type == 'classification':
		df = df[df['predicted_text'] == 'adverse event problem']
		df['span_start'] = '0'
		df['span_end'] = '0'
		df['span'] = '-'
	else:
		df = df[df['predicted_text'] != 'none']
		df['span_start'] = df.apply(lambda row: row['tweet'].lower().find(row['predicted_text'].lower()), axis=1) 
		df['span_end'] = df.apply(lambda row: row['tweet'].lower().find(row['predicted_text'].lower()) + len(row['predicted_text']), axis=1) 
		df['span'] = df['predicted_text']
	df = df[df['span_start'] != -1]
	df[['tweet_id', 'predicted_class', 'span_start', 'span_end', 'span']].to_csv(output_path, index=False,  header=None, sep='\t')


@click.command()
@click.argument('predictions_dir', default='')
@click.argument('output_dir', default='submissions_val')
def main(predictions_dir, output_dir):
	output = Path(output_dir)
	pred_dir = Path(predictions_dir)
	for folder in pred_dir.iterdir():
		task = 'classification' if folder.name.startswith('assert') else 'detection'
		format_submission(folder / 'test_predictions.csv', output / f'{folder.name}.tsv', task)


if __name__ == "__main__":
    main()
