import pandas as pd
import click
from pathlib import Path

def prepare_dataframe(df, input_template, output_templates):
    df['prefix'] = 'ner ade'
    df['input_text'] = df['input_text'].apply(lambda s: input_template.format(s))
    df['target_text'] = df['target_text'].apply(lambda s: output_templates[0] if s == 'none' else output_templates[1].format(s))
    return df 

@click.command()
@click.argument('data_dir', default='data/')
def main(data_dir):
    path = Path(data_dir)
    input_dir = path / 'raw_data' / 'ner_ade' 
    train = pd.read_csv(input_dir / 'train_ner_ade_smm4h_task2.csv')
    val = pd.read_csv(input_dir / 'eval_ner_ade_smm4h_task2.csv')
    test = pd.read_csv(input_dir / 'test_ner_ade_smm4h_task2.csv')

    prompts = {
        "prompt1": {
            "input": "Is there a negative drug effect in: {}",
            "output":  ["There isn't a negative drug effect.", "{} is a negative drug effect."]
        },
        "prompt2": {
            "input": "Did the patient suffer from a side effect? {}",
            "output": ["No, the patient didn’t suffer from a side effect.", "Yes, the patient suffered from {}."]
        },
        "prompt3": {
            "input": "{} Did the patient suffer from a side effect?",
            "output": ["No, the patient didn’t suffer from a side effect.", "Yes, the patient suffered from {}."]
        }
    }

    for prompt, templates in prompts.items():
        output_dir = path / prompt / 'ner_ade' 
        output_dir.mkdir(exist_ok=True, parents=True)

        ptrain = prepare_dataframe(train.copy(), templates["input"], templates["output"])
        pval = prepare_dataframe(val.copy(), templates["input"], templates["output"])
        ptest = prepare_dataframe(test.copy(), templates["input"], templates["output"])

        ptrain.to_csv(output_dir / 'train_ner_ade_smm4h_task2.csv', index=False)
        pval.to_csv(output_dir / 'eval_ner_ade_smm4h_task2.csv', index=False)
        ptest.to_csv(output_dir / 'test_ner_ade_smm4h_task2.csv', index=False)

if __name__ == "__main__":
    main()

