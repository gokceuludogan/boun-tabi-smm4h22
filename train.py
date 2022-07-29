from t5_model import *
from utils import *
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
  

@click.command()
@click.argument('task_type', default='assert_ade')
@click.argument('task_name', default='smm4h_task1')
@click.argument('data_dir', default='combiner_data')
@click.argument('output_dir', default='assert_ade_baseline')
@click.argument('model_path', default='t5-base')
def main(task_type, task_name, data_dir, output_dir, model_path):
    '''
    Model hyper-parameters, for detailed list of hyperparamets checkout model_args.py and global_args.py in /models/config
    The paramters description can also be found on https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    '''
    model_args = {
        "max_seq_length": 130,
        "train_batch_size": 16,
        "eval_batch_size": 8,
        "num_train_epochs": 12,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 500,
        "evaluate_during_training_verbose": True,
        "n_gpu": 1,
        "learning_rate": 1e-4,
        "manual_seed" : 25,

        "evaluate_generated_text": True,
        "gradient_accumulation_steps": 4,

        "use_multiprocessing": False,
        "fp16": True,

        "save_steps": -1,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "preprocess_input_data": True,
        "overwrite_output_dir": True,
        "output_dir": output_dir,
        "best_model_dir": f'{output_dir}/best_model',

        "wandb_project": None
    }

    # Train the T5 model on the given task type and name
    data_path = Path.cwd() / 'data' / data_dir / task_type
    train_df = pd.read_csv(data_path / f'train_{task_type}_{task_name}.csv')
    eval_df = pd.read_csv(data_path / f'eval_{task_type}_{task_name}.csv')
    test_df = pd.read_csv(data_path / f'test_{task_type}_{task_name}.csv')

    model = T5Model(model_path, args=model_args)
    if model_path == 't5-base':
        model.train_model(train_df, eval_data=eval_df)

    if "assert" in task_type:
        model.eval_model(test_df, metrics=compute_metric_assert, write_to_file=output_dir + "/test_predictions.csv")
    else:
        if "prompt" in data_dir:
            model.eval_model(test_df, metrics=compute_metric_ner, prompt=data_dir, write_to_file=output_dir + "/test_predictions.csv")
        else:    
            model.eval_model(test_df, metrics=compute_metric_ner,write_to_file=output_dir + "/test_predictions.csv")
        match_df = eval_ner("ner_preds.csv")
        calc_score(match_df)    

if __name__ == "__main__":
    main()
