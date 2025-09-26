import os
import json
import sys

def save_responses_to_jsonl(results: list[dict], append: bool = True) -> None:
    """
    Save model responses to JSONL file, each line contains:
    {id, claim, response, evidence_score, accuracy}
    Supports append mode, default is True.
    """
    # Read experiment name from environment variable for filename
    experiment_name = os.getenv('EXPERIMENT_NAME')
    if not experiment_name:
        raise EnvironmentError('Environment variable EXPERIMENT_NAME is not set')

    # JSONL file path
    file_path = f"case/{experiment_name}.jsonl"

    # Decide whether to save: read SAVE_JSONL environment variable or command line argument --save_jsonl
    save_flag_env = os.getenv('SAVE_JSONL')
    save_flag_arg = None
    for arg in sys.argv:
        if arg.startswith('--save_jsonl='):
            _, val = arg.split('=', 1)
            save_flag_arg = val
            

    # Prefer command line arguments, then environment variables
    if save_flag_arg is not None:
        save_flag = save_flag_arg.lower() == 'true'
    elif save_flag_env is not None:
        save_flag = save_flag_env.lower() == 'true'
    else:
        save_flag = False

    if not save_flag:
        print('SAVE_JSONL flag is False, skipping save')
        return
    
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Write file: decide mode based on append parameter
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    action = 'Appended' if append else 'Saved'
    print(f"{action} {len(results)} responses to {file_path}")
