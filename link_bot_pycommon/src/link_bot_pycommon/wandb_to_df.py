import pandas as pd
import wandb

NAME_KEY = 'method_name'


def wandb_to_df(config, user='petermitrano', project='propnet'):
    api = wandb.Api()

    # iterate once to find a list of all keys
    columns = [NAME_KEY]
    for runs in config['runs'].values():
        for run_id in runs:
            run = api.run(f'{user}/{project}/{run_id}')
            for k in run.config.keys():
                if k not in columns:
                    columns.append(k)
            for k in run.summary.keys():
                if k not in columns:
                    columns.append(k)

    df = []
    for method_name, runs in config['runs'].items():
        for run_id in runs:
            run = api.run(f'{user}/{project}/{run_id}')
            row = [method_name]
            for k in columns:
                if k in run.config:
                    row.append(run.config[k])
                elif k in run.summary:
                    row.append(run.summary[k])
                elif k == NAME_KEY:
                    pass
                else:
                    row.append(None)
            df.append(row)
    return pd.DataFrame(df, columns=columns)