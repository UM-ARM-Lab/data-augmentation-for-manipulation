import pandas as pd
import wandb


def wandb_to_df(user='armlab', project='propnet'):
    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=None, order="-created_at", per_page=50)
    columns = runs_to_columns(runs)
    df = runs_to_df(runs, columns)
    return df


def runs_to_columns(runs):
    columns = ['tags', 'run_id']
    for run in runs:
        for k in run.config.keys():
            if k not in columns:
                columns.append(k)
        for k in run.summary.keys():
            if k not in columns:
                columns.append(k)
    return columns


def wandb_to_df_with_config(config, user='armlab', project='propnet'):
    api = wandb.Api()

    # iterate once to find a list of all keys
    runs = []
    for run_ids in config['runs'].values():
        for run_id in run_ids:
            run = api.run(f'{user}/{project}/{run_id}')
            runs.append(run)

    columns = runs_to_columns(runs)
    df = runs_to_df(runs, columns)
    return df


def runs_to_df(runs, columns):
    df = []
    for run in runs:
        row = []
        for k in columns:
            if k in run.config:
                row.append(run.config[k])
            elif k in run.summary:
                row.append(run.summary[k])
            elif k == 'tags':
                row.append(run.tags)
            elif k == 'run_id':
                row.append(run.id)
            else:
                row.append(None)
        df.append(row)
    return pd.DataFrame(df, columns=columns)
