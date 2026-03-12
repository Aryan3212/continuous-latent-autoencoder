import yaml
with open('configs/exp0.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['train']['max_steps'] = 5
cfg['train']['log_interval_steps'] = 1
cfg['train']['eval_interval_steps'] = 2
cfg['train']['save_interval_steps'] = 4
cfg['train']['batch_size'] = 4
cfg['train']['val_batches'] = 2
cfg['train']['grad_accum_steps'] = 1

cfg['gan']['start_step'] = 2
cfg['eval']['asr']['steps'] = 2
cfg['eval']['asr']['batch_size'] = 4

with open('configs/exp0_test.yaml', 'w') as f:
    yaml.dump(cfg, f)
