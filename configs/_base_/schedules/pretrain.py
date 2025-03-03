lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6))

param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=100,
        by_epoch=True,
        eta_max=lr,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=100)

auto_scale_lr = dict(enable=False, base_batch_size=32)
