lr = 0.001
model_path = 'model/IC_models/densenet169_lr_0.001/'
crop_size = 32
log_step = 10
save_step = 500

num_epochs = 400
batch_size = 256
num_workers = 8
loading = False

# lr

# Model parameters
model = dict(
    net='densenet169',
    embed_size=256,
    hidden_size=512,
    num_layers=1,
    resnet=101
)
