lr = 0.0005
model_path = 'model/IC_models/baseline_lr_0.0005/'
crop_size = 32
log_step = 50
save_step = 1500

num_epochs = 500
batch_size = 256
num_workers = 8

loading = False

# Model parameters
model = dict(
    net='resnet101',
    embed_size=256,
    hidden_size=512,
    num_layers=1
)
