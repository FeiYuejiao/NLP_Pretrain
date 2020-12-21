lr = 0.001
model_path = 'model/IC_models/baseline_lr0.001_bs256/'
crop_size = 32
log_step = 50
save_step = 1500

num_epochs = 500
batch_size = 256
num_workers = 8

checkpoint = './model/IC_models/baseline_lr0.001_bs256/new_encoder-49-1500.ckpt'
loading = True

# Model parameters
model = dict(
    net='resnet101',
    embed_size=256,
    hidden_size=512,
    num_layers=1,
)
