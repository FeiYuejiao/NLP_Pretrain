lr = 0.001
model_path = 'model/IC_models/resnet50_lr_0.001_resnetconv1kernel_7/'
crop_size = 32
log_step = 50
save_step = 1500

num_epochs = 400
batch_size = 256
num_workers = 8

loading = False

# Model parameters
model = dict(
    net='resnet50',
    embed_size=256,
    hidden_size=512,
    num_layers=2,
)
