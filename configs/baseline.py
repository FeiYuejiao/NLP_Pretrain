lr = 0.001
model_path = 'model/IC_models/test/'
crop_size = 32
log_step = 10
save_step = 500

num_epochs = 500
batch_size = 256
num_workers =4

checkpoint = './model/IC_models/test/new_encoder-46-1500.ckpt'
loading = True

# Model parameters
model = dict(
    embed_size=256,
    hidden_size=512,
    num_layers=1,
    net='resnet101'
)
