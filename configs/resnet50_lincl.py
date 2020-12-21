lr = 0.001


model_path = 'model/IC_models/resnet50_imagenet_lincl/'
task = 'imagenet_cls'
crop_size = 224
log_step = 50
save_step = 1500

num_epochs = 400
batch_size = 16
num_workers = 16
seed = None
loading = False

# Model parameters
model = dict(
    net='resnet50',
    embed_size=256,
    hidden_size=512,
    num_layers=2,
)

# gpu config
gpu = 0
# dist_url = 'tcp://localhost:10001'
dist_url = None
world_size = -1
# world_size = 1
rank = 0
distributed = False
# distributed = True
# dist_backend = 'nccl'
dist_backend = None
# path
vocab_path = 'data/vocab.pkl'
image_dir = 'data/train2014'
caption_path = 'data/annotations/captions_train2014.json'
