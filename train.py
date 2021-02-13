from model import MusicTransformer
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule

import data
import utils
import preprocess
import datetime
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

# set config
# parser = custom.get_argument_parser()
# args = parser.parse_args()
# config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

model_dir = "/Users/mmg/Google Drive/data/final_fantasy_pt"
data_dir = "/Users/mmg/Google Drive/data/final_fantasy"

# load data
data_files = list(utils.find_files_by_extensions(data_dir, ['.pb']))

if len(data_files) == 0:
    print("no data files found!")
    exit(1)
          
train_files, valid_files = data.train_test_split(data_files)
midi_encoder = preprocess.MidiEncoder(
    num_velocity_bins=2,
    steps_per_second=100,
    min_pitch=21,
    max_pitch=98
)
max_seq = 1024
embedding_dim = 256
batch_size = 2
label_smooth = 0.1
epochs = 100

dataset = data.Data(
    sequences={
        'train': data.load_seq_files(train_files),
        'eval': data.load_seq_files(valid_files)
    },
    midi_encoder=midi_encoder,
)

# define model
mt = MusicTransformer(
    embedding_dim=embedding_dim,
    vocab_size=midi_encoder.vocab_size,
    num_layer=5,
    max_seq=max_seq,
    dropout=0.1,
    debug=False, 
    # loader_path=config.load_path
)
mt.to(device)
opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(embedding_dim, optimizer=opt)

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(label_smooth, midi_encoder.vocab_size, midi_encoder.token_pad),
    'bucket':  LogitsBucketting(midi_encoder.vocab_size)
})

print(mt)
print('| Summary - Device Info : {}'.format(torch.cuda.device))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+current_time+'/train'
eval_log_dir = 'logs/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

# Train Start
print(">> Train start...")
idx = 0
for e in range(epochs):
    print(">>> [Epoch was updated]")
    for b in range(len(dataset) // batch_size):
        scheduler.optimizer.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
            batch_x = torch.from_numpy(batch_x).contiguous().to(device, non_blocking=True, dtype=torch.int)
            batch_y = torch.from_numpy(batch_y).contiguous().to(device, non_blocking=True, dtype=torch.int)
        except IndexError:
            continue

        start_time = time.time()
        mt.train()
        sample = mt.forward(batch_x)
        metrics = metric_set(sample, batch_y)
        loss = metrics['loss']
        loss.backward()
        scheduler.step()
        end_time = time.time()

        print("[Loss]: {}".format(loss))

        train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=idx)
        train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
        train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)

        # result_metrics = metric_set(sample, batch_y)
        if b % 100 == 0:
            mt.eval()
            eval_x, eval_y = dataset.slide_seq2seq_batch(2, max_seq, 'eval')
            eval_x = torch.from_numpy(eval_x).contiguous().to(device, dtype=torch.int)
            eval_y = torch.from_numpy(eval_y).contiguous().to(device, dtype=torch.int)

            eval_preiction, weights = mt.forward(eval_x)

            eval_metrics = metric_set(eval_preiction, eval_y)
            torch.save(mt.state_dict(), model_dir+'/train-{}.pth'.format(e))
            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
                for i, weight in enumerate(weights):
                    attn_log_name = "attn/layer-{}".format(i)
                    utils.attention_image_summary(
                        attn_log_name, weight, step=idx, writer=eval_summary_writer)

            eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=idx)
            eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
            eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=idx)

            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))
        torch.cuda.empty_cache()
        idx += 1

        # switch output device to: gpu-1 ~ gpu-n
        sw_start = time.time()
        if torch.cuda.device_count() > 1:
            mt.output_device = idx % (torch.cuda.device_count() -1) + 1
        sw_end = time.time()

torch.save(single_mt.state_dict(), model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


