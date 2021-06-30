import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
# from tensorboard.plugins.hparams.api_pb2 import Experiment
# from tensorboard.uploader.proto.experiment_pb2 import Experiment
from comet_ml import Experiment
import MODELS
from cer import cer, wer
from gcommand_loader import GCommandLoader
from project.utils.functions import train_audio_transforms, valid_audio_transforms
from textTr import TextTransform


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(TextTransform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(TextTransform.int_to_text(decode))
    return decodes, targets

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(TextTransform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths





class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    experiment.log_metric('cer', avg_cer, step=iter_meter.get())
    experiment.log_metric('wer', avg_wer, step=iter_meter.get())

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))






# def run(hparams,model,train_loader,test_loader):
#
#     train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
#     test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)
#     torchaudio.transforms.FrequencyMasking()
#     torchaudio.transforms.TimeMasking()
#
#     train_audio_transforms = nn.Sequential(
#         torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
#         torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
#         torchaudio.transforms.TimeMasking(time_mask_param=35)
#     )
#
#     valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
#
#     text_transform = TextTransform()
#
#     optimizer = torch.optim.AdamW(model.parameters(), hparams['learning_rate'])
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
#         max_lr=hparams['learning_rate'],
#         steps_per_epoch=int(len(train_loader)),
#         epochs=hparams['epochs'],
#         anneal_strategy='linear')
#
#     use_cuda = torch.cuda.is_available()
#     torch.manual_seed(7)
#
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     criterion = torch.nn.CTCLoss(blank=28).to(device)
#
#     experiment = Experiment(api_key='dummy_key', project_name=project_name)
#     experiment.set_name(exp_name)
#
#     # track metrics
#     experiment.log_metric('loss', loss.item())




#api_key='dummy_key',disabled=True

def main(learning_rate=5e-4, batch_size=20, epochs=10,
    train_url="train-clean-100", test_url="test-clean",
    experiment=Experiment(api_key='dummy_key',disabled=True)):

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = GCommandLoader('./data/train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    # train_loader = data.DataLoader(dataset=train_dataset,
    #                             batch_size=hparams['batch_size'],
    #                             shuffle=True,
    #                             collate_fn=lambda x: data_processing(x, 'train'),
    #                             **kwargs)

    dataset = GCommandLoader('./data/do')

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)


    model = MODELS.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = torch.optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')

    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
        test(model, device, test_loader, criterion, epoch, iter_meter, experiment)



def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    # Model parser
    parser = MODELS.SpeechRecognitionModel.add_model_specific_args(parent_parser)
    # Data
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--data_root", default="/mnt/kingston/datasets/", type=str)
    parser.add_argument(
        "--data_train",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
    )
    parser.add_argument("--data_test", default=["test-clean"])
    # Training params (opt)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    # parser.add_argument("--precission", default=16, type=int)
    parser.add_argument("--early_stop_metric", default="wer", type=str)
    parser.add_argument("--logs_path", default="runs/", type=str)
    parser.add_argument("--experiment_name", default="DeepSpeech", type=str)
    parser.add_argument("--early_stop_patience", default=3, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    # Precission args
    parser.add_argument("--amp_level", default="02", type=str)
    parser.add_argument("--precision", default=32, type=int)

    args = parser.parse_args()
    # ---------------------
    # RUN TRAINING
    # ---------------------


if __name__ == "__main__":
    main()
