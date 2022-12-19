# Some utility imports
import sys
from torchmetrics import ConfusionMatrix
from nemo.utils.exp_manager import exp_manager
import pytorch_lightning as pl
import torch
import os
from omegaconf import OmegaConf
import argparse
from torchmetrics import Accuracy, F1Score, AUROC, ConfusionMatrix
from loguru import logger


# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr


@torch.no_grad()
def extract_logits(model, dataloader):
    logits_buffer = []
    label_buffer = []

    # Follow the above definition of the test_step
    for batch in dataloader:
        audio_signal, audio_signal_len, labels, labels_len = batch
        # print(audio_signal.size())
        # print(audio_signal_len.size())
        logits = model(input_signal=audio_signal,
                       input_signal_length=audio_signal_len)

        logits_buffer.append(logits)
        label_buffer.append(labels)
        print(".", end='')
    print()

    print("Finished extracting logits !")
    logits = torch.cat(logits_buffer, 0)
    labels = torch.cat(label_buffer, 0)
    return logits, labels


# metrics: VAD binary F1, FER, AUC, Pfa, Pmiss, ACC

def compute_metrics(pred, labels):
    accuracy = Accuracy('binary', num_classes=2)
    f1_score = F1Score('binary', num_classes=2)
    auroc = AUROC('binary', num_classes=2)
    matrix = ConfusionMatrix('binary', num_classes=2)

    acc = accuracy(pred, labels)
    f1score = f1_score(pred, labels)
    auc = auroc(pred, labels)
    tn, fp, fn, tp = matrix(pred, labels).ravel()
    fer = 100 * ((fp + fn) / pred.size()[0])
    p_miss = 100 * (fn / (fn + tp))
    p_fa = 100 * (fp / (fp + tn))

    return acc, f1score, auc, fer, p_miss, p_fa


def getfile_outlogger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/marblenet.yml")
    parser.add_argument("--log_file", default="result.log")

    args = parser.parse_args()

    # This line will print the entire config of the MarbleNet model
    config_path = args.config
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    trainer = pl.Trainer(**config.trainer)
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    log_file = args.log_file
    output_file = exp_dir/log_file
    logger = getfile_outlogger(str(output_file))
    vad_model = nemo_asr.models.EncDecClassificationModel(
        cfg=config.model, trainer=trainer)

    trainer.fit(vad_model)
    trainer.test(vad_model, ckpt_path=None)

    vad_model.setup_test_data(config.model.test_ds)
    test_dl = vad_model._test_dl

    cpu_model = vad_model.cpu()
    cpu_model.eval()
    logits, labels = extract_logits(cpu_model, test_dl)
    _, pred = logits.topk(1, dim=1, largest=True, sorted=True)
    pred = pred.squeeze()

    # compute metrics
    acc, f1score, auc, fer, p_miss, p_fa = compute_metrics(pred, labels)
    logger.info(
        f'acc: {acc}, f1score: {f1score}, auc: {auc}, fer: {fer}, p_miss: {p_miss}, p_fa: {p_fa}')
    # 参数量

    # 运行时间

    # export model to onnx
    onnx_model_file = exp_dir/'marblenet.onnx'
    audio_signal = torch.randn(10, 64, 64)
    audio_signal_len = torch.full(
        size=(audio_signal.shape[0], ), fill_value=10)

    # vad_model.export(output=str(onnx_model_file),
    #                  input_example=tuple([audio_signal, audio_signal_len]))
    # dynamic_axes = {"audio_signal": [0], "logits": [1]}
    vad_model.export(output=str(onnx_model_file),
                     input_example=tuple([audio_signal, audio_signal_len]))
