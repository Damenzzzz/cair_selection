import os, sys, glob, gc, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

MODEL = os.environ.get('MODEL', 'Qwen/Qwen3-8B')
NAME = os.environ.get('NAME', 'qwen8b')
FOLDS = [int(f) for f in sys.argv[1].split(',')]
BUDGET = float(os.environ.get('BUDGET_HOURS', '3')) * 3600
BS = 16
LR = 1e-4
MAX_LEN = 160
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA = os.path.dirname(glob.glob('/kaggle/input/**/CAIRLAB.csv', recursive=True)[0]) if os.path.exists('/kaggle/input') else '.'
SUB_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else 'submissions'
os.makedirs(SUB_DIR, exist_ok=True)

train_df = pd.read_csv(f'{DATA}/CAIRLAB.csv', header=None, names=['ID', 'Entity', 'Sentiment', 'Text'])
test_df = pd.read_csv(f'{DATA}/test.csv')
train_df['Text'] = train_df['Text'].fillna('')
test_df['Text'] = test_df['Text'].fillna('')
train_df = train_df[train_df['Text'].str.strip() != ''].reset_index(drop=True)
if os.environ.get('SMOKE'):
    train_df, test_df = train_df.iloc[:120].reset_index(drop=True), test_df.iloc[:8]

X = train_df['Entity'] + ' ' + train_df['Text']
y = train_df['Sentiment']
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups=train_df['ID']))

train_df['pos'] = train_df.groupby('ID').cumcount()
ratio = train_df['Text'].str.len() / train_df.groupby('ID')['Text'].transform('first').str.len()
keep = ((train_df['pos'] < 5) & (ratio > 0.5)).to_numpy()
is_orig = (train_df['pos'] == 0).to_numpy()

classes = sorted(y.unique())
y_num = y.map({c: i for i, c in enumerate(classes)}).values

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = device == 'cuda'

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def prompts(df):
    return [f'Entity: {e}\nTweet: {t}\nSentiment of the tweet toward the entity:' for e, t in zip(df['Entity'], df['Text'])]

def batches(df, shuffle=False):
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    texts = prompts(df)
    for i in range(0, len(df), BS):
        b = idx[i:i + BS]
        enc = tok([texts[j] for j in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt').to(device)
        yield enc, b

def predict(model, df):
    model.eval()
    out = []
    with torch.no_grad():
        for enc, _ in batches(df):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            out.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)

def load_model():
    if use_amp:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=['score'])
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=len(classes), quantization_config=bnb, torch_dtype=torch.float16, device_map={'': 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(classes)).float().to(device)
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    return get_peft_model(model, lora)

def train_fold(fold):
    tr_idx, va_idx = folds[fold]
    tr_idx = tr_idx[keep[tr_idx]]
    va_idx = va_idx[is_orig[va_idx]]
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]

    model = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    total = len(tr_df) // BS + 1
    start = time.time()
    progress = lambda step: max(step / total, (time.time() - start) / BUDGET)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, progress(s) / 0.05) * max(0.0, 1.0 - progress(s)))

    model.train()
    losses = []
    for step, (enc, b) in enumerate(batches(tr_df, shuffle=True)):
        if progress(step) >= 1:
            break
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            loss = model(**enc, labels=torch.tensor(y_num[tr_idx][b]).to(device)).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f'fold {fold} step {step}/{total} | {(time.time() - start) / 60:.0f} min | loss {np.nanmean(losses[-100:]):.4f}', flush=True)

    val_proba = predict(model, va_df)
    ok = val_proba.argmax(1) == y_num[va_idx]
    print(f'fold {fold}: оригиналы = {ok.mean():.4f} | шагов {step}/{total} | NaN {np.isnan(val_proba).mean():.3f}', flush=True)
    np.save(f'{SUB_DIR}/{NAME}_f{fold}_val.npy', val_proba)
    np.save(f'{SUB_DIR}/{NAME}_f{fold}_va_idx.npy', va_idx)
    np.save(f'{SUB_DIR}/{NAME}_f{fold}_test.npy', predict(model, test_df))

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

for fold in FOLDS:
    train_fold(fold)
