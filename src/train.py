
import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import NepaliEnglishDataset
from src.model.model import build_transformer
from src.utils.tokenizer import load_nepali_tokenizer, load_english_tokenizer
from src.utils.masks import create_masks
from src.utils.translate import translate

try:
    from sacrebleu.metrics import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("sacrebleu not installed. Run: pip install sacrebleu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Hyperparameter Controller
# ─────────────────────────────────────────────────────────────────────────────

class DynamicHyperparamController:
    """
    Monitors overfitting gap and dynamically adjusts weight_decay only.
    LR is handled exclusively by ReduceLROnPlateau scheduler.
    Dropout is fixed — prints RESTART warning when severe overfit detected.

    Gap thresholds:
      < 0.3   Healthy       - no action
      0.3-0.6 Mild overfit  - increase weight_decay
      0.6-1.0 Moderate      - increase weight_decay
      > 1.0   Severe        - increase weight_decay + print RESTART warning
    """

    def __init__(self, optimizer, writer,
                 wd_init=0.01, wd_max=0.1, wd_step=0.01,
                 patience=3,   cooldown=3):

        self.optimizer      = optimizer
        self.writer         = writer

        self.wd             = wd_init
        self.wd_max         = wd_max
        self.wd_step        = wd_step

        self.patience       = patience
        self.cooldown       = cooldown
        self.overfit_streak = 0
        self.last_action    = 0

    def _set_weight_decay(self, new_wd):
        for group in self.optimizer.param_groups:
            group['weight_decay'] = new_wd
        self.wd = new_wd

    def _get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, epoch, train_loss, val_loss):
        """Call once per epoch after validation."""
        gap        = val_loss - train_loss
        current_lr = self._get_current_lr()
        actions    = []

        # Severity
        if gap < 0.3:
            severity = "healthy"
            self.overfit_streak = 0
        elif gap < 0.6:
            severity = "mild"
            self.overfit_streak += 1
        elif gap < 1.0:
            severity = "moderate"
            self.overfit_streak += 1
        else:
            severity = "severe"
            self.overfit_streak += 1

        cooldown_ok = (epoch - self.last_action) >= self.cooldown

        if self.overfit_streak >= self.patience and cooldown_ok:
            self.last_action = epoch

            # Mild/Moderate/Severe: bump weight decay
            if severity in ("mild", "moderate", "severe"):
                new_wd = min(self.wd + self.wd_step, self.wd_max)
                if new_wd != self.wd:
                    old_wd = self.wd
                    self._set_weight_decay(new_wd)
                    actions.append(f"weight_decay: {old_wd:.3f} -> {new_wd:.3f}")

            # Severe: print restart warning only (user changes dropout manually)
            if severity == "severe":
                actions.append("[RESTART ADVISED] increase dropout manually and retrain from epoch 0")

            self.overfit_streak = 0

        # Log to TensorBoard
        self.writer.add_scalar("Hyperparams/Weight_Decay",   self.wd,             epoch)
        self.writer.add_scalar("Hyperparams/Learning_Rate",  current_lr,          epoch)
        self.writer.add_scalar("Diagnostics/Overfit_Gap",    gap,                 epoch)
        self.writer.add_scalar("Diagnostics/Overfit_Streak", self.overfit_streak, epoch)

        return {
            "gap":          gap,
            "severity":     severity,
            "weight_decay": self.wd,
            "lr":           current_lr,
            "actions":      actions,
        }

    def summary(self):
        return (
            f"  weight_decay  : {self.wd:.4f}\n"
            f"  learning_rate : {self._get_current_lr():.2e}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_accuracy(logits, targets, pad_token):
    preds   = torch.argmax(logits, dim=-1)
    mask    = (targets != pad_token)
    n       = mask.sum().item()
    if n == 0:
        return 0.0
    correct = (preds == targets) & mask
    return correct.sum().item() / n


def filter_long_sentences(df, ne_tokenizer, en_tokenizer, max_len=128):
    ne_lengths = df['nepali_sent'].apply(lambda x: len(ne_tokenizer.encode(str(x)).ids))
    en_lengths = df['english_sent'].apply(lambda x: len(en_tokenizer.encode(str(x)).ids))
    mask     = (ne_lengths <= max_len - 2) & (en_lengths <= max_len - 2)
    filtered = df[mask].reset_index(drop=True)
    print(f"  {len(df)} -> {len(filtered)} sentences ({len(df) - len(filtered)} removed)")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# BLEU Score Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_bleu(model, val_df, ne_tokenizer, en_tokenizer, device,
                   max_samples=500, max_len=128):
    if not SACREBLEU_AVAILABLE:
        return None

    model.eval()
    sample_df  = val_df.sample(min(max_samples, len(val_df)), random_state=42)
    hypotheses = []
    references  = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                       desc="BLEU evaluation", leave=False):
        src_sentence = str(row['nepali_sent'])
        ref_sentence = str(row['english_sent'])
        try:
                
            hypothesis = translate(
                model, src_sentence,
                ne_tokenizer, en_tokenizer,
                device, max_len=max_len,
                beam_width=5              
             )
            hypotheses.append(hypothesis)
            references.append(ref_sentence)
        except Exception:
            continue

    if not hypotheses:
        return None

    bleu   = BLEU(effective_order=True)
    result = bleu.corpus_score(hypotheses, [references])
    return result.score


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, optimizer, loss_fn,
                epochs, patience, save_dir, model_name, checkpoint_path,
                tensorboard_dir=None, scheduler=None,
                device=None, pad_id_ne=0, pad_id_en=0,
                ne_tokenizer=None, en_tokenizer=None, val_df_ref=None,
                dynamic_hp=True,
                wd_init=0.01, wd_max=0.1):

    os.makedirs(save_dir, exist_ok=True)

    # ── Per-epoch checkpoint directory ────────────────────────────────────────
    epoch_ckpt_dir = os.path.join(save_dir, "epoch_checkpoints")
    os.makedirs(epoch_ckpt_dir, exist_ok=True)

    best_model_full_path = os.path.join(save_dir, f"best_{model_name}")
    checkpoint_full_path = os.path.join(save_dir, checkpoint_path)
    log_file_path        = os.path.join(save_dir, "training_logs.csv")

    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(save_dir, "tensorboard")

    start_epoch                = 0
    best_val_loss              = float('inf')
    history                    = []
    global_step                = 0
    epochs_without_improvement = 0
    tb_logdir                  = None

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if os.path.exists(checkpoint_full_path):
        print(f"\nFound checkpoint at {checkpoint_full_path}. Resuming...")
        checkpoint = torch.load(checkpoint_full_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch                = checkpoint["epoch"] + 1
        global_step                = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss") or float("inf")
        epochs_without_improvement = checkpoint.get("patience_counter", 0)
        history                    = checkpoint.get("history", [])
        tb_logdir                  = checkpoint.get("tb_logdir")
        restored_wd = checkpoint.get("current_wd", wd_init)
        for group in optimizer.param_groups:
            group['weight_decay'] = restored_wd
        print(f"Resuming from Epoch {start_epoch}, Step {global_step}, WD={restored_wd:.4f}")
    else:
        print(f"\nNo checkpoint found. Starting fresh on {device}")

    if tb_logdir is None:
        run_name  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logdir = os.path.join(tensorboard_dir, run_name)

    writer = SummaryWriter(log_dir=tb_logdir)
    print(f"TensorBoard: tensorboard --logdir={os.path.abspath(tensorboard_dir)}\n")

    # ── Dynamic hyperparameter controller ─────────────────────────────────────
    hp_controller = None
    if dynamic_hp:
        actual_wd_init = optimizer.param_groups[0]['weight_decay']
        hp_controller = DynamicHyperparamController(
            optimizer=optimizer,
            writer=writer,
            wd_init=actual_wd_init, wd_max=wd_max,
        )
        print(f"Dynamic HP enabled — weight_decay: {actual_wd_init} -> max {wd_max}\n")

    for epoch in range(start_epoch, epochs):

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_iterator   = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_train_loss = 0.0
        epoch_steps      = 0

        for batch in train_iterator:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)

            trg_input    = trg[:, :-1]
            trg_expected = trg[:, 1:]

            src_mask, tgt_mask = create_masks(src, trg_input, pad_id_ne, pad_id_en)

            encoder_output = model.encode(src, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, trg_input, tgt_mask)
            logits         = model.project(decoder_output)

            loss = loss_fn(logits.view(-1, logits.size(-1)), trg_expected.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step      += 1
            epoch_train_loss += loss.item()
            epoch_steps      += 1

            if global_step % 100 == 0:
                acc        = calculate_accuracy(logits, trg_expected, pad_id_en)
                current_lr = optimizer.param_groups[0]["lr"]

                writer.add_scalar("Train/Loss_step",     loss.item(), global_step)
                writer.add_scalar("Train/Accuracy_step", acc,         global_step)
                writer.add_scalar("Train/LearningRate",  current_lr,  global_step)

                history.append({
                    "epoch":      epoch + 1,
                    "step":       global_step,
                    "train_loss": loss.item(),
                    "train_acc":  acc,
                    "val_loss":   None,
                    "val_acc":    None,
                    "bleu":       None,
                    "lr":         current_lr,
                })
                pd.DataFrame(history).to_csv(log_file_path, index=False)
                train_iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc":  f"{acc:.4f}",
                    "lr":   f"{current_lr:.2e}"
                })

        avg_train_loss = epoch_train_loss / epoch_steps if epoch_steps > 0 else float('nan')
        writer.add_scalar("Train/Loss_epoch", avg_train_loss, epoch + 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        total_val_acc  = 0.0

        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                trg = batch['trg'].to(device)

                trg_input    = trg[:, :-1]
                trg_expected = trg[:, 1:]

                src_mask, tgt_mask = create_masks(src, trg_input, pad_id_ne, pad_id_en)

                encoder_output = model.encode(src, src_mask)
                decoder_output = model.decode(encoder_output, src_mask, trg_input, tgt_mask)
                logits         = model.project(decoder_output)

                v_loss = loss_fn(logits.view(-1, logits.size(-1)), trg_expected.reshape(-1))
                total_val_loss += v_loss.item()
                total_val_acc  += calculate_accuracy(logits, trg_expected, pad_id_en)

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float("inf")
        avg_val_acc  = total_val_acc  / len(val_loader) if len(val_loader) > 0 else 0.0
        current_lr   = optimizer.param_groups[0]["lr"]

        # ── BLEU Score every 5 epochs ─────────────────────────────────────────
        bleu_score = None
        if (epoch + 1) % 5 == 0 and val_df_ref is not None:
            print(f"  Computing BLEU score (epoch {epoch+1})...")
            bleu_score = calculate_bleu(
                model, val_df_ref, ne_tokenizer, en_tokenizer, device,
                max_samples=500
            )
            if bleu_score is not None:
                print(f"  BLEU Score: {bleu_score:.2f}")
                writer.add_scalar("Val/BLEU", bleu_score, epoch + 1)

        # ── TensorBoard: val metrics ──────────────────────────────────────────
        writer.add_scalar("Val/Loss",     avg_val_loss,      epoch + 1)
        writer.add_scalar("Val/Accuracy", avg_val_acc * 100, epoch + 1)
        writer.add_scalars("Loss/Train_vs_Val", {
            "Train": avg_train_loss,
            "Val":   avg_val_loss,
        }, epoch + 1)

        # ── Dynamic HP step ───────────────────────────────────────────────────
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if hp_controller is not None:
            hp_state = hp_controller.step(epoch + 1, avg_train_loss, avg_val_loss)

            if hp_state["actions"]:
                print(f"\n  [DynamicHP] Gap={hp_state['gap']:.4f} ({hp_state['severity']})")
                for action in hp_state["actions"]:
                    print(f"    -> {action}")
                if any("RESTART" in a for a in hp_state["actions"]):
                    print(f"\n  *** RESTART ADVISED ***")
                    print(f"  Severe overfitting detected — increase dropout and retrain from epoch 0\n")

        history.append({
            "epoch":      epoch + 1,
            "step":       global_step,
            "train_loss": None,
            "train_acc":  None,
            "val_loss":   avg_val_loss,
            "val_acc":    avg_val_acc,
            "bleu":       bleu_score,
            "lr":         None,
        })
        pd.DataFrame(history).to_csv(log_file_path, index=False)

        current_wd = optimizer.param_groups[0]['weight_decay']
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | "
              f"Val Loss={avg_val_loss:.4f} | Val Acc={avg_val_acc*100:.2f}% | "
              f"LR={current_lr:.2e} | WD={current_wd:.4f}")

        # ── Build checkpoint dict (shared for all saves) ──────────────────────
        checkpoint_dict = {
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss":        best_val_loss,
            "patience_counter":     epochs_without_improvement,
            "history":              history,
            "tb_logdir":            tb_logdir,
            "current_wd":           optimizer.param_groups[0]['weight_decay'],
        }
        if scheduler is not None:
            checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

        # ── Save last checkpoint (overwritten every epoch, for resuming) ──────
        torch.save(checkpoint_dict, checkpoint_full_path)

        # ── Save per-epoch checkpoint (never overwritten) ─────────────────────
        epoch_ckpt_path = os.path.join(epoch_ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint_dict, epoch_ckpt_path)
        print(f"  Checkpoint saved: {epoch_ckpt_path}")

        # ── Save model weights only (lightweight, for quick loading) ──────────
        epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"  Model saved:      {epoch_model_path}")

        # ── Track best model ──────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            print(f"  --> NEW BEST! ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_full_path)
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement. Patience: {epochs_without_improvement}/{patience}")

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            print(f"Best val loss: {best_val_loss:.4f}")
            if hp_controller is not None:
                print(f"\nFinal hyperparameter state:")
                print(hp_controller.summary())
            break

    writer.close()
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    ne_tokenizer = load_nepali_tokenizer()
    en_tokenizer = load_english_tokenizer()

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print("\nFiltering long sentences:")
    train_df = filter_long_sentences(train_df, ne_tokenizer, en_tokenizer)
    val_df   = filter_long_sentences(val_df,   ne_tokenizer, en_tokenizer)
    test_df  = filter_long_sentences(test_df,  ne_tokenizer, en_tokenizer)

    train_dataset = NepaliEnglishDataset(train_df, ne_tokenizer, en_tokenizer, max_len=128)
    val_dataset   = NepaliEnglishDataset(val_df,   ne_tokenizer, en_tokenizer, max_len=128)
    test_dataset  = NepaliEnglishDataset(test_df,  ne_tokenizer, en_tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=6, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False,
                              num_workers=6, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False,
                              num_workers=6, pin_memory=True)

    print(f"\nData loaded: {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_transformer(
        src_vocab_size=ne_tokenizer.get_vocab_size(),
        tgt_vocab_size=en_tokenizer.get_vocab_size(),
        src_seq_len=128,
        tgt_seq_len=128,
        d_model=512,
        N=4,
        h=8,
        dropout=0.3,
        d_ff=2048
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        eps=1e-9,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    pad_id_ne = ne_tokenizer.token_to_id("[PAD]")
    pad_id_en = en_tokenizer.token_to_id("[PAD]")

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id_en, label_smoothing=0.1).to(device)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=50,
        patience=7,
        save_dir=os.path.join(PROJECT_ROOT, "models"),
        model_name="nepali_to_eng.pth",
        checkpoint_path="last_checkpoint.pth",
        scheduler=scheduler,
        device=device,
        pad_id_ne=pad_id_ne,
        pad_id_en=pad_id_en,
        ne_tokenizer=ne_tokenizer,
        en_tokenizer=en_tokenizer,
        val_df_ref=val_df,
        dynamic_hp=True,
        wd_init=0.01,
        wd_max=0.1,
    )