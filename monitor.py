"""Overnight training monitor v2 — clean and robust."""
import json
import os
import re
import time

LOG_FILE = "outputs/training_log.txt"
METRICS_FILE = "stitch_frontend/metrics.json"

result_pattern = re.compile(
    r"Train Loss: ([\d.]+) \| Accuracy: ([\d.]+) \s*\|\|\s* Val Loss: ([\d.]+) \| Accuracy: ([\d.]+)"
)
epoch_pattern = re.compile(r"Epoch \d+/(\d+)")

last_epoch_count = -1

print("🌙 Overnight monitor started — checking every 30s...")
print(f"   Log : {LOG_FILE}")
print(f"   JSON: {METRICS_FILE}")

while True:
    try:
        if not os.path.exists(LOG_FILE):
            time.sleep(30)
            continue

        with open(LOG_FILE, "r") as f:
            content = f.read()

        results = result_pattern.findall(content)
        epoch_matches = epoch_pattern.findall(content)

        current_ep = len(results)
        total_epochs = int(epoch_matches[0]) if epoch_matches else 50

        if results:
            train_acc  = [round(float(r[1]) * 100, 2) for r in results]
            val_acc    = [round(float(r[3]) * 100, 2) for r in results]
            train_loss = [round(float(r[0]), 4) for r in results]
            val_loss   = [round(float(r[2]), 4) for r in results]
            best_val   = max(val_acc)

            done = ("Early stopping triggered." in content) or (current_ep >= total_epochs)
            status = "completed" if done else "training"

            metrics = {
                "status": status,
                "phase": "Phase 2 - Fine-tuning",
                "current_epoch": current_ep,
                "total_epochs": total_epochs,
                "best_val_acc": best_val,
                "target_acc": 98.0,
                "history": {
                    "epoch": current_ep,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            }

            os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f, indent=2)

            if current_ep != last_epoch_count:
                last_epoch_count = current_ep
                gap = 98.0 - best_val
                print(f"  ✅ Epoch {current_ep:>2}/{total_epochs} | "
                      f"Val: {val_acc[-1]:.2f}% | Best: {best_val:.2f}% | "
                      f"Gap to 98%: {gap:.2f}%  [{status}]")

                if done:
                    print(f"\n🎉 Training COMPLETE! Final best accuracy: {best_val:.2f}%")
                    break
        else:
            print(f"  ⏳ Waiting for first epoch to complete... (log size: {len(content)} chars)")

    except Exception as e:
        print(f"  ⚠️  Monitor error: {e}")

    time.sleep(30)
