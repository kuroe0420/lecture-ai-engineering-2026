# セクション7の実行後、追加のコードセルで実行する。
# ランタイムを再起動する前なら、学習の再実行なしで記録できる。
import json
import shutil
from pathlib import Path

if "trainer" not in globals():
    raise RuntimeError("学習済みのtrainerがありません。学習を実行したランタイムで実行してください。")

history = trainer.state.log_history
loss_rows = [row for row in history if "loss" in row]
summary_rows = [row for row in history if "train_runtime" in row]
if not summary_rows:
    raise RuntimeError("学習完了時の計測値がありません。学習の完了を確認してください。")

last_loss = loss_rows[-1] if loss_rows else {}
summary = summary_rows[-1]
training_metrics = {
    "runtime_seconds": summary["train_runtime"],
    "last_logged_loss": last_loss.get("loss"),
    "last_loss_step": last_loss.get("step"),
    "mean_training_loss": summary.get("train_loss"),
    "global_step": trainer.state.global_step,
    "completed_epochs": trainer.state.epoch,
}
run_config = {key: value for key, value in CONFIG.items()
              if key not in ("work_dir", "drive_dir")}
folder = Path(CONFIG["work_dir"])
folder.mkdir(parents=True, exist_ok=True)

saved = []
for name, data in {
    "training_metrics.json": {"config": run_config, "training": training_metrics},
    "training_history.json": history,
}.items():
    path = folder / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    saved.append(path)

# セクション7を実行済みの場合は、設定が一致する結果にだけ追記する。
result_path = folder / "result.json"
if result_path.exists():
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    if result_data.get("config") == run_config:
        result_data["training"] = training_metrics
        result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(result_path)

if globals().get("USE_DRIVE", False):
    drive_folder = Path(CONFIG["drive_dir"])
    drive_folder.mkdir(parents=True, exist_ok=True)
    for path in saved:
        shutil.copy2(path, drive_folder / path.name)
    print("Drive保存先:", drive_folder)

print(json.dumps(training_metrics, ensure_ascii=False, indent=2))
print("保存先:", folder)

# Driveへの保存を終えてから、PCへのダウンロードを開始する。
from google.colab import files

for name in [
    "result.json",
    "eval_before.json",
    "eval_after.json",
    "training_metrics.json",
    "training_history.json",
]:
    path = folder / name
    if path.exists():
        print(f"ダウンロード：{name}")
        files.download(str(path))
    else:
        print(f"見つかりません：{path}")
