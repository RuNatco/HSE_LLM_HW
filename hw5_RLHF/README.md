# ДЗ5. RL

`Qwen/Qwen3.5-2B` · `Team-ACE/ToolACE` · seed 42 · [Конфиги](config.json)

[Отчёт](DZ5.pdf) · [Метрики и predictions BFCL](results/)

## Воспроизведение

Из папки проекта:

```bash
docker build -t hw5-rl .
docker run --rm -it --gpus 1 --ipc=host \
  -v "$PWD":/workspace hw5-rl
```

В контейнере:

```bash
set -e
python3 evaluate.py Qwen/Qwen3.5-2B --output results/base
python3 prepare_data.py

for run in rl1_grpo_strict rl2_grpo_shaped rl3_drgrpo_shaped rl2_grpo_shaped_90; do
  python3 train.py "$run"
  python3 export_model.py --run "runs/$run" --output "checkpoints/$run"
  python3 evaluate.py "./checkpoints/$run" --output "results/$run"
done
```
