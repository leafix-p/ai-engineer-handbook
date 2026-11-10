#!/usr/bin/env bash
# 监听本目录变化，自动提交并推送。
# 依赖：inotify-tools（Linux）或 fswatch（macOS）。
# 用法：chmod +x sync.sh && ./sync.sh
# 注意：会自动 git add -A，请确保 .gitignore 已排除密钥、模型权重、数据。
set -euo pipefail
cd "$(dirname "$0")"

push_if_changed() {
  sleep 3
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "docs: auto sync $(date '+%F %T')"
    git push
    echo "[sync] $(date '+%F %T') pushed"
  fi
}

if command -v inotifywait >/dev/null 2>&1; then
  while true; do
    inotifywait -r -e modify,create,delete,move --exclude '\.git' . >/dev/null
    push_if_changed
  done
elif command -v fswatch >/dev/null 2>&1; then
  fswatch -o . | while read -r _; do
    push_if_changed
  done
else
  echo "请先安装 inotify-tools（Linux）或 fswatch（macOS）" >&2
  exit 1
fi
