#!/usr/bin/env bash
# 현재 CADDreamer-jjy 워킹 디렉터리 상태(.gitignore 반영)를 스냅샷으로 떠서
# gpu-workbench 저장소의 지정 브랜치 아래 CADDreamer-jjy/ 폴더로 그대로 얹어 push한다.
#
# - 로컬 작업 폴더/브랜치/원격(origin)은 전혀 건드리지 않는다 (checkout 없음).
# - 대상 브랜치의 다른 폴더(brepgen_abc, NurbGen-Project 등)는 절대 건드리지 않는다.
# - 매번 대상 브랜치의 최신 tip 위에 fast-forward로 커밋 1개를 새로 얹는다 (force push 없음).
#
# 사용법:
#   ./sync_to_workbench.sh                 # 확인 후 push
#   ./sync_to_workbench.sh --dry-run       # push 안 하고 검증만
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f freecad_stub.py ]]; then
    echo "!! CADDreamer-jjy 저장소 루트에서 실행해야 합니다." >&2
    exit 1
fi

REMOTE_URL="git@github-personal:Capstone-Project-Unnamed/gpu-workbench.git"
BRANCH="zoo/caddreamer-test"
SUBDIR="CADDreamer-jjy"
AUTHOR_NAME="zoo"
AUTHOR_EMAIL="opff5423@gmail.com"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

echo "=== 1) 원격 브랜치 최신 tip 가져오기 ($BRANCH) ==="
TMP_REF="refs/remotes/_sync/$(echo "$BRANCH" | tr '/' '-')"
git fetch "$REMOTE_URL" "$BRANCH:$TMP_REF"
PARENT=$(git rev-parse "$TMP_REF")
echo "PARENT=$PARENT"

echo ""
echo "=== 2) 현재 워킹 디렉터리 스냅샷 (.gitignore 반영) ==="
TMP_INDEX=$(mktemp -u /tmp/gitidx-sync.XXXXXX)
GIT_INDEX_FILE="$TMP_INDEX" git add -A
CONTENT_TREE=$(GIT_INDEX_FILE="$TMP_INDEX" git write-tree)
rm -f "$TMP_INDEX"
echo "CONTENT_TREE=$CONTENT_TREE"

echo ""
echo "=== 3) 대상 브랜치 트리 위에 $SUBDIR/ 로 얹기 ==="
TMP_INDEX2=$(mktemp -u /tmp/gitidx-sync2.XXXXXX)
GIT_INDEX_FILE="$TMP_INDEX2" git read-tree "$PARENT"
GIT_INDEX_FILE="$TMP_INDEX2" git rm -r --cached --ignore-unmatch -q -- "$SUBDIR" || true
GIT_INDEX_FILE="$TMP_INDEX2" git read-tree --prefix="$SUBDIR/" "$CONTENT_TREE"
FINAL_TREE=$(GIT_INDEX_FILE="$TMP_INDEX2" git write-tree)
rm -f "$TMP_INDEX2"
echo "FINAL_TREE=$FINAL_TREE"

echo ""
echo "=== 4) 변경 요약 (기존 브랜치 tip 대비) ==="
git diff --stat "$PARENT" "$FINAL_TREE" -- . ":!$SUBDIR" || true
echo "-- (위가 비어있어야 정상: $SUBDIR/ 밖은 안 건드림) --"
git diff --stat "$PARENT" "$FINAL_TREE" -- "$SUBDIR" | tail -5

TS=$(date +%Y-%m-%d\ %H:%M)
NEW_COMMIT=$(GIT_AUTHOR_NAME="$AUTHOR_NAME" GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
    GIT_COMMITTER_NAME="$AUTHOR_NAME" GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
    git commit-tree "$FINAL_TREE" -p "$PARENT" -m "Sync CADDreamer-jjy snapshot ($TS)")
echo ""
echo "NEW_COMMIT=$NEW_COMMIT"

echo ""
echo "=== 5) push ${DRY_RUN:+(dry-run)} ==="
if $DRY_RUN; then
    git push --dry-run "$REMOTE_URL" "$NEW_COMMIT:refs/heads/$BRANCH"
    echo "-- dry-run 종료, 실제로 안 올렸습니다 --"
else
    git push "$REMOTE_URL" "$NEW_COMMIT:refs/heads/$BRANCH"
    echo "-- push 완료: $BRANCH -> $NEW_COMMIT --"
fi

git update-ref -d "$TMP_REF"
