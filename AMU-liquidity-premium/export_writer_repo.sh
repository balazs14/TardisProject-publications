#!/usr/bin/env bash
# Exports a flattened, self-contained snapshot of liquidity-premium.tex + its
# frozen figures/tables/bib into a standalone git repo a coworker can clone
# and push to. Does NOT touch the main TardisProject or publications repos.
#
# Usage:
#   ./export_writer_repo.sh [target_dir]
#
# target_dir defaults to ~/static_export_no_generate (outside any existing repo).
# Re-run any time after a real rebuild to refresh the frozen artifacts snapshot;
# it commits on top of the existing writer repo history.
#
# Only copies the specific artifact files that liquidity-premium.tex actually
# references (figures/tables), NOT the whole artifacts/ dir -- that dir also
# holds multi-GB cache parquet files and a tmp/ scratch dir that must never be
# copied.

set -euo pipefail

PAPER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PUB_DIR=$(cd "$PAPER_DIR/.." && pwd)
TARGET_DIR=${1:-"$HOME/my-container/TardisProject-AMU-paper-no-generate"}

mkdir -p "$TARGET_DIR"

if [[ ! -d "$TARGET_DIR/.git" ]]; then
	git -C "$TARGET_DIR" init -q
	echo "Initialized new git repo at $TARGET_DIR"
fi

# Editable source of truth.
cp "$PAPER_DIR/liquidity-premium.tex" "$TARGET_DIR/"

# Frozen, read-only-by-convention references: bib + shared style files.
cp "$PAPER_DIR/bibl.bib" "$TARGET_DIR/"
cp "$PUB_DIR/agsm.bst" "$PUB_DIR/newtxmath.sty" "$PUB_DIR/newtxtext.sty" "$PUB_DIR/tablestyles.sty" "$TARGET_DIR/"

# Frozen figures/tables snapshot: only the specific files
# liquidity-premium.tex references, dereferenced from the artifacts symlink.
# NEVER `cp -R` the whole artifacts/ dir -- it also holds multi-GB cache
# parquet files and a tmp/ scratch dir.
ARTIFACT_FILES=$(grep -oE 'artifacts/[A-Za-z0-9_./-]+\.(png|pdf|tex)' "$PAPER_DIR/liquidity-premium.tex" | sed 's#^artifacts/##' | sort -u)

mkdir -p "$TARGET_DIR/artifacts"
while IFS= read -r rel; do
	[[ -z "$rel" ]] && continue
	src="$PAPER_DIR/artifacts/$rel"
	if [[ -f "$src" ]]; then
		cp -L "$src" "$TARGET_DIR/artifacts/$rel"
	else
		echo "warning: referenced artifact missing, skipping: $rel" >&2
	fi
done <<< "$ARTIFACT_FILES"

cat > "$TARGET_DIR/README-WRITER.md" <<'EOF'
# Writer copy of liquidity-premium.tex

This is a frozen, standalone snapshot. It contains only what's needed to
compile and edit the prose:

- `liquidity-premium.tex` — **edit this.** Text/structure changes only.
- `bibl.bib`, `agsm.bst`, `*.sty` — required to compile; do not need to be
  edited to add citations to existing sources already in `bibl.bib`.
- `artifacts/` — a frozen snapshot of the figures and tables. Do not
  regenerate or edit these; they'll go stale relative to the rest of the
  project. If a figure/table needs to change, ask the maintainer for a
  refreshed export.

## Compiling

```
latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error -outdir=build liquidity-premium.tex
```

## Sending changes back

Push to this repo's remote (or send a diff/patch of `liquidity-premium.tex`).
Only that file will be merged back into the main project; everything else
here is a disposable snapshot.
EOF

git -C "$TARGET_DIR" add -A
if ! git -C "$TARGET_DIR" diff --cached --quiet; then
	git -C "$TARGET_DIR" commit -q -m "Refresh writer export snapshot"
	echo "Committed refreshed snapshot in $TARGET_DIR"
else
	echo "No changes to commit; snapshot already up to date."
fi

echo
echo "Next steps:"
echo "  1. Create a remote (e.g. a private GitHub repo) for $TARGET_DIR."
echo "  2. git -C \"$TARGET_DIR\" remote add origin <url> && git -C \"$TARGET_DIR\" push -u origin main"
echo "  3. Give the coworker clone/push access to that remote."
