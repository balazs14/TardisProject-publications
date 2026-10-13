#!/usr/bin/env bash
# Pulls the coworker's edited liquidity-premium.tex out of the writer repo
# (see export_writer_repo.sh, default ~/static_export_no_generate) and copies
# it back into this project, ignoring everything else in that repo (frozen
# artifacts/bib/style files, etc).
#
# Usage:
#   ./import_writer_edits.sh <writer_repo_dir>
#
# Fetches/pulls in the writer repo first if it has a remote, then copies
# liquidity-premium.tex back here. Review with `git diff` before committing.

set -euo pipefail

PAPER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WRITER_DIR=${1:?"usage: import_writer_edits.sh <writer_repo_dir>"}

if git -C "$WRITER_DIR" remote get-url origin >/dev/null 2>&1; then
	git -C "$WRITER_DIR" pull --ff-only
fi

cp "$WRITER_DIR/liquidity-premium.tex" "$PAPER_DIR/liquidity-premium.tex"

echo "Copied $WRITER_DIR/liquidity-premium.tex into $PAPER_DIR."
echo "Review with: git -C \"$PAPER_DIR\" diff -- liquidity-premium.tex"
