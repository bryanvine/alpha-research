#!/usr/bin/env bash
# Regenerate the public live-NAV page and publish it to the alpha-live Pages repo
# (alpha.vineai.tech). Commits/pushes only when content changed. Run by the daily cron
# after 50_live_book.py. Safe to run anytime.
set -euo pipefail
cd /apps/alpha-research
./.venv/bin/python scripts/51_live_page.py /apps/alpha-live
cd /apps/alpha-live
git add -A
if git diff --cached --quiet; then
  echo "live page: no change"
else
  git commit -q -m "live: NAV update $(date -u +%Y-%m-%dT%H:%MZ)"
  git push -q origin main
  echo "live page: published"
fi
