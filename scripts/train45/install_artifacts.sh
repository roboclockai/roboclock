#!/usr/bin/env bash
# Put the committed train45 artifacts where the role scripts read them (docs/five-domain-training.md, 3.4).
#
#   bash scripts/train45/install_artifacts.sh            # prompts + every role
#   bash scripts/train45/install_artifacts.sh A          # prompts + role A only (also B, C, D; several allowed)
#   CHECK=1 bash scripts/train45/install_artifacts.sh    # only report what is missing / identical / different
#   FORCE=1 bash scripts/train45/install_artifacts.sh C  # overwrite targets that differ
#
# artifacts/MANIFEST.tsv lists role, sha256, repo path and install target of each file. The prompts are
# always installed (every role builds its prompt from them). A target that already exists with other
# content is left alone unless FORCE=1. Every installed file is checked against its sha256 afterwards.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${HERE}/artifacts/MANIFEST.tsv"
ROLES=("$@")
(( ${#ROLES[@]} )) || ROLES=(A B C D)
want_role() {
    [[ "$1" == prompts ]] && return 0
    local r
    for r in "${ROLES[@]}"; do [[ "$r" == "$1" ]] && return 0; done
    return 1
}

fail=0
while IFS=$'\t' read -r role sha src dst; do
    [[ "$role" == \#* || -z "$role" ]] && continue
    want_role "$role" || continue
    repo="${HERE}/${src}"
    have_repo=$(sha256sum "$repo" | cut -d' ' -f1)
    if [[ "$have_repo" != "$sha" ]]; then
        echo "BAD   ${src}: repo copy does not match MANIFEST.tsv" >&2
        fail=1
        continue
    fi
    if [[ -e "$dst" ]]; then
        have=$(sha256sum "$dst" | cut -d' ' -f1)
        if [[ "$have" == "$sha" ]]; then
            echo "same  ${dst}"
            continue
        fi
        if [[ -z "${FORCE:-}" || -n "${CHECK:-}" ]]; then
            echo "DIFF  ${dst} (left alone; FORCE=1 overwrites)"
            fail=1
            continue
        fi
    elif [[ -n "${CHECK:-}" ]]; then
        echo "miss  ${dst}"
        continue
    fi
    install -D -m "$( [[ "$dst" == *.sh ]] && echo 755 || echo 644 )" "$repo" "$dst"
    [[ "$(sha256sum "$dst" | cut -d' ' -f1)" == "$sha" ]] || { echo "BAD   ${dst} after install" >&2; fail=1; continue; }
    echo "put   ${dst}"
done < "$MANIFEST"

if (( fail )); then
    echo "install_artifacts: some files were not installed (see above)" >&2
    exit 1
fi
echo "install_artifacts: OK (roles: ${ROLES[*]} + prompts)"
