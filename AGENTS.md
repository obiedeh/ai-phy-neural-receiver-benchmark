# Repository instructions

## Repository hygiene

Read and follow the [Repository hygiene policy in obiedeh-brand/AGENTS.md](https://github.com/obiedeh/obiedeh-brand/blob/main/AGENTS.md#repository-hygiene-standing-every-repo), the single source of these rules.

- Name boards only: `jetson-agx-thor`, `jetson-orin-nx`, `rtx-5090-workstation`.
- No host names, home-directory paths or LAN addresses in tracked prose, logs, commit messages or branch names; write commands against `~`.
- Never bypass or disable hooks, or change `core.hooksPath` to evade them. Check diffs and commit messages against `~/.config/git/private-terms.txt` without exposing its contents.
- Never rewrite history, force push or move tags. Amend only commits that have never been pushed.
- Keep handoff notes, reviews and plans outside repositories in `~/handoff_notes/`.
