"""refinr -- three steps, in order:

    refinr look  ./docs     measure   free and instant; facts about the corpus
    refinr guess ./docs     guess     the model proposes what the data is and is for
    refinr tag   ./docs     confirm   you accept or reject each guess

Then `look` again: it is now domain-aware, because confirmed tags feed back in.

Nothing the model guesses counts until a human has touched it.
"""

import argparse
import sys
import time

from . import config as config_module
from . import guess as guess_module
from . import report, signals, tags as tags_module
from .corpus import Corpus
from .store import Store


# --- shared -------------------------------------------------------------
def _load(folder, tier=None, **overrides):
    cfg, cfg_path = config_module.load(folder)
    cfg = cfg.merged(max_tier=tier, **overrides)
    if cfg_path:
        print(f"config: {cfg_path}")
    return cfg


def _models_ready():
    from .models import health_check
    ok, message = health_check()
    if not ok:
        print(f"models unreachable: {message}", file=sys.stderr)
        print("is ollama running?  try: ollama serve", file=sys.stderr)
        return False
    print(f"models ok: {message}")
    return True


def _corpus(folder, cfg, with_store=True):
    store = None
    if with_store and cfg.max_tier >= 1:
        from .models import EMBED_MODEL
        store = Store(folder, EMBED_MODEL)
    corpus = Corpus(folder, cfg, store=store)
    if not corpus.paragraphs:
        print("no readable documents found", file=sys.stderr)
        return None, None
    return corpus, store


# --- step 1: look -------------------------------------------------------
def cmd_look(args):
    cfg = _load(args.folder, tier=args.tier, near_duplicate=args.near,
                target_words=args.chunk_words,
                enabled=[s.strip() for s in args.signals.split(",")] if args.signals else None)

    selected = signals.select(cfg)
    if not selected:
        print("no signals selected", file=sys.stderr)
        return 1
    if any(s.TIER >= 1 for s in selected) and not _models_ready():
        return 1

    started = time.perf_counter()
    corpus, store = _corpus(args.folder, cfg)
    if corpus is None:
        return 1

    findings = signals.run_all(corpus, cfg)
    elapsed = time.perf_counter() - started

    if store is not None:
        store.save()
        store.save_corpus(corpus.paragraphs)

    print()
    print(report.render(corpus, findings, selected, elapsed=elapsed,
                        max_per_signal=10**6 if args.all else 4))

    confirmed = tags_module.TagSet(args.folder).confirmed()
    if confirmed:
        # Honesty note: tags exist but no signal consumes them yet. Say what is
        # true, not what the roadmap wishes were true.
        print(f"\n{len(confirmed)} confirmed tag(s) on file -- these will steer "
              f"LLM-based signals when they land; structural signals ignore them")
    else:
        print("\nnext: `refinr guess` -- have the model work out what this data is")
    return 0


# --- step 2: guess ------------------------------------------------------
def cmd_guess(args):
    if not _models_ready():
        return 1
    cfg = _load(args.folder, tier=0, target_words=args.chunk_words)
    corpus, _ = _corpus(args.folder, cfg, with_store=False)
    if corpus is None:
        return 1

    tagset = tags_module.TagSet(args.folder)
    print(f"reading a sample of {corpus}...")
    proposals, sampled = guess_module.run(corpus, context=tagset.context())
    if not proposals:
        print("the model did not return anything usable. try again, or a larger sample.",
              file=sys.stderr)
        return 1

    tagset.replace_proposals(
        proposals, groups=(tags_module.DATA_TYPE, tags_module.USE_CASE)).save()

    print(f"\nsampled {len(sampled)} chunks from {len(corpus.sources())} files\n")
    for group in (tags_module.DATA_TYPE, tags_module.USE_CASE):
        print(f"{tags_module.TITLES[group]}")
        for tag in tagset.group(group):
            print(f"  {tag.line()}")
            if tag.reasoning:
                print(f"      {tag.reasoning}")
        print()

    print("These are guesses, not findings -- nothing here affects the report yet.")
    print("next: `refinr tag` to confirm or reject each one")
    return 0


# --- step 3: tag --------------------------------------------------------
def cmd_tag(args):
    tagset = tags_module.TagSet(args.folder)
    if not len(tagset):
        print("no tags yet -- run `refinr guess` first")
        return 1

    if args.list:
        for group in tags_module.GROUPS:
            entries = tagset.group(group)
            if not entries:
                continue
            print(f"\n{group}")
            for tag in entries:
                print(f"  {tag.line()}")
        return 0

    pending = [t for t in tagset.tags if t.status == tags_module.PROPOSED]
    if not pending:
        print("every guess has been decided. `refinr tag --list` to review.")
        return 0

    print(f"{len(pending)} guess(es) to decide.  [y] yes  [n] no  [e] edit  [s] skip  [q] quit\n")
    for tag in pending:
        print(f"  {tag.group}: {tag.label}")
        if tag.reasoning:
            print(f"     {tag.reasoning}")
        try:
            answer = input("  ? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer.startswith("q"):
            break
        if answer.startswith("y"):
            tag.status = tags_module.CONFIRMED
        elif answer.startswith("n"):
            tag.status = tags_module.REJECTED
        elif answer.startswith("e"):
            edited = input("  corrected label: ").strip()
            if edited:
                tag.label, tag.status, tag.source = edited, tags_module.CONFIRMED, "user"
        print()

    tagset.save()
    confirmed = tagset.confirmed()
    print(f"{len(confirmed)} tag(s) confirmed -> {tagset.path}")
    if confirmed:
        print("\nnext: `refinr look` -- the report is now domain-aware")
    return 0


# --- inventory: what's IN it ---------------------------------------------
def cmd_inventory(args):
    from . import inventory as inv

    cfg = _load(args.folder, max_topics=args.topics)
    if not _models_ready():
        return 1
    corpus, store = _corpus(args.folder, cfg)
    if corpus is None:
        return 1

    print(f"grouping paragraphs by meaning ({corpus})...")
    clusters, unclustered = inv.cluster(corpus, cfg)
    if store is not None:
        store.save()
        store.save_corpus(corpus.paragraphs)
    print(f"  {len(clusters)} group(s), {len(unclustered)} paragraph(s) unclustered, "
          f"{corpus.embed_calls} new embedding(s)")

    if clusters and not args.no_labels:
        print(f"asking {len(clusters)} name(s) from the model (~30-60s each)...")
        inv.label(clusters, corpus, cfg, on_progress=print)
        proposals = inv.to_tags(clusters)
        if proposals:
            tags_module.TagSet(args.folder).replace_proposals(
                proposals, groups=(tags_module.TOPIC,)).save()

    data = inv.snapshot(corpus, clusters, unclustered, cfg)
    path = inv.save(args.folder, data)

    print()
    print(inv.render(data))
    print(f"\nsaved -> {path}")
    if clusters and not args.no_labels:
        print("next: `refinr tag` to confirm the topic names, "
              "then `refinr passport` for the verdict")
    return 0


# --- passport: the verdict ------------------------------------------------
def cmd_passport(args):
    from . import passport as passport_module

    cfg = _load(args.folder)
    page = passport_module.run(args.folder, cfg)
    print(page)
    return 0


# --- bench: grade the signals ---------------------------------------------
def cmd_bench(args):
    from . import bench
    if (args.tier is None or args.tier >= 1) and not _models_ready():
        return 1
    return bench.main(args.folder, case=args.case, tier=args.tier,
                      sweep=args.sweep)


# --- layer 1: web demo --------------------------------------------------
def cmd_web(args):
    if not _models_ready():
        return 1
    from .webapp import serve
    serve(port=args.port, open_browser=not args.no_browser)
    return 0


# --- wiring -------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="refinr",
        description="measure -> guess -> confirm.  Find what is not earning its place.",
    )
    sub = parser.add_subparsers(dest="command")

    look = sub.add_parser("look", help="step 1: measure the corpus (free, instant)")
    look.add_argument("folder")
    look.add_argument("--tier", type=int, default=None, choices=[0, 1, 2])
    look.add_argument("--signals", default=None)
    look.add_argument("--near", type=float, default=None)
    look.add_argument("--chunk-words", type=int, default=None)
    look.add_argument("--all", action="store_true", help="list every finding")
    look.set_defaults(func=cmd_look)

    guess = sub.add_parser("guess", help="step 2: model proposes what this data is")
    guess.add_argument("folder")
    guess.add_argument("--chunk-words", type=int, default=None)
    guess.set_defaults(func=cmd_guess)

    tag = sub.add_parser("tag", help="step 3: confirm or reject the guesses")
    tag.add_argument("folder")
    tag.add_argument("--list", action="store_true", help="show tags without prompting")
    tag.set_defaults(func=cmd_tag)

    inventory = sub.add_parser("inventory", help="what's IN it: measured topic coverage")
    inventory.add_argument("folder")
    inventory.add_argument("--topics", type=int, default=None,
                           help="cap on topic groups (default from config)")
    inventory.add_argument("--no-labels", action="store_true",
                           help="skip the model; measured groups only")
    inventory.set_defaults(func=cmd_inventory)

    passport = sub.add_parser("passport", help="one-page verdict vs your [profile]")
    passport.add_argument("folder")
    passport.set_defaults(func=cmd_passport)

    bench = sub.add_parser("bench", help="grade the signals against planted defects")
    bench.add_argument("folder", nargs="?", default="testdata",
                       help="bench root; each subdir with a manifest.json is a case")
    bench.add_argument("--case", default=None, help="run one case by directory name")
    bench.add_argument("--tier", type=int, default=None, choices=[0, 1, 2])
    bench.add_argument("--sweep", default=None,
                       metavar="FIELD=START:STOP[:STEP]|FIELD=V1,V2,...",
                       help="re-run per value, print precision/recall per row")
    bench.set_defaults(func=cmd_bench)

    web = sub.add_parser("web", help="layer 1: local demo web app (small files, plain language)")
    web.add_argument("--port", type=int, default=7358)
    web.add_argument("--no-browser", action="store_true")
    web.set_defaults(func=cmd_web)

    show = sub.add_parser("signals", help="list available signals")
    show.set_defaults(func=lambda a: [
        print(f"  tier {s.TIER}  {s.NAME:<14} detects: {', '.join(s.DETECTS)}")
        for s in signals.registry()] and 0 or 0)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
