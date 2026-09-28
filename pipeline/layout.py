"""Where everything lives: the one mapping from (stage, model, stream, condition) to paths, shared by run and table.

    data/<dataset>/                                               registered datasets (configs/datasets/): released
                                                                  streams, peers' generated answers (<dataset>/<peer>/),
                                                                  misleading streams (+ manifest.json)
    outputs/features/<model>/<stream>/shard<k>.pt                 the judge's features (pipeline.features)
    outputs/record/<model>/<stream>/<order>.fit-<fit>.jsonl       the record along the stream (+ .quality.json)
    outputs/eval/<model or run>/<stream>/<condition>/             an evaluation (pipeline.evaluate)
    outputs/tables/<experiment>.md, outputs/runs/<experiment>/    the result table, the resolved config and commands
    logs/<experiment>/<job>.log

An experiment with `own_answer: true` adds the central model's own no-consultation answer to every stream it reads
(data/<dataset>+<model>/): its features, record and every evaluation but No consultation live under <dataset>+own, so
they never mix with the peers-only results; No consultation stays shared with the base stream.

A smoke run (--smoke) uses outputs/smoke/ and logs/smoke/ for everything it writes, built datasets included, so it can
never be mistaken for, or skip, a real run. An experiment that only evaluates released datasets sets
`smoke: {use_released_data: true}`: its smoke run reads them from paths.data and never builds any.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.config import REPO
from pipeline.registry import load_registry


class Layout:
    def __init__(self, cfg: dict, smoke: bool = False):
        self.cfg, self.smoke = cfg, smoke
        paths = cfg.get("paths", {})
        self.repo = REPO
        self.data = self._abs(paths.get("data", "data"))
        outputs = self._abs(paths.get("outputs", "outputs"))
        logs = self._abs(paths.get("logs", "logs"))
        self.outputs = outputs / "smoke" if smoke else outputs
        self.logs = logs / "smoke" if smoke else logs
        # built datasets: a smoke run builds its own under outputs/smoke/data, unless the experiment only reads released ones
        self.derived = self.outputs / "data" if smoke and not cfg.get("smoke", {}).get("use_released_data") else self.data
        self.own = bool(cfg.get("own_answer"))
        self.models_root = Path(paths.get("models_root", "models"))
        self.registry = load_registry(paths.get("registry"))

    def _abs(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.repo / p

    # --- registered names -> paths --------------------------------------------------------------------------------
    def model(self, tag: str) -> dict:
        spec = dict(self.registry.model(tag))
        path = Path(spec["path"])
        spec["path"] = path if path.is_absolute() else self.models_root / path
        return spec

    def peer_models(self, peers: list[str]) -> list[dict]:
        """Registered peers in peer_0 ... order, each merged over its model; `name` is the model directory, which names
        the peer's answers."""
        out = []
        for i, tag in enumerate(peers):
            peer = self.registry.peer(tag)
            spec = dict(self.model(peer["model"]), **{k: v for k, v in peer.items() if k not in ("model", "name", "file")})
            out.append(dict(spec, tag=tag, model=peer["model"], index=i, name=spec["path"].name))
        return out

    def stream(self, name: str) -> dict:
        """A registered dataset: {'name', 'kind', 'path', 'peers' (count), 'peer_names', 'base', 'answers', 'regime', ...}.

        A released stream is read from paths.data; what the pipeline builds (answers, misleading streams) goes to
        paths.data too, or to outputs/smoke/data in a smoke run, so a smoke run never writes next to real data.
        """
        d = dict(self.registry.dataset(name))
        base = self.registry.stream_of(name)
        d["peer_names"] = list(base["peers"])
        d["peers"] = len(d["peer_names"])
        if d["kind"] == "stream":
            root = (self.outputs / "data") if (self.smoke and d.get("built")) else Path(self.cfg.get("paths", {}).get("data", "data"))
            d["path"] = self._abs(root / d["path"])
        elif d["kind"] == "answers":
            d["path"] = self.derived / d.get("path", name)
        else:
            d["path"] = self.derived / d.get("path", f"{name}/{Path(base['path']).name}")
        return d

    def own_stream(self, model: str, dataset: str) -> Path:
        """The dataset's stream with the central model's own answer as its last answer (built, so under smoke/ in a smoke run)."""
        return (self.outputs / "data" if self.smoke else self.data) / f"{dataset}+{model}" / "test.jsonl"

    def _own(self, stream: str) -> str:
        return f"{stream}+own" if self.own else stream

    # --- artifacts --------------------------------------------------------------------------------------------------
    def features_dir(self, model: str, stream: str) -> Path:
        return self.outputs / "features" / model / self._own(stream)

    def review_dir(self, model: str, stream: str) -> Path:
        """The lead's checks of a team's reports (pipeline.review)."""
        return self.outputs / "review" / model / self._own(stream)

    def record_file(self, model: str, stream: str) -> Path:
        rec = self.cfg.get("record", {})
        fit = rec.get("fit", "self")
        order = rec.get("order", "shuffled0")
        extra = ""
        if int(rec.get("dim", 256)) != 256 or float(rec.get("lam", 100.0)) != 100.0 or rec.get("design", "qc") != "qc":
            extra = f".{rec.get('design', 'qc')}-d{int(rec.get('dim', 256))}-lam{float(rec.get('lam', 100.0)):g}"
        return self.outputs / "record" / model / self._own(stream) / f"{order}.fit-{fit}{extra}.jsonl"

    @staticmethod
    def addresses_file(record_file: Path) -> Path:
        return record_file.with_name(record_file.name[: -len(".jsonl")] + ".addresses.pt")

    @staticmethod
    def quality_file(record_file: Path) -> Path:
        return record_file.with_name(record_file.name[: -len(".jsonl")] + ".quality.json")

    def eval_dataset(self, dataset: str, condition: dict) -> str:
        """The dataset whose result a condition reads: a no-consultation condition sees no peer answers, so every dataset built
        on a stream (its misleading variants) shares the stream's result."""
        return self.registry.stream_of(dataset)["name"] if condition.get("mode") == "solo" else dataset

    def eval_dir(self, model: str, stream: str, condition: str) -> Path:
        solo = self.cfg.get("conditions", {}).get(condition, {}).get("mode") == "solo"
        return self.outputs / "eval" / model / (stream if solo else self._own(stream)) / condition

    def table_file(self, experiment: str) -> Path:
        return self.outputs / "tables" / f"{experiment}.md"

    def run_dir(self, experiment: str) -> Path:
        return self.outputs / "runs" / experiment

    def log(self, experiment: str, job: str) -> Path:
        return self.logs / experiment / f"{job}.log"

