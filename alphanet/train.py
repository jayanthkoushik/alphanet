import logging
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional

import ignite.contrib.handlers
import ignite.engine
import ignite.handlers
import ignite.metrics
import torch
from corgy import Corgy
from corgy.types import KeyValuePairs, OutputBinFile, OutputDirectory, SubClass
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Annotated, Literal

from alphanet._dataset import NNsResult, SplitLTDataGroupInfo, SplitLTDataset
from alphanet._pt import DEFAULT_DEVICE, PTOpt, TBLogs
from alphanet._samplers import AllFewSampler, ClassBalancedBaseSampler, SamplerBuilder
from alphanet._utils import log_alphas, log_metrics
from alphanet.alphanet import AlphaNet, AlphaNetClassifier

logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


class TrainingConfig(Corgy):
    n_neighbors: Annotated[
        int, "number of nearest neighbors to use (excluding self)"
    ] = 5
    nn_dist: Annotated[
        Literal["euclidean", "cosine", "random"],
        "distance metric for nearest neighbors",
    ] = "euclidean"
    ptopt: Annotated[PTOpt, "optimization parameters"] = PTOpt(
        optim_cls=torch.optim.AdamW,
        optim_params={"lr": 1e-3},
        lr_sched_cls=torch.optim.lr_scheduler.StepLR,
        lr_sched_params={"step_size": 10, "gamma": 0.1},
    )
    train_epochs: Annotated[int, "number of epochs to train for"] = 25
    min_epochs: Annotated[
        int, "minimum number of epochs to use when selecting best model"
    ] = 5
    train_batch_size: Annotated[int, "batch size for training"] = 64
    eval_batch_size: Annotated[
        Optional[int], "batch size for evaluation, overriding training batch size"
    ] = 1024
    sampler_builder: Annotated[
        SamplerBuilder, "arguments to build combined sampler"
    ] = SamplerBuilder(
        sampler_classes=(AllFewSampler, ClassBalancedBaseSampler),
        sampler_args=(KeyValuePairs(""), KeyValuePairs("r=0.1")),
    )
    tb_logs: Annotated[TBLogs, "tensorboard log directory"] = TBLogs()
    train_datagrp: str = "train"
    val_datagrp: Optional[str] = "val"
    test_datagrp: str = "test"
    pred_scale: float = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.val_datagrp:
            self.val_datagrp = None
        if self.train_epochs < self.min_epochs:
            raise ValueError("`train_epochs` should not be less than `min_epochs`")

    def as_dict(self, recursive=False):  # pylint: disable=arguments-differ
        d = super().as_dict(recursive)
        del d["tb_logs"]
        if isinstance(d["ptopt"]["optim_cls"], SubClass):
            d["ptopt"]["optim_cls"] = d["ptopt"]["optim_cls"].which
        d["ptopt"]["optim_params"] = dict(d["ptopt"]["optim_params"])
        if isinstance(d["ptopt"]["lr_sched_cls"], SubClass):
            d["ptopt"]["lr_sched_cls"] = d["ptopt"]["lr_sched_cls"].which
        d["ptopt"]["lr_sched_params"] = dict(d["ptopt"]["lr_sched_params"])
        d["sampler_builder"]["sampler_args"] = [
            dict(_arg) for _arg in d["sampler_builder"]["sampler_args"]
        ]
        return d


class EpochData(Corgy):
    n_train__per__class: Dict[int, int]
    n_train__per__split: Dict[str, int]
    mean_batch_loss__seq: List[float]
    mean_alpha__vec: Tensor
    min_alpha__vec: Tensor
    max_alpha__vec: Tensor
    val_metrics: Optional[Dict[str, Any]]
    val_acc__per__split: Optional[Dict[str, float]]
    val_acc__per__class: Optional[Dict[int, float]] = None
    alphanet_classifier_state_dict: Optional[Dict[str, Any]]


class TrainResult(Corgy):
    train_data_info: SplitLTDataGroupInfo
    val_data_info: Optional[SplitLTDataGroupInfo]
    nn_info: NNsResult
    alphanet: AlphaNet
    alphanet_source__mat__seq: List[Tensor]
    fbclass_ordered_idx__vec: Tensor
    training_config: TrainingConfig
    epoch_data__seq: List[EpochData]
    best_epoch: Optional[int]
    best_alphanet_classifier_state_dict: Dict[str, Any]
    test_metrics: Dict[str, Any]
    test_acc__per__split: Dict[str, float]
    test_acc__per__class: Optional[Dict[int, float]] = None

    def as_dict(self, recursive=False):  # pylint: disable=arguments-differ
        d = super().as_dict(recursive)
        if recursive:
            d["epoch_data__seq"] = [
                _epoch_data.as_dict(recursive) for _epoch_data in self.epoch_data__seq
            ]
        return d

    @classmethod
    def from_dict(cls, d):  # pylint: disable=arguments-differ
        o = super().from_dict(d)
        o.epoch_data__seq = [
            EpochData.from_dict(_epoch_data_dict)
            for _epoch_data_dict in d["epoch_data__seq"]
        ]
        return o

    def load_best_alphanet_classifier(self) -> AlphaNetClassifier:
        self.alphanet.set_sources(self.alphanet_source__mat__seq)
        alphanet_classifier = AlphaNetClassifier(
            self.alphanet,
            self.train_data_info.n_base_classes,
            self.fbclass_ordered_idx__vec,
        )
        alphanet_classifier.load_state_dict(self.best_alphanet_classifier_state_dict)
        return alphanet_classifier

    def state_dict(self):
        return self.as_dict(recursive=True)

    def load_state_dict(self, d):
        o = self.__class__.from_dict(d)
        try:
            self.train_data_info = o.train_data_info
            self.val_data_info = o.val_data_info
            self.nn_info = o.nn_info
            self.alphanet = o.alphanet
            self.alphanet_source__mat__seq = o.alphanet_source__mat__seq
            self.fbclass_ordered_idx__vec = o.fbclass_ordered_idx__vec
            self.training_config = o.training_config
            self.epoch_data__seq = o.epoch_data__seq
            self.best_epoch = o.best_epoch
            self.best_alphanet_classifier_state_dict = (
                o.best_alphanet_classifier_state_dict
            )
            self.test_metrics = o.test_metrics
            self.test_acc__per__class = o.test_acc__per__class
            self.test_acc__per__split = o.test_acc__per__split
        except AttributeError:
            pass


class TrainCmd(Corgy):
    save_file: Annotated[Optional[OutputBinFile], "file to save training results"]
    ckpt_dir: Annotated[
        Optional[OutputDirectory], "directory to save/load checkpoints"
    ] = None
    n_ckpt: Annotated[int, "number of most recent checkpoints to keep"] = 2
    load_from_ckpt: Annotated[
        bool,
        "whether to load the latest checkpoint (ignored if `ckpt_dir` not provided)",
    ] = True
    dataset: Annotated[
        SplitLTDataset, "name of dataset as defined in 'config/datasets.toml'"
    ]
    alphanet: Annotated[AlphaNet, "AlphaNet parameters"]
    training: Annotated[TrainingConfig, "training parameters"]

    def __call__(self) -> TrainResult:
        logging.info("loading train data...")
        orig_train_data = self.dataset.load_data(self.training.train_datagrp)
        logging.info("loading train data...done")
        if (_val_datagrp := self.training.val_datagrp) is not None:
            logging.info("loading val data...")
            orig_val_data = self.dataset.load_data(_val_datagrp)
            logging.info("loading val data...done")
        else:
            logging.warning("no validation data")
            orig_val_data = None
        logging.info("loading test data...")
        orig_test_data = self.dataset.load_data(self.training.test_datagrp)
        logging.info("loading test data...done")
        logging.info("loading classifier...")
        full_clf = self.dataset.load_classifier(DEFAULT_DEVICE)
        logging.info("loading classifier...done")

        ############################################################

        # Get an ordered list of 'base' and 'few' split classes, so they can be
        # accessed consistently by index.
        logging.info("preparing indexes...")
        bclass_ordered__seq = list(
            orig_train_data.info.class__set__per__split["many"]
            | orig_train_data.info.class__set__per__split["medium"]
        )
        fclass_ordered__seq = list(orig_train_data.info.class__set__per__split["few"])

        fbclass_ordered__vec = torch.tensor(
            fclass_ordered__seq + bclass_ordered__seq,
            dtype=torch.long,
            device=DEFAULT_DEVICE,
        )
        # Get positions for each class in the combined sequence of ordered 'few' and
        # 'base' classes. So, if `j = fbclass_ordered_idx__vec[i]`, then
        # `fbclass_ordered__vec[j] == i`; i.e., `fbclass_ordered_idx__vec` maps classes
        # to their positions in `fbclass_ordered__vec`.
        fbclass_ordered_idx__vec = torch.argsort(fbclass_ordered__vec)

        assert len(fbclass_ordered__vec.shape) == 1
        assert len(fbclass_ordered_idx__vec.shape) == 1
        assert all(
            fbclass_ordered__vec[fbclass_ordered_idx__vec[_i]] == _i
            for _i in range(orig_train_data.info.n_classes)
        )
        logging.info("preparing indexes...done")

        ############################################################

        # Load nearest neighbor weights.
        logging.info("loading nns...")
        nns_result = self.dataset.load_nns(
            self.training.nn_dist,
            self.training.n_neighbors,
            DEFAULT_DEVICE,
            generate=True,
            datagrp=orig_train_data,
        )

        # `nns_result.nn_clf__per__fclass` contains the nearest neighbor classifiers
        # for each 'few' split class. Use `fclass_ordered__seq` to order them.
        fclass_nn_clf_w_ordered__seq = [
            nns_result.nn_clf__per__fclass[_fclass].weight
            for _fclass in fclass_ordered__seq
        ]

        assert all(
            torch.allclose(full_clf.weight[_fclass], _fclass_nn_clf_w[0])
            for _fclass, _fclass_nn_clf_w in zip(
                fclass_ordered__seq, fclass_nn_clf_w_ordered__seq
            )
        )
        logging.info("loading nns...done")

        ############################################################

        # Set up model.

        logging.info("setting up model...")
        self.alphanet.set_sources(fclass_nn_clf_w_ordered__seq)

        # Create 'b's for the 'few' split classes, initialized to values from the
        # pre-trained classifier `full_clf`.
        _fclass_clf_b_init__vec = torch.tensor(
            [full_clf.bias[_fclass] for _fclass in fclass_ordered__seq]
        )

        # Extract 'base' split classifiers--(W, b) for 'base' split classes.
        # `bclass_ordered__seq` is used to ensure that the order of the classes is
        # consistent. So, `i`th row of `bclass_clf` corresponds to the classifier for
        # class `bclass_ordered__seq[i]`.
        _idx = torch.tensor(bclass_ordered__seq, device=DEFAULT_DEVICE)
        _bclass_clf_weight_init__mat = torch.index_select(full_clf.weight, 0, _idx)
        _bclass_clf_bias_init__vec = torch.index_select(full_clf.bias, 0, _idx)

        alphanet_classifier = AlphaNetClassifier(
            self.alphanet,
            len(bclass_ordered__seq),
            fbclass_ordered_idx__vec,
            _fclass_clf_b_init__vec,
            _bclass_clf_weight_init__mat,
            _bclass_clf_bias_init__vec,
            self.training.pred_scale,
        ).to(DEFAULT_DEVICE)

        # # Initialize weights.
        # _hact = self.alphanet.hact
        # if _hact == "relu" or _hact == "leaky_relu":
        #     _a = 0.01 if _hact == "leaky_relu" else 0

        #     def _w_init_fn(_w):
        #         torch.nn.init.kaiming_normal_(_w, a=_a, nonlinearity=_hact)

        # else:
        #     _w_init_fn = torch.nn.init.xavier_normal_
        # _b_init_fn = torch.nn.init.zeros_

        # for _param in self.alphanet.linear_layer__seq + [self.alphanet.conv_layer]:
        #     _w_init_fn(_param.weight)
        #     _b_init_fn(_param.bias)

        self.training.ptopt.set_weights(alphanet_classifier.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        logging.info("setting up model...done")

        ############################################################

        logging.info("setting up train data sampler...")
        train_data_sampler = self.training.sampler_builder.build(orig_train_data)
        logging.info("setting up train data sampler...done")
        logging.info("setting up val/test data loaders...")
        _val_test_data_loaders = [
            DataLoader(
                TensorDataset(
                    _orig_data.feat__mat, torch.tensor(_orig_data.label__seq)
                ),
                self.training.eval_batch_size or self.training.train_batch_size,
                shuffle=False,
            )
            for _orig_data in [orig_val_data, orig_test_data]
            if _orig_data is not None
        ]
        if orig_val_data is not None:
            val_data_loader, test_data_loader = _val_test_data_loaders
        else:
            val_data_loader = None
            test_data_loader = _val_test_data_loaders[0]
        logging.info("setting up val/test data loaders...done")

        def get_train_data_loader():
            logging.info("generating training data...")
            _dataset = train_data_sampler.get_dataset()
            _lo = DataLoader(_dataset, self.training.train_batch_size, shuffle=True)
            logging.info("generating training data...done")
            return _lo

        logging.info("setting up training...")
        train_result = TrainResult()
        train_result.train_data_info = orig_train_data.info
        train_result.val_data_info = (
            orig_val_data.info if orig_val_data is not None else None
        )
        train_result.nn_info = nns_result
        train_result.alphanet = self.alphanet
        train_result.alphanet_source__mat__seq = fclass_nn_clf_w_ordered__seq
        train_result.fbclass_ordered_idx__vec = fbclass_ordered_idx__vec
        train_result.training_config = self.training
        train_result.epoch_data__seq = []
        train_result.best_epoch = None

        train_engine = ignite.engine.create_supervised_trainer(
            alphanet_classifier, self.training.ptopt.optimizer, loss_fn, DEFAULT_DEVICE
        )
        eval_engine = ignite.engine.create_supervised_evaluator(
            alphanet_classifier,
            {
                "accuracy": ignite.metrics.Accuracy(),
                "top-5 accuracy": ignite.metrics.TopKCategoricalAccuracy(5),
                "f1": ignite.metrics.Fbeta(1),
            },
            DEFAULT_DEVICE,
        )

        ignite.contrib.handlers.ProgressBar(persist=True).attach(
            train_engine, output_transform=lambda _x: {"loss": _x}
        )

        @train_engine.on(ignite.engine.Events.EPOCH_STARTED)
        def _(engine: ignite.engine.Engine):
            if engine.state.epoch > 1:
                logging.info(
                    "recreating training dataset with new 'base' split samples"
                )
                engine.set_data(get_train_data_loader())
            engine.state.my_mean_batch_loss__seq = []

        @train_engine.on(ignite.engine.Events.ITERATION_COMPLETED)
        def _(engine: ignite.engine.Engine):
            engine.state.my_mean_batch_loss__seq.append(engine.state.output)
            self.training.tb_logs.writer.add_scalar(
                "batch_loss", engine.state.output, engine.state.iteration
            )

        @train_engine.on(ignite.engine.Events.EPOCH_COMPLETED)
        def _(engine: ignite.engine.Engine):
            epoch_data = EpochData()
            epoch_data.mean_batch_loss__seq = engine.state.my_mean_batch_loss__seq
            epoch_data.n_train__per__class = Counter(
                engine.state.dataloader.dataset.tensors[1].tolist()  # type: ignore
            )
            epoch_data.n_train__per__split = {}
            for (
                _split,
                _split_class__set,
            ) in orig_train_data.info.class__set__per__split.items():
                epoch_data.n_train__per__split[_split] = sum(
                    epoch_data.n_train__per__class[_class]
                    for _class in _split_class__set
                )

            # Log alphas.
            with torch.no_grad():
                _alpha__mat = torch.stack(
                    [
                        self.alphanet.get_target_alphas(_i)
                        for _i in range(self.alphanet.n_targets)
                    ]
                )
            assert _alpha__mat.shape == (
                self.alphanet.n_targets,
                self.alphanet.n_sources_vecs,
            )
            (
                epoch_data.mean_alpha__vec,
                epoch_data.min_alpha__vec,
                epoch_data.max_alpha__vec,
            ) = log_alphas(_alpha__mat, self.training.tb_logs, engine.state.epoch)

            # Log val metrics.
            if val_data_loader is not None:
                eval_engine.run(val_data_loader)
                epoch_data.val_metrics = eval_engine.state.metrics
                epoch_data.val_acc__per__split = (
                    eval_engine.state.my_accuracy__per__split
                )
                epoch_data.val_acc__per__class = (
                    eval_engine.state.my_accuracy__per__class
                )

                log_metrics(
                    epoch_data.val_metrics,
                    epoch_data.val_acc__per__split,  # type: ignore
                    self.training.tb_logs,
                    "validation",
                    engine.state.epoch,
                )
                # Update best model.
                # Note: `engine.state.epoch` starts at 1.
                if engine.state.epoch >= self.training.min_epochs:
                    _best_val_acc = (
                        -1
                        if train_result.best_epoch is None
                        else train_result.epoch_data__seq[  # type: ignore
                            train_result.best_epoch
                        ].val_metrics["accuracy"]
                    )
                    if epoch_data.val_metrics["accuracy"] > _best_val_acc:
                        logging.info(
                            "epoch %d is new best with val accuracy: %.2g",
                            engine.state.epoch,
                            epoch_data.val_metrics["accuracy"],
                        )
                        train_result.best_epoch = engine.state.epoch - 1
                        train_result.best_alphanet_classifier_state_dict = deepcopy(
                            alphanet_classifier.state_dict()
                        )
            else:
                epoch_data.val_metrics = epoch_data.val_acc__per__split = None

            # Add epoch data to `train_result`.
            train_result.epoch_data__seq.append(epoch_data)
            if self.training.ptopt.lr_scheduler is not None:
                self.training.ptopt.lr_scheduler.step()

        @eval_engine.on(ignite.engine.Events.STARTED)
        def _(engine: ignite.engine.Engine):
            engine.state.my_y__seq = []
            engine.state.my_yhat__seq = []
            engine.state.my_accuracy__per__split = None
            engine.state.my_accuracy__per__class = None

        @eval_engine.on(ignite.engine.Events.ITERATION_COMPLETED)
        def _(engine: ignite.engine.Engine):
            _scores__batch, _y__batch = engine.state.output  # type: ignore
            _yhat__batch = torch.argmax(_scores__batch, dim=1)
            engine.state.my_y__seq.extend(_y__batch.tolist())
            engine.state.my_yhat__seq.extend(_yhat__batch.tolist())

        @eval_engine.on(ignite.engine.Events.COMPLETED)
        def _(engine: ignite.engine.Engine):
            _correct_preds__per__split: Dict[str, int] = defaultdict(int)
            _total_preds__per__split: Dict[str, int] = defaultdict(int)
            _correct_preds__per__class: Dict[int, int] = defaultdict(int)
            _total_preds__per__class: Dict[int, int] = defaultdict(int)

            for _y_i, _yhat_i in zip(engine.state.my_y__seq, engine.state.my_yhat__seq):
                _split = next(
                    _k
                    for _k, _v in orig_train_data.info.class__set__per__split.items()
                    if _y_i in _v
                )
                _correct = int(_y_i == _yhat_i)
                _correct_preds__per__split[_split] += _correct
                _total_preds__per__split[_split] += 1
                _correct_preds__per__class[_y_i] += _correct
                _total_preds__per__class[_y_i] += 1

            engine.state.my_accuracy__per__split = {
                _split: float(
                    _correct_preds__per__split[_split]
                    / _total_preds__per__split[_split]
                )
                for _split in _total_preds__per__split
            }
            engine.state.my_accuracy__per__class = {
                _class: float(
                    _correct_preds__per__class[_class]
                    / _total_preds__per__class[_class]
                )
                for _class in _total_preds__per__class
            }

            assert torch.isclose(
                torch.tensor(
                    [sum(_correct_preds__per__split.values())], dtype=torch.float
                ),
                torch.tensor(
                    [engine.state.metrics["accuracy"] * len(engine.state.my_y__seq)]
                ),
            )
            _y_counter: Dict[int, int] = Counter(engine.state.my_y__seq)
            assert all(
                _total_preds__per__class[_class] == _y_counter[_class]
                for _class in _y_counter
            )

        @train_engine.on(ignite.engine.Events.COMPLETED)
        def _():
            # Load state from epoch with the best overall validation accuracy.
            if self.training.train_epochs > 0 and val_data_loader is not None:
                logging.info(
                    "best epoch by overall val accuracy: %d",
                    train_result.best_epoch + 1,
                )
                alphanet_classifier.load_state_dict(
                    train_result.best_alphanet_classifier_state_dict
                )
            else:
                assert train_result.best_epoch is None
                train_result.best_alphanet_classifier_state_dict = deepcopy(
                    alphanet_classifier.state_dict()
                )

            eval_engine.run(test_data_loader)
            train_result.test_metrics = eval_engine.state.metrics
            train_result.test_acc__per__split = (
                eval_engine.state.my_accuracy__per__split
            )
            train_result.test_acc__per__class = (
                eval_engine.state.my_accuracy__per__class
            )
            log_metrics(
                train_result.test_metrics,
                train_result.test_acc__per__split,
                self.training.tb_logs,
                "test",
            )

        # Set up checkpointing.
        if self.ckpt_dir is not None:
            _ckpt_data = {
                "model": alphanet_classifier,
                "optimizer": self.training.ptopt.optimizer,
                "lr_scheduler": self.training.ptopt.lr_scheduler,
                "trainer": train_engine,
                "train_result": train_result,
            }
            _ckpt_handler = ignite.handlers.ModelCheckpoint(
                self.ckpt_dir,
                n_saved=self.n_ckpt,
                require_empty=False,
                global_step_transform=lambda *_: train_engine.state.epoch,
            )
            train_engine.add_event_handler(
                ignite.engine.Events.EPOCH_COMPLETED, _ckpt_handler, _ckpt_data
            )

            # Load exisiting checkpoint if available.
            if self.load_from_ckpt:
                _ckpt_files = list(self.ckpt_dir.glob("checkpoint_*.pt"))
                if not _ckpt_files:
                    logging.info("no previous checkpoint found")
                else:
                    _latest_ckpt_file = max(
                        _ckpt_files, key=lambda _f: int(_f.stem.split("_")[1])
                    )
                    logging.warning("loading from checkpoint: '%s'", _latest_ckpt_file)
                    _ckpt_handler.load_objects(_ckpt_data, _latest_ckpt_file)

        logging.info("setting up training...done")
        logging.info("training...")
        train_engine.run(get_train_data_loader(), max_epochs=self.training.train_epochs)
        self.training.tb_logs.writer.close()
        logging.info("training...done")
        if self.save_file is not None:
            logging.info("saving results...")
            torch.save(train_result.as_dict(recursive=True), self.save_file)
            self.save_file.close()
            logging.info("saving results...done")
        return train_result
