import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset, DuetDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton
from eval.lma_similarity import compute_similarity
from dataset.quaternion import ax_from_6v


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        duet=False,
        guidance_weight_music=2,
        guidance_weight_lead=None,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = feature_type == "baseline"

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4
        self.duet = duet

        music_feature_dim = 35 if use_baseline_feats else 4800
        if duet:
            # 双人模式：cond = cat([主舞动作(151), 音乐特征])
            # 与 DuetDataset.__getitem__ 里的拼接顺序一致
            feature_dim = music_feature_dim + repr_dim  # 4951 (jukebox) 或 186 (baseline)
        else:
            # 单人模式：cond = 音乐特征（原始行为，兼容预训练 checkpoint）
            feature_dim = music_feature_dim  # 4800 (jukebox) 或 35 (baseline)

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        self.start_epoch = 1  # overwritten below if resuming from a checkpoint

        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]
            # Resume training from the epoch after the one that was saved
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
                print(f"[Checkpoint] Resuming from epoch {self.start_epoch}")

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
            lead_dim=repr_dim if duet else 0,
        )

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=guidance_weight_music,
            guidance_weight_lead=guidance_weight_lead,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            state_dict = maybe_wrap(
                checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                num_processes,
            )
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            # strict=False allows cross-mode fine-tuning (solo → duet or duet → solo):
            # layers whose shape changed (e.g. cond_projection when feature_dim differs)
            # are skipped and left at their random initialization.
            if missing or unexpected:
                print(f"[Checkpoint] Partial load — {len(missing)} missing keys, "
                      f"{len(unexpected)} unexpected keys.")
                print(f"  Skipped (shape mismatch / new layers): {missing[:5]}"
                      f"{'...' if len(missing) > 5 else ''}")

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        # 根据模式选择数据集类和缓存文件名，两套互不干扰
        if self.duet:
            DatasetClass = DuetDataset
            cache_prefix = f"duet_{opt.feature_type}"
        else:
            DatasetClass = AISTPPDataset
            cache_prefix = f"solo_{opt.feature_type}"

        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"{cache_prefix}_train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"{cache_prefix}_test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = DatasetClass(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                feature_type=opt.feature_type,
                force_reload=opt.force_reload,
            )
            test_dataset = DatasetClass(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                feature_type=opt.feature_type,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        # jukebox features are 2.75 MB/file; cap workers to avoid I/O contention
        pin = torch.cuda.is_available()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(self.start_epoch, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            # train
            self.train()
            for step, (x, cond, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                # Independent conditioning dropout for duet mode.
                # Randomly zero each portion of the cond vector separately so
                # the model learns all four conditioning regimes:
                #   (music, lead), (music, ∅), (∅, lead), (∅, ∅)
                # The full-null (∅, ∅) case is also covered by GaussianDiffusion's
                # cond_drop_prob=0.25 (which replaces the projected cond with a
                # learned null embedding).  The input-level zeros here train the
                # model to handle partial conditioning at inference time.
                if self.duet:
                    cond = cond.clone()
                    B = cond.shape[0]
                    if opt.drop_prob_lead > 0:
                        mask_lead = (torch.rand(B, 1, 1, device=cond.device)
                                     < opt.drop_prob_lead)
                        cond[:, :, :self.repr_dim] = cond[:, :, :self.repr_dim].masked_fill(
                            mask_lead, 0.0
                        )
                    if opt.drop_prob_music > 0:
                        mask_music = (torch.rand(B, 1, 1, device=cond.device)
                                      < opt.drop_prob_music)
                        cond[:, :, self.repr_dim:] = cond[:, :, self.repr_dim:].masked_fill(
                            mask_music, 0.0
                        )

                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # compute val loss
                    val_loss = 0
                    with torch.no_grad():
                        for val_step, (xv, condv, _, _) in enumerate(test_data_loader):
                            xv = xv.to(self.accelerator.device)
                            condv = condv.to(self.accelerator.device)
                            _, (lv, _, _, _) = self.diffusion(xv, condv, t_override=None)
                            val_loss += lv.detach().cpu().numpy()
                    val_loss /= len(test_data_loader)
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "Val Loss": val_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                    }
                    print(
                        f"[Epoch {epoch}] train={avg_loss:.4f}  val={val_loss:.4f}"
                        f"  v={avg_vloss:.4f}  fk={avg_fkloss:.4f}  foot={avg_footloss:.4f}"
                    )
                    wandb.log(log_dict)
                    ckpt = {
                        "epoch": epoch,   # saved so training can resume from here
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename, wavnames) = next(iter(test_data_loader))
                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                        duet=self.duet,
                        repr_dim=self.repr_dim,
                    )

                    # ── LMA 评估（仅 duet 模式）────────────────────────────
                    if self.duet:
                        lma_score = self._eval_lma(test_data_loader, n_pairs=10)
                        print(f"[Epoch {epoch}] LMA total={lma_score:.4f}  "
                              f"(random≈0.500, real-pair≈0.930)")
                        log_dict["LMA"] = lma_score
                        wandb.log(log_dict)
                    # ──────────────────────────────────────────────────────

                    print(f"[MODEL SAVED at Epoch {epoch}]")
                    self.train()

            # Save latest.pt more frequently so preempted jobs lose at most
            # save_latest_interval epochs. Overwrites the same file every time.
            if (epoch % opt.save_latest_interval) == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    latest_ckpt = {
                        "epoch": epoch,
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(latest_ckpt, os.path.join(wdir, "latest.pt"))

        if self.accelerator.is_main_process:
            wandb.run.finish()

    @torch.no_grad()
    def _eval_lma(self, test_loader, n_pairs: int = 10) -> float:
        """推理 n_pairs 对测试样本，计算 LMA 主舞-伴舞相似度均值。"""
        scores = []
        smpl = self.diffusion.smpl
        for i, (x, cond, _, _) in enumerate(test_loader):
            if i >= n_pairs:
                break
            cond = cond[:1].to(self.accelerator.device)  # 每次取 1 对
            shape = (1, self.horizon, self.repr_dim)

            # 生成伴舞
            dev = self.accelerator.device
            samples = self.diffusion.ddim_sample(shape, cond).detach().cpu()
            samples = self.normalizer.unnormalize(samples)
            _, samples = torch.split(samples, (4, self.repr_dim - 4), dim=2)
            pos_f = samples[:, :, :3].to(dev)
            q_f   = ax_from_6v(samples[:, :, 3:].reshape(1, self.horizon, 24, 6).to(dev))
            follower_joints = smpl.forward(q_f, pos_f)[0].detach().cpu().numpy()

            # 从 cond 提取主舞关节位置
            lead_raw = cond[:, :, :self.repr_dim]           # [1, T, 151]
            lead_raw = self.normalizer.unnormalize(lead_raw.cpu())
            _, lead_pose = torch.split(lead_raw, (4, self.repr_dim - 4), dim=2)
            pos_l = lead_pose[:, :, :3].to(dev)
            q_l   = ax_from_6v(lead_pose[:, :, 3:].reshape(1, self.horizon, 24, 6).to(dev))
            lead_joints = smpl.forward(q_l, pos_l)[0].detach().cpu().numpy()

            result = compute_similarity(lead_joints, follower_joints)
            scores.append(result["total_score"])

        return float(sum(scores) / len(scores)) if scores else 0.0

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render,
            duet=self.duet,
            repr_dim=self.repr_dim,
        )
