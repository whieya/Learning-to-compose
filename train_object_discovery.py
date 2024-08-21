from pathlib import Path
import yaml
from omegaconf import OmegaConf
from data.datasets import make_dataloaders
from utils.utils import (
    set_all_seeds,
)
from accelerate.utils import set_seed
from models.slot_attention.model import Comp_Model, Slate
from models.slot_attention.trainer import SlotAttentionTrainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # training configs
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--diffusion_path', type=str, default='test')
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--lr_lsd', type=float, default=1e-4)
    parser.add_argument('--lr_sa', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--use_exp_decay', action='store_true', default=False)
    parser.add_argument('--exp_decay_rate', type=float, default=0.5)
    parser.add_argument('--use_warmup_lr', action='store_true', default=False)
    parser.add_argument('--exp_decay_steps', type=int, default=20000)
    parser.add_argument('--load_ckpt_path', type=str, default=None)
    parser.add_argument('--use_accel', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--eval_miou', action='store_true', default=False)


    # data
    parser.add_argument('--dataset_name', type=str, default='clevr')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--subset_portion', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=128)

    # logging
    parser.add_argument('--log_method', type=str, default='wandb')
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--log_n_imgs', type=int, default=4)

    # loss weights
    parser.add_argument('--lambda_composition', type=float, default=0.0)
    parser.add_argument('--lambda_oneshot', type=float, default=0.0)
    parser.add_argument('--lambda_mask_reg', type=float, default=0.0)
    parser.add_argument('--lambda_slot_diffusion', type=float, default=0.0)

    # sds hyperparameters
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--ddim_steps', type=int, default=10)
    parser.add_argument('--scale_latent', type=float, default=1.0)  # legacy 
    parser.add_argument('--diff_dim', type=int, default=192)

    #slot attention
    parser.add_argument('--slot_encode_RGB', action='store_true', default=False)
    parser.add_argument('--cnn_enc_type', type=str, default='unet')
    parser.add_argument('--latent_size', type=int, default=192)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--mlp_size', type=int, default=192)
    parser.add_argument('--num_slots', type=int, default=0)
    parser.add_argument('--attention_iters', type=int, default=7)
    parser.add_argument('--cnn_downsample', type=int, default=1)
    parser.add_argument('--share_slot_init', action='store_true', default=False)

    #transformer 
    parser.add_argument('--num_dec_blocks', type=int, default=2)
    parser.add_argument('--d_tf', type=int, default=192)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--autoregressive', action='store_true', default=False)

    return parser.parse_args()

def main(config):
    curr_dir = Path.cwd()

    # retrieve data root
    with open('data/data_paths.yaml', 'r') as f:
        path_dict = yaml.safe_load(f)
        config.dataset_path = path_dict[config.dataset_name]

    # set dataset size
    if config.debug:
        config.exp_name='debug'
        config.data_sizes = [160, 160, 160]   
    else:
        if config.dataset_name == 'clevr':
            config.data_sizes = [90000, 5000, 5000]   
        else:
            train_full_len = 40000
            train_data_size = int(config.subset_portion * train_full_len)
            config.data_sizes = [train_data_size, 5000, 5000]   

    assert len(config.data_sizes) == 3, "Need a train/validation/test split."

    train_config_path = curr_dir / "train_config.yaml"

    # Check if previous run exists
    load_checkpoint = False
    if train_config_path.exists():  # previous run is found

        if not config.allow_resume:
            raise FileExistsError(
                f"Previous run found in '{curr_dir}' but flag 'allow_resume' is False"
            )

        # Load config and check it matches
        with open(train_config_path) as configfile:
            prev_config = OmegaConf.load(configfile)
        config.uuid = prev_config.uuid  # use original uuid from previous train config
        ignore_list = [
            "allow_resume",
            "trainer.steps",
            "trainer.checkpoint_steps",
            "trainer.logweights_steps",
            "trainer.logimages_steps",
            "trainer.logloss_steps",
            "device",
            "num_workers",
            "batch_size",
            # added
            "exp_name",
            "trainer.num_slots",
            "trainer.downsample",
            "trainer.optimizer_config.lr",
            "trainer.use_exp_decay", 
            ]

        load_checkpoint = True

    set_all_seeds(config.seed)

    print("----- Creating model -----")
    # main model for composing slots
    model = Comp_Model(config,
                       resolution=config.image_size, num_slots=config.num_slots,
                       ckpt=config.diffusion_path,
                       log_n_imgs=config.log_n_imgs, 
                       dataset_name=config.dataset_name,
                       max_steps=config.max_steps,
                       ddim_steps=config.ddim_steps,
                       scale_latent=config.scale_latent,
                       slot_dim=config.latent_size,
                       diff_dim=config.diff_dim,
                       share_slot_init=config.share_slot_init,
                       )

    # slot encoder
    model_sa = Slate(
            image_size=config.image_size,
            latent_size=config.latent_size, 
            input_channels=3 if config.slot_encode_RGB else 4,
            num_slots=config.num_slots,
            mlp_size=config.mlp_size,
            attention_iters = config.attention_iters,
            slot_encode_RGB=config.slot_encode_RGB,
            num_dec_blocks=config.num_dec_blocks,
            d_tf=config.d_tf,
            num_heads=config.num_heads,
            autoregressive=config.autoregressive,
            cnn_enc_type=config.cnn_enc_type,
            cnn_downsample=config.cnn_downsample,
            )

    model.init_SA_module(model_sa)

    print("Creating data loaders")
    dataloaders = make_dataloaders(
        dataset_name=config.dataset_name,
        dataset_path=config.dataset_path,
        data_sizes=config.data_sizes[:2],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory="cuda" in config.device and config.num_workers > 0,
        eval_mode=config.eval_miou,
        steps=config.steps,
    )

    print("Creating trainer")
    trainer = SlotAttentionTrainer(
        config=config,
        device=config.device,
        steps=config.steps,
        clip_grad_norm=None,
        use_exp_decay=config.use_exp_decay,
        exp_decay_rate=config.exp_decay_rate,
        exp_decay_steps=config.exp_decay_steps,
        use_warmup_lr=config.use_warmup_lr,
        debug=False,
        working_dir=curr_dir,
        num_slots=config.num_slots,
        warmup_steps=config.warmup_steps,
        lambda_composition=config.lambda_composition,
        lambda_oneshot=config.lambda_oneshot,
        lambda_mask_reg=config.lambda_mask_reg,
        lambda_slot_diffusion=config.lambda_slot_diffusion,
        )

    # set upt the model
    trainer.setup(model, model_sa, dataloaders, config.load_ckpt_path)

    print("================== Training starts ==================")
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)

