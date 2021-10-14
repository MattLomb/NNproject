from utils import biggan_norm, biggan_denorm

configs = dict(
    StyleGAN2_ffhq_d=dict(
        task="txt2img",
        dim_z=512,
        batch_size=1,   # 1 Because in tf on CPU StyleGan works only with batchsize = 1
        pop_size=4,     # Size of x in _evaluation
        algorithm="nsga2",
        # latent
        # model
        use_discriminator=True,
        norm=biggan_norm,
        denorm=biggan_denorm,
        problem_args=dict(
            n_var=512,  # X in _evaluation has size of (pop_size, n_var)
            n_obj=2,
            n_constr=512,
            xl=-10,
            xu=10,
        )
    ),
    StyleGAN2_ffhq_nod = dict(
        task="txt2img",
        dim_z=512,
        batch_size=1,   # 1 Because in tf on CPU StyleGan works only with batchsize = 1
        pop_size=8,     # Size of x in _evaluation
        algorithm="ga",
        use_discriminator=False,
        norm=biggan_norm,
        denorm=biggan_denorm,
        problem_args=dict(
            n_var=512,  # X in _evaluation has size of (pop_size, n_var)
            n_obj=1,
            n_constr=512,
            xl=-10,
            xu=10,
        )
    )
)


def get_config(name):
    return configs[name]
