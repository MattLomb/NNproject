from utils import biggan_norm, biggan_denorm

configs = dict(
    StyleGAN2_ffhq_d = dict(
        task = "txt2img",
        dim_z = 512,
        #latent
        #model
        use_discriminator = True,
        norm = biggan_norm,
        denorm = biggan_denorm,
        problem_args = dict(
            n_var = 512,
            n_obj = 2,
            n_constr = 512,
            xl = -10,
            xu = 10,
        )        
    )
)

def get_config(name):
    return configs[name]
