import os
import re
import pickle
import argparse
import io
import requests
import utils


pretrained_model_urls = {
    'car-config-e':                    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl',
    'car-config-f':                    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl',
    'cat-config-f':                    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl',
    'church-config-f':                 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl',
    'ffhq-config-e':                   'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-e.pkl',
    'ffhq-config-f':                   'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
    'horse-config-f':                  'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-f.pkl',
    'car-config-e-Gorig-Dorig':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl',
    'car-config-e-Gorig-Dresnet':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl',
    'car-config-e-Gorig-Dskip':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl',
    'car-config-e-Gresnet-Dorig':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl',
    'car-config-e-Gresnet-Dresnet':    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl',
    'car-config-e-Gresnet-Dskip':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl',
    'car-config-e-Gskip-Dorig':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl',
    'car-config-e-Gskip-Dresnet':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl',
    'car-config-e-Gskip-Dskip':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl',
    'ffhq-config-e-Gorig-Dorig':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl',
    'ffhq-config-e-Gorig-Dresnet':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl',
    'ffhq-config-e-Gorig-Dskip':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl',
    'ffhq-config-e-Gresnet-Dorig':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl',
    'ffhq-config-e-Gresnet-Dresnet':   'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl',
    'ffhq-config-e-Gresnet-Dskip':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl',
    'ffhq-config-e-Gskip-Dorig':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl',
    'ffhq-config-e-Gskip-Dresnet':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
    'ffhq-config-e-Gskip-Dskip':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl',
}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return utils.AttributeDict
        return super(Unpickler, self).find_class(module, name)


def load_tf_models_file(fpath):
    with open(fpath, 'rb') as fp:
        return Unpickler(fp).load()


def load_tf_models_url(url):
    print('Downloading file {}...'.format(url))
    with requests.Session() as session:
        with session.get(url) as ret:
            fp = io.BytesIO(ret.content)
            return Unpickler(fp).load()



_PERMITTED_MODELS = ['G_main', 'G_mapping', 'G_synthesis_stylegan2', 'D_stylegan2', 'D_main', 'G_synthesis']

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Convert tensorflow stylegan2 model to pytorch.',
        epilog='Pretrained models that can be downloaded:\n{}'.format(
            '\n'.join(pretrained_model_urls.keys()))
    )

    parser.add_argument(
        '-i',
        '--input',
        help='File path to pickled tensorflow models.',
        type=str,
        default=None,
    )

    parser.add_argument(
        '-d',
        '--download',
        help='Download the specified pretrained model. Use --help for info on available models.',
        type=str,
        default=None,
    )

    parser.add_argument(
        '-o',
        '--output',
        help='One or more output file paths. Alternatively a directory path ' + \
            'where all models will be saved. Default: current directory',
        type=str,
        nargs='*',
        default=['.'],
    )

    return parser


def main():
    args = get_arg_parser().parse_args()
    assert bool(args.input) != bool(args.download), \
        'Incorrect input format. Can only take either one ' + \
        'input filepath to a pickled tensorflow model or ' + \
        'a model name to download, but not both at the same ' + \
        'time or none at all.'
    if args.input:
        unpickled = load_tf_models_file(args.input)
    else:
        assert args.download in pretrained_model_urls.keys(), \
            'Unknown model {}. Use --help for list of models.'.format(args.download)
        unpickled = load_tf_models_url(pretrained_model_urls[args.download])
    if not isinstance(unpickled, (tuple, list)):
        unpickled = [unpickled]

    for tf_state in unpickled:
        print(tf_state.components)

    print('Converting tensorflow models and saving them...')
    '''
    converted = [convert_from_tf(tf_state) for tf_state in unpickled]
    if len(args.output) == 1 and (os.path.isdir(args.output[0]) or not os.path.splitext(args.output[0])[-1]):
        if not os.path.exists(args.output[0]):
            os.makedirs(args.output[0])
        for tf_state, torch_model in zip(unpickled, converted):
            torch_model.save(os.path.join(args.output[0], tf_state['name'] + '.pth'))
    else:
        assert len(args.output) == len(converted), 'Found {} models '.format(len(converted)) + \
            'in pickled file but only {} output paths were given.'.format(len(args.output))
        for out_path, torch_model in zip(args.output, converted):
            torch_model.save(out_path)
    '''
    print('Done!')



if __name__ == '__main__':
    main()
