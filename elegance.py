# coding:utf-8

import argparse
import time

try:
    import util
except:
    from . import util

plugin_choices = util.get_plugin_name("plugin")
parser = argparse.ArgumentParser(description='elegance')

parser.add_argument("plugin", type=str,
                    choices=plugin_choices,
                    help='the name of plugin you need to use (options:{})'.format(plugin_choices))

parser.add_argument('input', type=str,
                    help='the path of image you need to process')

parser.add_argument('--output', type=str,
                    default="output.png",
                    help='the path of image you  processed(default=output.jpg)')


parser.add_argument("-p", '--params', type=str,
                    nargs="*",
                    default=None,
                    dest="params",
                    help='the params of plugin')

parser.add_argument("-split", type=int,
                    default=util.np.inf,
                    dest="split",
                    help='the size of  image need to split,like 256  default np.inf')

parser.add_argument("-c", '--combine',
                    action="store_true",
                    dest="combine",
                    help='whether combine the origin image and  processed image')

parser.add_argument('-dir',
                    action="store_true",
                    dest="is_dir",
                    help='whether dir the path your input')


def main(args):

    print("[INFO]USING PLUGIN:", args.plugin)
    params = util.str2dict(args.params)
    generate, preprocess, postprocess = util.load_plugin(args.plugin, params)
    print("[INFO]Model Struct:")
    print(generate)
    print("[INFO]Predict...")
    generate.eval()
    t1 = time.time()
    if args.is_dir:
        util.predict_dir(generate, args.input, args.output,split_size=args.split, preprocess=preprocess, combine=args.combine,
                         postprocess=postprocess)
    else:
        util.predict_save(generate, args.input, args.output,split_size=args.split, preprocess=preprocess, combine=args.combine,
                          postprocess=postprocess)
    print("[INFO]Finish.Cost time:{:.5f}s".format(time.time() - t1))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
