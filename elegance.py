# coding:utf-8

import argparse

try:
    import util
except:
    from . import util

plugin_choices = util.get_plugin_name("plugin")
parser = argparse.ArgumentParser(description='elegance')
parser.add_argument('--input', type=str,
                    default="input.jpg",
                    help='the path of image you need to process(default=input.jpg)')

parser.add_argument('--output', type=str,
                    default="output.jpg",
                    help='the path of image you  processed(default=output.jpg)')

parser.add_argument("-p", '--plugin', type=str,
                    default="blur",
                    dest="plugin",
                    choices=plugin_choices,
                    help='the name of plugin you need to use (options:{} default=blur)'.format(plugin_choices))
parser.add_argument("-c", '--combine',
                    action="store_true",
                    dest="combine",
                    help='whether combine the origin image and  processed image')


def main(args):
    print("INFO:USING PLUGIN:", args.plugin)
    generate, preprocess = util.get_net_and_preprocess(args.plugin)
    print("INFO:Model Struct:")
    print(generate)
    print("INFO:Predict...")
    util.predict_save(generate, args.input, args.output, preprocess=preprocess, combine=args.combine)
    print("INFO:Finish.")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
