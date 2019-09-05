import os


def run(opt):
    chpt_path = opt.output_path + str(opt.id) + 'checkpoint.txt'
    dirname = os.path.dirname(chpt_path)
    if not os.path.exists(dirname):
        print("file does not exist.")
        return 0




