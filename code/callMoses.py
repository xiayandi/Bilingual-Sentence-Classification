__author__ = 'yandixia'
import subprocess

moses_dir = '/users/chenli/demo/tool/moses/'
pair_file = 'pair_file'
command_train_dir = moses_dir + "scripts/training/"
command_token_dir = moses_dir + "scripts/tokenizer/"
corpus_prefix = "abb"

work_dir = para['work_dir']
lm_file = para['char_lm_file']
corpus_dir = work_dir + "corpus/"

cmd = ["rm", "-rf", work_dir]
subprocess.call(cmd)

cmd = ["mkdir", work_dir]
subprocess.call(cmd)

cmd = ["mkdir", work_dir + "corpus"]
subprocess.call(cmd)

cmd = [command_train_dir + "clean-corpus-n.perl", corpus_dir + corpus_prefix, "fr", "en",
       corpus_dir + corpus_prefix + ".clean", "1", "200"]
subprocess.call(cmd)

cmd = [command_train_dir + "train-model.perl", "--root-dir", work_dir, "--f", "fr", "--e", "en", "--corpus",
       corpus_dir + corpus_prefix + ".clean", "-lm", "0:5:" + lm_file, "--giza-option", "m1=8,m2=0,mh=8,m3=6,m4=0"]
subprocess.call(cmd)
