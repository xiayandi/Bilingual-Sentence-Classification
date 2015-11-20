#!/usr/bin/env bash
java -cp ../../bin/coreNLP/stanford-corenlp-3.5.2.jar\
:../../bin/coreNLP/stanford-chinese-corenlp-2015-04-20-models.jar\
:../../bin/coreNLP/xom.jar\
:../../bin/coreNLP/joda-time.jar\
:../../bin/coreNLP/jollyday.jar\
:../../bin/coreNLP/ejml-0.23.jar \
-Xmx12g edu.stanford.nlp.pipeline.StanfordCoreNLP \
-props ../../bin/coreNLP/chinese_properties -file $1 \
-outputDirectory $2 \
