#!/usr/bin/env bash
java -cp ../../bin/coreNLP/stanford-corenlp-3.5.2.jar\
:../../bin/coreNLP/stanford-corenlp-3.5.2-models.jar\
:../../bin/coreNLP/xom.jar\
:../../bin/coreNLP/joda-time.jar\
:../../bin/coreNLP/jollyday.jar\
:../../bin/coreNLP/ejml-0.23.jar \
-Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP \
-props ../../bin/coreNLP/english_properties -file $1 \
-outputDirectory ../exp
