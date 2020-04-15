#!/usr/bin/env python
import sys
import os
import shutil

sys.path.append('scripts')

from gen_exper_desc.common_gen_desc import *
from data_convert.convert_common import *

MODEL_SRC_PATH='scripts/exper/sample_exper_desc/one_feat.model'
MODEL_DST_REL_PATH='models'
MODEL_DST_NAME = 'one_feat.model'

class ParserBM25Coeff(BaseParser):
  def initAddArgs(self):
    self.parser.add_argument('--field_name',
                             metavar='BM25 field name',
                             help='a field for BM25 score', required=True)

parser = ParserBM25Coeff('BM25 tuning param generator')
parser.parseArgs()

args = parser.getArgs()
args_var = vars(args)

fieldName = args.field_name
outdir = args_var[OUT_DIR_PARAM]
outModelDir = os.path.join(outdir, MODEL_DST_REL_PATH)
if not os.path.exists(outModelDir):
  os.makedirs(outModelDir)
shutil.copyfile(MODEL_SRC_PATH, os.path.join(outModelDir, MODEL_DST_NAME))

modelRelName = os.path.join(args_var[REL_DESC_PATH_PARAM], MODEL_DST_REL_PATH, MODEL_DST_NAME)

class ExtrBM25JsonGEN:
  def __init__(self, fieldName):
    self.fieldName = fieldName

  def __call__(self):
    for bi in range(7):
      for k1i in range(7):
        b = 0.3 + 0.15 * bi
        k1 = 0.4 + 0.2 * k1i
        bstr = '%g' % b
        k1str = '%g' % k1
        fid = 'bm25tune_%s_k1=%s_b=%s' % (fieldName, k1str, bstr)

        jsonDesc = {
                    "extractors" : [
                    {
                      "type" : "TFIDFSimilarity",
                      "params" : {
                        "indexFieldName" : self.fieldName,
                        "similType" : "bm25",
                        "k1"        : k1str,
                        "b"         : bstr
                      }
                    }
                    ]
                    }

        # Test only is true, b/c there's nothing to train, but we need to provide the model
        testOnly=True
        yield fid, jsonDesc, testOnly, modelRelName 

prefix = f'bm25tune_{fieldName}'
genRerankDescriptors(args, ExtrBM25JsonGEN(fieldName),
                     f'{prefix}.json', prefix)


